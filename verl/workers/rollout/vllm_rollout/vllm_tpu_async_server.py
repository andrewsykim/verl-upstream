# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vLLM rollout on Cloud TPU.

vLLM's TPU backend differs from the CUDA one in ways that reach into placement,
so it gets its own server/replica pair rather than a pile of conditionals in the
CUDA path:

* **The engine owns the chips.** On TPU, vLLM only supports its Ray executor
  (``vllm_torchtpu``'s TPU worker requires a ``tcp://`` rendezvous, which vLLM's
  multiproc executor never produces). That executor spawns one Ray actor per chip,
  each claiming the accelerator resource, so verl's own rollout workers must hold
  none and instead be pinned beside the engine.

* **The executor assumes it owns the cluster.** It derives the TorchTPU slice from
  every accelerator node in ``ray.nodes()``, which on a multi-slice cluster mixes
  the trainer's chips into the generator's slice.

* **No cudagraphs, no sleep mode, no memory fraction.** Several of verl's engine
  defaults are CUDA-specific and are rejected outright.
"""

import asyncio
import logging
import os
from typing import Optional

import ray

from verl.plugin.platform import get_platform
from verl.utils.net_utils import is_valid_ipv6_address
# Slice-builder ports for the rollout engine. Must stay disjoint from the trainer's
# range (TRAINER_SLICEBUILDER_BASE_PORT) so the two PJRT runtimes on a cluster never
# try to rendezvous with each other.
from verl.utils.tpu_utils import ROLLOUT_SLICEBUILDER_BASE_PORT
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# TPU env keys that torch_tpu and vLLM both derive. Once verl has settled them for a
# specific slice, later writes are suppressed so the executor's cluster-wide view
# cannot clobber them.
_FROZEN_TPU_ENV_KEYS = (
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TPU_PROCESS_ADDRESSES",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_HOST_BOUNDS",
    "TORCH_TPU_TOPOLOGY",
)

def _patch_platform_noset_env_vars():
    """Ensure vLLM's TPU platform declares RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS
    and excludes TPU_VISIBLE_CHIPS / TPU_VISIBLE_DEVICES from worker env copying.

    Without this, Ray injects TPU_VISIBLE_CHIPS="0" on RayWorkerProc actors created with
    resources={"TPU": 1}. When subsequent tasks call get_node_and_physical_gpu_ids(),
    Ray tries to index assigned_ids into original_ids (which only contains ["0"]),
    causing an IndexError for all TPU device indices > 0 on multi-device hosts.
    """
    try:
        from vllm.platforms import current_platform

        if "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS" not in current_platform.ray_noset_device_env_vars:
            current_platform.ray_noset_device_env_vars = list(current_platform.ray_noset_device_env_vars) + [
                "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS"
            ]
    except Exception:
        pass
    try:
        from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

        if "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS" not in TpuPlatform.ray_noset_device_env_vars:
            TpuPlatform.ray_noset_device_env_vars = list(TpuPlatform.ray_noset_device_env_vars) + [
                "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS"
            ]
    except Exception:
        pass
    try:
        from vllm.v1.executor.ray_utils import WORKER_SPECIFIC_ENV_VARS

        WORKER_SPECIFIC_ENV_VARS.update({"TPU_VISIBLE_CHIPS", "TPU_VISIBLE_DEVICES"})
    except Exception:
        pass
    try:
        from vllm.ray.ray_env import RAY_NON_CARRY_OVER_ENV_VARS

        RAY_NON_CARRY_OVER_ENV_VARS.update({"TPU_VISIBLE_CHIPS", "TPU_VISIBLE_DEVICES"})
    except Exception:
        pass
    try:
        from verl.workers.rollout.vllm_rollout.utils import patch_vllm_tpu_multihost_dp

        patch_vllm_tpu_multihost_dp()
    except Exception:
        pass


def confine_vllm_to_slice(allowed_ips: set[str]) -> None:
    """Stop vLLM's TPU Ray executor from absorbing the whole cluster into one slice.

    ``RayDistributedExecutorV2._slice_host_layout`` walks every node in
    ``ray.nodes()`` that advertises the accelerator resource and treats the union as
    a single TPU slice. With a trainer slice and a rollout slice in the same cluster
    that yields twice as many slice-builder addresses as the rollout slice has chips,
    and PJRT refuses to initialize::

        TPU initialization failed: Invalid --deepsea_slice_builder_worker_addresses
        specified. Expected 8 worker addresses, got 16.

    (which surfaces to Python only as the far less helpful
    ``RuntimeError: PjRtClient is not initialized``).

    Hiding the other slices makes the executor's own derivation correct. Must be
    called in the process that constructs the engine.
    """
    _patch_platform_noset_env_vars()
    os.environ.pop("TPU_VISIBLE_CHIPS", None)
    os.environ.pop("TPU_VISIBLE_DEVICES", None)
    if getattr(ray, "_verl_tpu_slice_confined", False):
        return

    original_nodes = ray.nodes

    def patched_nodes(*args, **kwargs):
        resource_name = get_device_name_ray_resource()
        return [
            node
            for node in original_nodes(*args, **kwargs)
            if resource_name not in node.get("Resources", {}) or node.get("NodeManagerAddress") in allowed_ips
        ]

    ray.nodes = patched_nodes

    original_cluster_resources = ray.cluster_resources

    def patched_cluster_resources(*args, **kwargs):
        resources = original_cluster_resources(*args, **kwargs)
        resource_name = get_device_name_ray_resource()
        resources[resource_name] = sum(
            int(node["Resources"][resource_name])
            for node in patched_nodes()
            if resource_name in node.get("Resources", {})
        )
        return resources

    ray.cluster_resources = patched_cluster_resources

    original_setitem = os.environ.__class__.__setitem__

    def patched_setitem(self, key, value):
        if key in _FROZEN_TPU_ENV_KEYS and key in os.environ:
            return
        original_setitem(self, key, value)

    os.environ.__class__.__setitem__ = patched_setitem

    # Also patch RayDistributedExecutorV2 / RayDistributedExecutor so that:
    # 1. _slice_host_layout filters out hosts not in allowed_ips.
    # 2. _get_dp_geometry correctly returns (dp_size, dp_rank, slice_world_size)
    #    even when EngineCoreActor reconfigures parallel_config for independent DP.
    for mod_name in (
        "vllm_torchtpu.executors.ray_distributed_executor_v2",
        "vllm_torchtpu.executors.ray_distributed_executor",
    ):
        try:
            mod = __import__(mod_name, fromlist=["RayDistributedExecutorV2", "RayDistributedExecutor"])
            for cls_name in ("RayDistributedExecutorV2", "RayDistributedExecutor"):
                cls = getattr(mod, cls_name, None)
                if cls and hasattr(cls, "_slice_host_layout") and not getattr(cls, "_verl_patched", False):
                    orig_layout = cls._slice_host_layout

                    def make_patched(orig):
                        def patched_layout(self, device_str: str):
                            hosts, chips = orig(self, device_str)
                            placement_ips = os.environ.get("VLLM_RAY_DP_PLACEMENT_NODE_IPS") or os.environ.get(
                                "VERL_TPU_ROLLOUT_SLICE_IPS"
                            )
                            if placement_ips:
                                ordered = [ip.strip() for ip in placement_ips.split(",") if ip.strip()]
                                filtered_hosts = [h for h in ordered if h in allowed_ips]
                            else:
                                filtered_hosts = [h for h in hosts if h in allowed_ips]
                            filtered_chips = {k: v for k, v in chips.items() if k in allowed_ips}
                            return filtered_hosts, filtered_chips

                        return patched_layout

                    cls._slice_host_layout = make_patched(orig_layout)
                    cls._verl_patched = True

                if cls and hasattr(cls, "_get_dp_geometry") and not getattr(cls, "_verl_geom_patched", False):
                    orig_geom = cls._get_dp_geometry

                    def make_patched_geom(orig):
                        def patched_geom(self):
                            dp_size, dp_rank, slice_world_size = orig(self)
                            if dp_size <= 1 and (dp_rank > 0 or len(allowed_ips) > 1):
                                env_dp_size = int(os.environ.get("TORCH_TPU_DP_SIZE", "0"))
                                effective_dp_size = env_dp_size or len(allowed_ips)
                                dp_index = getattr(self.parallel_config, "data_parallel_index", None)
                                if dp_index is None:
                                    dp_index = getattr(self.parallel_config, "data_parallel_rank", 0) or 0
                                return effective_dp_size, int(dp_index), self.world_size * effective_dp_size
                            return dp_size, dp_rank, slice_world_size

                        return patched_geom

                    cls._get_dp_geometry = make_patched_geom(orig_geom)
                    cls._verl_geom_patched = True
        except Exception as exc:
            logger.debug("Could not patch %s: %s", mod_name, exc)

    ray._verl_tpu_slice_confined = True


try:
    from vllm.v1.engine.core import EngineCoreProc

    _ORIGINAL_RUN_ENGINE_CORE = EngineCoreProc.run_engine_core
except Exception:
    _ORIGINAL_RUN_ENGINE_CORE = None


def _run_confined_engine_core(*args, **kwargs):
    """Entrypoint wrapper for EngineCoreProc background process on TPU.

    When vLLM v1 spawns EngineCoreProc in a background process via multiprocessing,
    this function ensures the child process applies slice confinement before
    constructing RayDistributedExecutor.
    """
    os.environ.pop("TPU_VISIBLE_CHIPS", None)
    os.environ.pop("TPU_VISIBLE_DEVICES", None)
    slice_ips = os.environ.get("VERL_TPU_ROLLOUT_SLICE_IPS", "")
    if slice_ips:
        confine_vllm_to_slice({ip.strip() for ip in slice_ips.split(",") if ip.strip()})
    _patch_platform_noset_env_vars()
    patch_vllm_tpu_multihost_dp()
    from vllm.v1.engine.core import EngineCoreProc

    target_fn = _ORIGINAL_RUN_ENGINE_CORE or getattr(EngineCoreProc, "_verl_original_run_engine_core", None)
    if target_fn is None:
        raise RuntimeError("Original EngineCoreProc.run_engine_core not found")
    return target_fn(*args, **kwargs)


def get_device_name_ray_resource() -> str:
    from verl.plugin.platform import get_platform

    return get_platform().ray_resource_name()


async def parse_absl_flags_on_engine(engine) -> None:
    """Parse torch_tpu's absl flags inside every engine worker.

    torch_tpu configures its fx passes through absl flags. A Ray worker never goes
    through an absl main, so the registry stays unparsed and the first TPU
    compilation dies with ``UnparsedFlagAccessError``.
    """

    def _parse(_worker):
        from torch_tpu._internal.distributed.multiprocessing import parse_absl_flags

        parse_absl_flags()

    await engine.collective_rpc(_parse)


class vLLMTPUHttpServer(vLLMHttpServer):
    """vLLM http server driving one TPU slice."""

    def _post_init(self, cuda_visible_devices: str) -> None:
        super()._post_init(cuda_visible_devices)
        os.environ.pop("TPU_VISIBLE_CHIPS", None)
        os.environ.pop("TPU_VISIBLE_DEVICES", None)
        if getattr(self, "config", None) and getattr(self.config, "data_parallel_size", 1) > 1:
            os.environ["TORCH_TPU_DP_SIZE"] = str(self.config.data_parallel_size)
        # The engine is constructed later in launch_server(), but the executor reads
        # ray.nodes() from this process, so confine it now.
        slice_ips = os.environ.get("VERL_TPU_ROLLOUT_SLICE_IPS", "")
        if slice_ips:
            confine_vllm_to_slice({ip for ip in slice_ips.split(",") if ip})

    async def run_server(self, args):
        global _ORIGINAL_RUN_ENGINE_CORE
        if getattr(self, "config", None) and getattr(self.config, "data_parallel_size", 1) > 1:
            os.environ["TORCH_TPU_DP_SIZE"] = str(self.config.data_parallel_size)
        try:
            from vllm.v1.engine.core import EngineCoreActor, EngineCoreProc

            if not hasattr(EngineCoreProc, "_verl_original_run_engine_core"):
                EngineCoreProc._verl_original_run_engine_core = EngineCoreProc.run_engine_core
            if _ORIGINAL_RUN_ENGINE_CORE is None:
                _ORIGINAL_RUN_ENGINE_CORE = EngineCoreProc._verl_original_run_engine_core
            EngineCoreProc.run_engine_core = _run_confined_engine_core

            if not hasattr(EngineCoreActor, "_verl_original_init"):
                orig_actor_init = EngineCoreActor.__init__

                def _confined_actor_init(self, *args, **kwargs):
                    os.environ.pop("TPU_VISIBLE_CHIPS", None)
                    os.environ.pop("TPU_VISIBLE_DEVICES", None)
                    slice_ips = os.environ.get("VERL_TPU_ROLLOUT_SLICE_IPS", "")
                    if slice_ips:
                        confine_vllm_to_slice({ip.strip() for ip in slice_ips.split(",") if ip.strip()})
                    if "TORCH_TPU_DP_SIZE" not in os.environ and slice_ips:
                        num_hosts = len([ip for ip in slice_ips.split(",") if ip.strip()])
                        if num_hosts > 1:
                            os.environ["TORCH_TPU_DP_SIZE"] = str(num_hosts)
                    _patch_platform_noset_env_vars()
                    orig_actor_init(self, *args, **kwargs)

                EngineCoreActor.__init__ = _confined_actor_init
                EngineCoreActor._verl_original_init = orig_actor_init
        except Exception as exc:
            logger.warning("Could not patch EngineCoreProc/Actor: %s", exc)

        _patch_platform_noset_env_vars()
        await super().run_server(args)
        # Must happen before the first generation, which is what triggers the first
        # TPU compilation in each worker.
        await parse_absl_flags_on_engine(self.engine)

    def _postprocess_engine_args(self, args: dict) -> None:
        # vLLM on TPU only supports the Ray executor: its worker requires a tcp://
        # rendezvous, and the multiproc executor only emits one behind a ROCm-specific
        # branch.
        args["distributed_executor_backend"] = "ray"
        if args.get("data_parallel_size", 1) > 1:
            args["data_parallel_backend"] = "ray"
            args.pop("nnodes", None)
            args.pop("node_rank", None)
            args.pop("master_addr", None)
            args.pop("master_port", None)
            args.pop("data_parallel_start_rank", None)
            args["data_parallel_size_local"] = max(
                1, self.gpus_per_node // max(1, self.config.tensor_model_parallel_size)
            )
            args.pop("data_parallel_address", None)
            args.pop("data_parallel_rpc_port", None)
        # Torch-compiled graph capture, sleep mode and the HBM fraction knob are all
        # CUDA-only; the TPU engine rejects them.
        for unsupported in ("compilation_config", "enable_sleep_mode", "gpu_memory_utilization"):
            args.pop(unsupported, None)
        args["enforce_eager"] = True

    async def wake_up(self, tags: list[str] | None = None):
        logger.info("vLLMTPUHttpServer.wake_up: no-op on TPU")

    async def sleep(self):
        logger.info("vLLMTPUHttpServer.sleep: no-op on TPU")

    async def release_kv_cache(self):
        logger.info("vLLMTPUHttpServer.release_kv_cache: no-op on TPU")

    async def resume_kv_cache(self):
        logger.info("vLLMTPUHttpServer.resume_kv_cache: no-op on TPU")

    async def clear_kv_cache(self):
        logger.info("vLLMTPUHttpServer.clear_kv_cache: no-op on TPU")


class vLLMTPUReplica(vLLMReplica):
    """Rollout replica placing a vLLM engine on a dedicated TPU slice."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(vLLMTPUHttpServer)
        self._slice_nodes = None

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def rollout_worker_use_gpu(self) -> bool:
        # vLLM's TPU Ray executor spawns its own actor per chip and those claim the
        # slice; verl's rollout workers would otherwise deadlock against them.
        return False

    def rollout_worker_node_ip(self) -> Optional[str]:
        return self._rollout_slice_nodes()[0]["NodeManagerAddress"]

    def rollout_worker_node_ips(self) -> Optional[list[str]]:
        return [n["NodeManagerAddress"] for n in self._rollout_slice_nodes()]

    def _rollout_slice_nodes(self) -> list[dict]:
        """Pick the TPU slice this replica should generate on.

        Slices are identified by the KubeRay ``ray.io/tpu-slice-name`` node label.
        Only slices with every chip still free are eligible, which keeps the replica
        off whichever slice the trainer already reserved.
        """
        if self._slice_nodes is not None:
            return self._slice_nodes

        resource_name = get_device_name_ray_resource()
        available = ray._private.state.available_resources_per_node()

        slices: dict[str, list[dict]] = {}
        for node in ray.nodes():
            if not node.get("Alive") or resource_name not in node.get("Resources", {}):
                continue
            labels = node.get("Labels") or node.get("labels") or {}
            slices.setdefault(labels.get("ray.io/tpu-slice-name", node["NodeID"]), []).append(node)

        def _fully_free(nodes: list[dict]) -> bool:
            return all(
                available.get(n["NodeID"], {}).get(resource_name, 0) >= n["Resources"][resource_name] for n in nodes
            )

        def _sort_key(node: dict) -> int:
            labels = node.get("Labels") or node.get("labels") or {}
            return int(labels.get("ray.io/tpu-worker-id", 0))

        for name in sorted(slices):
            nodes = sorted(slices[name], key=_sort_key)
            if sum(int(n["Resources"][resource_name]) for n in nodes) < self.world_size:
                continue
            if not _fully_free(nodes):
                continue
            logger.info("rollout replica %d selected TPU slice %s", self.replica_rank, name)
            self._slice_nodes = nodes
            return nodes

        raise RuntimeError(
            f"No free TPU slice with at least {self.world_size} chips for rollout replica "
            f"{self.replica_rank}. Slices seen: "
            + ", ".join(f"{name}({len(nodes)} nodes)" for name, nodes in sorted(slices.items()))
        )

    # ------------------------------------------------------------------
    # Server launch
    # ------------------------------------------------------------------

    def _get_server_name_prefix(self) -> str:
        return "vllm_tpu_"

    def _server_env_vars(self) -> dict[str, str]:
        nodes = self._rollout_slice_nodes()
        devices_per_host = int(nodes[0]["Resources"]["TPU"])
        host_ips = [n["NodeManagerAddress"] for n in nodes]
        addresses = [
            f"{ip}:{ROLLOUT_SLICEBUILDER_BASE_PORT + i}"
            for ip in host_ips
            for i in range(devices_per_host)
        ]
        world_size = len(addresses)
        topology = "2,2,4,2" if world_size == 32 else ("2,2,1,2" if world_size == 8 else f"1,1,1,{world_size}")

        libtpu_init_args = " ".join(
            [
                "--xla_tpu_use_dynamic_smem_negotiation=true",
                "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864",
                "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=67108864",
                "--xla_tpu_use_enhanced_launch_barrier=false",
            ]
        )

        return {
            "VERL_TPU_ROLLOUT_SLICE_IPS": ",".join(host_ips),
            "TORCH_TPU_BASE_PORT": str(ROLLOUT_SLICEBUILDER_BASE_PORT),
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(addresses),
            "TORCH_TPU_TOPOLOGY": topology,
            "TORCH_TPU_DP_SIZE": str(self.config.data_parallel_size),
            "TPU_ACCELERATOR_TYPE": "tpu7x",
            "TPU_NAME": "tpu-rollout",
            "TPU_WORKER_ID": "0",
            "TPU_SKIP_MDS_QUERY": "true",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "VLLM_PLUGINS": "torchtpu,torchtpu_layers",
            "SKIP_JAX_PRECOMPILE": "1",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
            "LIBTPU_INIT_ARGS": libtpu_init_args,
            "TORCH_DYNAMO_RECOMPILE_LIMIT": "100",
            "PYTHONUNBUFFERED": "1",
            "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
            "TPU_HOST_BOUNDS": topology,
            "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1,1",
            "TPU_MULTIHOST_BACKEND": "ray",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "1",
            "VLLM_RAY_DP_PLACEMENT_NODE_IPS": ",".join(host_ips),
            "VLLM_RAY_DP_PACK_STRATEGY": "strict",
            "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY": "VERL_,TORCH_TPU_,RAY_",
            "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": (
                "TPU_ACCELERATOR_TYPE,TPU_NAME,TPU_HOST_BOUNDS,TPU_CHIPS_PER_HOST_BOUNDS,"
                "TPU_MULTIHOST_BACKEND,TPU_SKIP_MDS_QUERY,RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS,"
                "LIBTPU_INIT_ARGS,SKIP_JAX_PRECOMPILE,TORCH_TPU_DP_SIZE,TORCH_TPU_SLICEBUILDER_ADDRESSES,"
                "TORCH_TPU_TOPOLOGY,TPU_PROCESS_ADDRESSES"
            ),
        }

    async def launch_servers(self):
        """Launch http server on the rollout TPU slice."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        slice_nodes = self._rollout_slice_nodes()
        prefix = self._get_server_name_prefix()
        if self.is_reward_model:
            name = f"{prefix}server_reward_{self.replica_rank}_0{self.name_suffix}"
        elif self.is_teacher_model:
            name = f"{prefix}server_teacher_{self.replica_rank}_0{self.name_suffix}"
        else:
            name = f"{prefix}server_{self.replica_rank}_0{self.name_suffix}"

        env_vars = {
            **{var: "1" for var in get_platform().ray_noset_envvars()},
            **get_platform().rollout_env_vars(),
            **self._server_env_vars(),
        }
        env_vars.pop("TPU_VISIBLE_CHIPS", None)
        env_vars.pop("TPU_VISIBLE_DEVICES", None)

        # In TPU multi-host ray DP, a single server actor on node 0 drives all workers across the slice.
        node_id = slice_nodes[0]["NodeID"]
        workers = self.workers[: self.gpus_per_replica_node]
        server = self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": env_vars},
            name=name,
            max_concurrency=self.max_concurrency,
        ).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=workers,
            replica_rank=self.replica_rank,
            node_rank=0,
            gpus_per_node=self.gpus_per_replica_node,
            nnodes=self.nnodes,
            cuda_visible_devices="",
        )
        self.servers = [server]

        # launch http server on node 0
        master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
        await self.servers[0].launch_server.remote(
            master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
