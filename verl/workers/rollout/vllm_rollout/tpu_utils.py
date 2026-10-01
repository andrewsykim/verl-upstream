# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
import copy
import gc
import json
import logging
import multiprocessing
import multiprocessing.process
import os
import sys
import time
import types
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from verl.plugin.platform.platform_tpu import get_tpu_topology_map
from verl.utils.device import get_resource_name
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension as _BaseWorkerExtension

# --- Google TPU specific global constants ---
TPU_WEIGHT_REGISTRY_ACTOR_NAME = "TPUWeightRegistry"
TPU_WEIGHT_REGISTRY_NAMESPACE = "verl"
TPU_ROLLOUT_BASE_PORT = 8070
DEFAULT_TPU_TOPOLOGY_MAP = {
    1: "1,1,1",
    2: "1,2,1",
    4: "2,2,1",
    8: "2,4,1",
    16: "4,4,1",
    32: "4,8,1",
    64: "8,8,1",
    128: "8,16,1",
    256: "16,16,1",
}


def _resolve_tpu_topology_bounds(
    total_chips: int, num_nodes: int, fallback_map: dict | None = None
) -> tuple[str, str, str, str]:
    """Dynamically resolves (topology, host_bounds, chips_per_host_bounds, chips_per_host) for TPU slices."""
    topo_map = dict(fallback_map) if fallback_map else {}
    topo_map.update(get_tpu_topology_map())

    unit_bounds = topo_map.get(1, "1,1,1")
    topology = os.environ.get("TORCH_TPU_TOPOLOGY") or topo_map.get(total_chips, unit_bounds)
    inferred_chips_per_host = max(1, total_chips // max(1, num_nodes))
    chips_per_host = str(os.environ.get("VLLM_TPU_CHIPS_PER_HOST", inferred_chips_per_host))

    if total_chips <= 4 and unit_bounds == "1,1,1":
        host_bounds = unit_bounds
        chips_per_host_bounds = topology if num_nodes == 1 else unit_bounds
    else:
        host_bounds = topology
        chips_per_host_bounds = unit_bounds

    return topology, host_bounds, chips_per_host_bounds, chips_per_host


# -------------------------------------------

try:
    from verl.checkpoint_engine.tpu_checkpoint_engine import load_weights_on_worker
except ImportError:
    load_weights_on_worker = None

# Fallback imports for TPU vLLM platforms
try:
    from vllm_torchtpu.executors import ray_distributed_executor
except ImportError:
    ray_distributed_executor = None

try:
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
except ImportError:
    EngineArgs = None
    AsyncEngineArgs = None

try:
    import vllm.envs as vllm_envs
except ImportError:
    vllm_envs = None

try:
    import vllm_torchtpu.envs as tpu_envs
except ImportError:
    tpu_envs = None

try:
    from vllm_torchtpu.worker.tpu_worker import TPUWorker
except ImportError:
    TPUWorker = None

try:
    from vllm.utils import get_ip
except ImportError:
    try:
        from vllm.utils.network_utils import get_ip
    except ImportError:
        get_ip = None

try:
    from vllm.v1.executor.ray_executor import RayWorkerMetaData
except ImportError:
    try:
        from vllm.v1.executor.ray_utils import RayWorkerMetaData
    except ImportError:

        class RayWorkerMetaData:
            def __init__(self, worker, created_rank):
                self.worker = worker
                self.created_rank = created_rank
                self.adjusted_rank = None
                self.ip = None


try:
    from vllm.utils import get_open_port
except ImportError:
    try:
        from vllm_torchtpu.utils import get_open_port
    except ImportError:
        try:
            from vllm.utils.network_utils import get_open_port
        except ImportError:
            get_open_port = None

try:
    from vllm_torchtpu.platforms.tpu_platform import get_distributed_init_method
except ImportError:
    try:
        from vllm.utils.network_utils import get_distributed_init_method
    except ImportError:
        get_distributed_init_method = None

try:
    from vllm.platforms import current_platform
except ImportError:
    current_platform = None


class PickleableProcessWrapper:
    """
    A pickle-compatible wrapper for multiprocessing target functions to ensure
    vLLM-on-TPU worker patches are automatically applied in the spawned processes.
    """

    def __init__(self, target):
        self.target = target

    def __call__(self, *args, **kwargs):
        patch_vllm_for_tpu()
        if self.target is not None:
            return self.target(*args, **kwargs)


def patch_multiprocessing_for_tpu() -> None:
    """
    Monkey-patch multiprocessing.process.BaseProcess to wrap target entrypoints
    with PickleableProcessWrapper, propagating vLLM-on-TPU patches to child processes.
    """
    if getattr(multiprocessing.process.BaseProcess, "_tpu_patched", False):
        return

    original_init = multiprocessing.process.BaseProcess.__init__

    def patched_init(self, *args, **kwargs):
        target = kwargs.get("target", None)
        if target is None and len(args) > 1:
            target = args[1]

        if target is not None:
            wrapped_target = PickleableProcessWrapper(target)
            if "target" in kwargs:
                kwargs["target"] = wrapped_target
            elif len(args) > 1:
                args = list(args)
                args[1] = wrapped_target
                args = tuple(args)

        original_init(self, *args, **kwargs)

    multiprocessing.process.BaseProcess.__init__ = patched_init
    multiprocessing.process.BaseProcess._tpu_patched = True


class vLLMRaidenWorkerExtension(_BaseWorkerExtension):
    """vLLM Worker Extension for Google Cloud TPU with Raiden P2P weight synchronization.

    Workloads using the 'raiden' checkpoint engine for high-speed TPU weight sync
    must use this worker extension to bind TPU model tensors with WeightSynchronizer
    and orchestrate dynamic zero-copy H2D DMA transfers during rollout.
    """

    def _get_vllm_model(self):
        """Extract the underlying nn.Module from the vLLM V1 worker."""
        worker = getattr(self, "worker", self)
        return worker.model_runner.model

    def init_raiden_sync_on_worker(self, parallelism: int = 8) -> bool:
        """Initialize Raiden WeightSynchronizer listener and register with central RaidenController."""
        if hasattr(self, "_raiden_ws") and self._raiden_ws is not None:
            return True

        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            logging.getLogger(__name__).warning("Raiden Sampler: could not locate vllm_model to bind parameters.")
            return False

        from tpu_sync.rpc import raiden_service_pb2

        from verl.checkpoint_engine.raiden_checkpoint_engine import (
            create_torch_weight_synchronizer,
        )

        bind_ip = ray.util.get_node_ip_address().strip("[]")
        rank_val = getattr(self, "rank", 0)
        listener_port = 12000 + rank_val

        # 1. Fetch RaidenController address and un-fused global shapes from TPUWeightRegistry
        from verl.checkpoint_engine.tpu_weight_registry import get_tpu_weight_registry

        registry = get_tpu_weight_registry()

        controller_addr, global_shapes_map = None, {}
        for _ in range(30):
            try:
                addr, shapes = ray.get(
                    [
                        registry.get_controller_address.remote(),
                        registry.get_global_shapes.remote(),
                    ]
                )
                if addr and shapes:
                    controller_addr, global_shapes_map = addr, shapes
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not controller_addr or not global_shapes_map:
            raise RuntimeError(
                f"Raiden Sampler Rank {rank_val}: No RaidenController address or "
                f"global shapes found in TPUWeightRegistry after timeout"
            )

        # 2. Determine tensor parallel size from vLLM V1 config
        worker = getattr(self, "worker", self)
        parallel_config = getattr(getattr(worker, "vllm_config", None), "parallel_config", None)
        tp_size = getattr(parallel_config, "tensor_parallel_size", None)
        if not tp_size or tp_size <= 0:
            raise RuntimeError(
                f"Raiden Sampler Rank {rank_val}: Unable to determine tensor_parallel_size (tp_size) "
                f"from worker.vllm_config.parallel_config!"
            )

        # 3. Allocate local TPU staging buffers matching Trainer un-fused layout and sharding specs
        ROW_PARALLEL_SUFFIXES = (".o_proj.weight", ".down_proj.weight")

        staging_tensors = {}
        variable_protos = []
        valid_params = []

        for idx, (name, g_shape) in enumerate(sorted(global_shapes_map.items(), key=lambda x: x[0])):
            g_shape = list(g_shape)
            spec_axes = [""] * len(g_shape)
            local_shape = list(g_shape)

            if len(g_shape) == 2:
                if any(name.endswith(s) for s in ROW_PARALLEL_SUFFIXES):
                    # Row parallel: partition input dimension (dim 1) across TP ranks
                    spec_axes = ["", "tp"]
                    local_shape[1] = g_shape[1] // tp_size
                elif g_shape[0] % tp_size == 0:
                    # Column parallel or vocab parallel: partition output dimension (dim 0) across TP ranks
                    spec_axes = ["tp", ""]
                    local_shape[0] = g_shape[0] // tp_size

            sharding_mesh = [tp_size if axis == "tp" else 1 for axis in spec_axes]

            # Allocate local TPU staging buffer with matching layout
            t = torch.empty(local_shape, dtype=torch.bfloat16, device=torch.device("tpu"))
            staging_tensors[name] = t
            valid_params.append((name, t))

            variable_protos.append(
                raiden_service_pb2.VariableMetadataProto(
                    name=name,
                    shape=g_shape,
                    mesh_shape=sharding_mesh,
                    layout=list(range(len(local_shape) - 1, -1, -1)),
                    item_size=t.element_size(),
                    layer_idx=idx,
                    sharding_spec=spec_axes,
                )
            )

        self._raiden_staging = staging_tensors
        self._sorted_vllm_params = valid_params

        try:
            from torch_tpu._internal import sync as torch_tpu_sync

            torch_tpu_sync.synchronize(wait=True)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not synchronize via torch_tpu: {e}")

        logging.getLogger(__name__).info(
            f"Raiden Sampler Rank {rank_val}: binding {len(valid_params)} staging tensors to WeightSynchronizer "
            f"(sample: name={valid_params[0][0]}, shape={valid_params[0][1].shape}, "
            f"total numel={sum(t.numel() for _, t in valid_params)})"
        )

        self._raiden_ws = create_torch_weight_synchronizer(
            [[t] for _, t in valid_params],
            local_port=0,
            parallelism=parallelism,
            listener_port=listener_port,
            bind_ip=bind_ip,
        )

        try:
            from tpu_sync.rpc import raiden_controller

            ctrl_client = raiden_controller.RaidenControllerClientFacade(controller_addr)
            unit_id = raiden_controller.RaidenId("sampler", str(rank_val), "weights")
            ctrl_client.register_work_unit(
                unit_id,
                [f"{bind_ip}:{self._raiden_ws.local_port}"],
                f"{bind_ip}:{self._raiden_ws.listener_port}",
                mesh_shape=[1, tp_size],
                variables=variable_protos,
                mesh_axes=["fsdp", "tp"],
            )
            logging.getLogger(__name__).info(
                f"Raiden Sampler Rank {rank_val} bound {len(valid_params)} dynamic staging tensors and registered "
                f"directly with RaidenController ({controller_addr}): mesh_shape=[1, {tp_size}], "
                f"data_port={self._raiden_ws.local_port}, listener_port={self._raiden_ws.listener_port}"
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Raiden Sampler Rank {rank_val} failed to register with RaidenController: {e}"
            )
            raise

        return True

    @torch.no_grad()
    def install_raiden_weights(self) -> int:
        """Install received weights from host staging buffer into TPU HBM via zero-copy H2D DMA
        and fuse/transpose them directly into vLLM model parameters."""
        if not hasattr(self, "_raiden_ws") or self._raiden_ws is None:
            logging.getLogger(__name__).warning(
                "Raiden Sampler: install_raiden_weights called before _raiden_ws was initialized."
            )
            return 0

        t_start = time.perf_counter()
        t_h2d_start = time.perf_counter()
        self._raiden_ws.h2d()
        t_h2d = time.perf_counter() - t_h2d_start

        # Unpack, fuse, and transpose staging tensors into vLLM model parameters
        vllm_model = self._get_vllm_model()
        model_sd = vllm_model.state_dict() if hasattr(vllm_model, "state_dict") else vllm_model.model.state_dict()
        module_dict = dict(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else {}

        def resolve_model_key(k: str) -> Optional[str]:
            if k in model_sd:
                return k
            if k.startswith("model.") and k[6:] in model_sd:
                return k[6:]
            if f"model.{k}" in model_sd:
                return f"model.{k}"
            return None

        def get_parent_mod(target_key: str):
            parent = target_key.rsplit(".", 1)[0] if "." in target_key else ""
            if parent in module_dict:
                return module_dict[parent]
            if parent.startswith("model.") and parent[6:] in module_dict:
                return module_dict[parent[6:]]
            if f"model.{parent}" in module_dict:
                return module_dict[f"model.{parent}"]
            return None

        def to_target_layout(tensor: torch.Tensor, target_local: torch.Tensor, flipped: bool) -> torch.Tensor:
            if tensor.ndim == 2 and (
                flipped or (tensor.shape != target_local.shape and tensor.T.shape == target_local.shape)
            ):
                return tensor.transpose(0, 1).contiguous()
            return tensor.contiguous()

        FUSED_MAP = [
            (
                ".self_attn.qkv_proj.weight",
                [".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            ),
            (
                ".self_attn.qkv_proj.bias",
                [".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            ),
            (".mlp.gate_up_proj.weight", [".mlp.gate_proj.weight", ".mlp.up_proj.weight"]),
            (".mlp.gate_up_proj.bias", [".mlp.gate_proj.bias", ".mlp.up_proj.bias"]),
        ]

        consumed_staging = set()

        # 1. Handle fused projections (QKV and Gate-Up)
        for target_suffix, src_suffixes in FUSED_MAP:
            primary_src = src_suffixes[0]
            for name in list(self._raiden_staging.keys()):
                if primary_src in name:
                    layer_src_keys = [name.replace(primary_src, s) for s in src_suffixes]
                    if all(sk in self._raiden_staging for sk in layer_src_keys):
                        target_name = name.replace(primary_src, target_suffix)
                        resolved_key = resolve_model_key(target_name)
                        if resolved_key is not None:
                            target_param = model_sd[resolved_key]
                            target_local = (
                                target_param.to_local() if hasattr(target_param, "to_local") else target_param
                            )
                            parent_mod = get_parent_mod(resolved_key)
                            is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))
                            parts = [self._raiden_staging[sk] for sk in layer_src_keys]
                            fused = torch.cat(parts, dim=0)
                            fused_adapted = to_target_layout(fused, target_local, is_flipped)
                            target_local.copy_(fused_adapted)
                            consumed_staging.update(layer_src_keys)

        # 2. Handle all remaining non-fused parameters (o_proj, down_proj, layernorms, embeddings)
        for name, src_t in self._raiden_staging.items():
            if name in consumed_staging:
                continue
            resolved_key = resolve_model_key(name)
            if resolved_key is not None:
                target_param = model_sd[resolved_key]
                target_local = target_param.to_local() if hasattr(target_param, "to_local") else target_param
                parent_mod = get_parent_mod(resolved_key)
                is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))
                adapted = to_target_layout(src_t, target_local, is_flipped)
                target_local.copy_(adapted)

        # 3. Handle tied word embeddings
        if (
            hasattr(vllm_model, "lm_head")
            and hasattr(vllm_model, "model")
            and hasattr(vllm_model.model, "embed_tokens")
        ):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.model.embed_tokens.weight)
        elif hasattr(vllm_model, "lm_head") and hasattr(vllm_model, "embed_tokens"):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.embed_tokens.weight)

        t_sync_start = time.perf_counter()
        try:
            from torch_tpu._internal import sync as torch_tpu_sync

            torch_tpu_sync.synchronize(wait=True)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not synchronize via torch_tpu: {e}")
        t_sync = time.perf_counter() - t_sync_start
        t_total = time.perf_counter() - t_start

        print(
            f"[RAIDEN TELEMETRY | Sampler Worker] Sampler Rank {getattr(self, 'rank', 0)}: "
            f"install_raiden_weights completed in {t_total:.4f}s (H2D={t_h2d:.4f}s, TPUSyncBarrier={t_sync:.4f}s)",
            flush=True,
        )
        logging.getLogger(__name__).info(
            f"[RAIDEN TELEMETRY | Sampler Worker] install_raiden_weights completed in {t_total:.4f}s "
            f"(H2D={t_h2d:.4f}s, TPUSyncBarrier={t_sync:.4f}s)"
        )
        return 1

    def get_model_weights_stats(self, include_shards: bool = False) -> dict:
        """Computes deterministic parameter count, L1 norm, and L2 norm across all model parameters.
        When include_shards=True, also returns CPU tensor bytes for each parameter to allow global re-assembly.
        """
        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            return {"error": "No model found on worker"}

        if not hasattr(self, "_sorted_vllm_params") or not self._sorted_vllm_params:
            raise RuntimeError(
                f"get_model_weights_stats is only supported for Raiden after initialization, "
                f"but self._sorted_vllm_params is not initialized on worker rank {getattr(self, 'rank', 'unknown')}!"
            )

        rank_val = getattr(self, "rank", 0)

        from verl.checkpoint_engine.raiden_checkpoint_engine import compute_tensor_stats

        res = compute_tensor_stats(self._sorted_vllm_params)
        res["rank"] = rank_val

        return res


def _patched_run_engine_core(*args, **kwargs):
    try:
        patch_vllm_for_tpu()
    except Exception as e:
        print(f"[TPU DEBUG] Error re-applying patch in EngineCoreProc: {e}", flush=True)

    global _orig_run_engine_core
    if _orig_run_engine_core is None:
        import vllm.v1.engine.core as v1_core

        _orig_run_engine_core = getattr(v1_core.EngineCoreProc, "_unpatched_run_engine_core", None)

    if hasattr(_orig_run_engine_core, "__func__"):
        _orig_run_engine_core = _orig_run_engine_core.__func__

    if _orig_run_engine_core is not None:
        return _orig_run_engine_core(*args, **kwargs)
    raise RuntimeError("[TPU ERROR] _orig_run_engine_core could not be resolved in _patched_run_engine_core")


def patch_vllm_for_tpu() -> None:
    """
    Apply TPU-specific patches and workarounds to vLLM and torchtpu-vllm workers.
    Ensures correct topology routing, un-clashed TCP ports, custom weight registry loading,
    and driver-worker environment synchronization on GKE TPU v6e instances.
    """
    logger = logging.getLogger(__name__)

    try:
        import torch
        import torch._dynamo
        import torch._ops
        import torch.compiler

        def _allow(op):
            if op is not None:
                for fn in (
                    getattr(torch.compiler, "allow_in_graph", None),
                    getattr(torch._dynamo, "allow_in_graph", None),
                ):
                    if fn:
                        try:
                            fn(op)
                        except Exception:
                            pass

        for ns_name in ["_c10d_functional", "c10d_functional"]:
            if hasattr(torch.ops, ns_name):
                ns = getattr(torch.ops, ns_name)
                _allow(ns)
                for name in dir(ns):
                    try:
                        attr = getattr(ns, name)
                        _allow(attr)
                        if hasattr(attr, "default"):
                            _allow(attr.default)
                    except Exception:
                        pass
    except Exception:
        pass

    def dummy_reset_encoder_cache(*args, **kwargs):
        pass

    def load_weights_from_ray_registry(self, step_key: int):
        rank_val = getattr(
            self, "rank", getattr(self, "adjusted_rank", getattr(self, "rpc_rank", int(os.environ.get("RANK", "0"))))
        )

        shm_dir = "/tmp/verl_weight_cache/shared"
        os.makedirs(shm_dir, exist_ok=True)
        shm_file_path = f"{shm_dir}/state_dict_{step_key}.pt"
        shm_tmp_path = f"{shm_dir}/state_dict_{step_key}.tmp"
        shm_ready_path = f"{shm_dir}/state_dict_{step_key}.ready"
        lock_path = f"{shm_dir}/state_dict_{step_key}.lock"
        state_dict_data = None

        # Atomic lock to decide the designated master for this step on this physical VM node
        is_master = False
        lock_fd = None
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            is_master = True
        except FileExistsError:
            is_master = False

        if is_master:
            # Clean up older steps' cache files from /tmp to ensure NVMe disk space never exhausts
            try:
                for file_name in os.listdir(shm_dir):
                    if file_name.startswith("state_dict_"):
                        parts = file_name.split("_")
                        if len(parts) >= 3:
                            try:
                                old_step_str = parts[2].split(".")[0]
                                old_step = int(old_step_str)
                                if old_step != step_key:
                                    old_path = os.path.join(shm_dir, file_name)
                                    os.remove(old_path)
                            except ValueError:
                                pass
            except Exception as e:
                logger.warning(f"Error during cache cleanup: {e}")

            # Node master block: fetch weights once, convert, and cache to /tmp
            try:
                registry = ray.get_actor(TPU_WEIGHT_REGISTRY_ACTOR_NAME, namespace=TPU_WEIGHT_REGISTRY_NAMESPACE)
                state_dict_ref = ray.get(registry.get_weights.remote(step_key))
            except Exception as e:
                logger.warning(f"Failed to get weights from TPUWeightRegistry: {e}")
                if lock_fd is not None:
                    os.close(lock_fd)
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass
                return 0

            if state_dict_ref is None:
                logger.warning(f"No weights registered under step_key={step_key}")
                if lock_fd is not None:
                    os.close(lock_fd)
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass
                return 0

            if isinstance(state_dict_ref, list) and all(isinstance(r, ray.ObjectRef) for r in state_dict_ref):
                grouped_dict = {}
                for chunk_ref in state_dict_ref:
                    chunk_obj = ray.get(chunk_ref)
                    if isinstance(chunk_obj, tuple) and len(chunk_obj) == 2:
                        group_name, group_sd = chunk_obj
                        grouped_dict[group_name] = group_sd
                    elif isinstance(chunk_obj, dict) and "grouped" in chunk_obj:
                        grouped_dict.update(chunk_obj["grouped"])
                    del chunk_obj
                state_dict_data = {"grouped": grouped_dict}
            else:
                if isinstance(state_dict_ref, list) and len(state_dict_ref) == 1:
                    state_dict_ref = state_dict_ref[0]  # unwrap the nesting from set_weights
                if isinstance(state_dict_ref, str):
                    state_dict_data = torch.load(state_dict_ref, map_location="cpu", weights_only=False)
                elif isinstance(state_dict_ref, ray.ObjectRef):
                    state_dict_data = ray.get(state_dict_ref)
                else:
                    state_dict_data = state_dict_ref

            # Convert NumPy arrays back to native PyTorch CPU tensors
            if isinstance(state_dict_data, dict) and "grouped" in state_dict_data:
                for group_name, group_sd in state_dict_data["grouped"].items():
                    flat_tensors = group_sd["flat_tensors"]
                    for dtype, arr in list(flat_tensors.items()):
                        if isinstance(arr, np.ndarray):
                            flat_tensors[dtype] = torch.from_numpy(arr)

            # Save atomically using temp file and rename
            torch.save(state_dict_data, shm_tmp_path)
            os.replace(shm_tmp_path, shm_file_path)

            # Release memory before mapping
            del state_dict_data
            gc.collect()

            # Signal ready to all other processes on this node
            with open(shm_ready_path, "w") as f:
                f.write("ready")

            if lock_fd is not None:
                os.close(lock_fd)
                # The lock file must be removed on the success path too. It is
                # created with O_CREAT|O_EXCL and /tmp survives across Ray jobs
                # on the same pod, so leaking it makes every subsequent job fail
                # master election and silently reload a stale cached state dict.
                try:
                    os.remove(lock_path)
                except Exception:
                    pass

            # Stagger ranks before loading to serialize memory traffic
            time.sleep((rank_val % 4) * 0.4)

            # Load memory-mapped version to share pages across ranks on the same host
            gc.collect()
            try:
                state_dict_data = torch.load(shm_file_path, map_location="cpu", weights_only=False, mmap=True)
            except Exception as e:
                logger.warning(f"Shared memory state dict {shm_file_path} not available: {e}")
                state_dict_data = None
        else:
            # Node worker processes: wait for ready marker
            t_wait_start = time.time()
            while not os.path.exists(shm_ready_path):
                time.sleep(0.05)
                if time.time() - t_wait_start > 30:
                    break

            # Stagger ranks before loading to serialize memory traffic
            time.sleep((rank_val % 4) * 0.4)

            gc.collect()
            try:
                state_dict_data = torch.load(shm_file_path, map_location="cpu", weights_only=False, mmap=True)
            except Exception as e:
                logger.warning(f"Shared memory state dict {shm_file_path} not available: {e}")
                state_dict_data = None

        if hasattr(self, "worker") and self.worker is not None:
            worker_inst = self.worker
        else:
            worker_inst = self

        if hasattr(worker_inst, "get_model"):
            vllm_model = worker_inst.get_model()
        else:
            vllm_model = worker_inst.model_runner.model

        num_keys = 0
        if load_weights_on_worker is not None:
            res_loader = load_weights_on_worker(vllm_model, state_dict_data, rank_val)
            if isinstance(res_loader, tuple):
                num_keys = res_loader[0]
            else:
                num_keys = res_loader

        del state_dict_data
        gc.collect()

        return num_keys

    patch_multiprocessing_for_tpu()

    if "RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK" in os.environ:
        del os.environ["RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK"]

    try:
        orig_ray_init = ray.init

        def patched_ray_init(*args, **kwargs):
            if "runtime_env" in kwargs and kwargs["runtime_env"]:
                rt_env = kwargs["runtime_env"]
                if isinstance(rt_env, dict) and "worker_process_setup_hook" in rt_env:
                    rt_env = dict(rt_env)
                    del rt_env["worker_process_setup_hook"]
                    kwargs["runtime_env"] = rt_env
            if "RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK" in os.environ:
                del os.environ["RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK"]
            return orig_ray_init(*args, **kwargs)

        ray.init = patched_ray_init
    except Exception as e:
        logger.warning(f"Failed to patch ray.init to clear worker_process_setup_hook: {e}")

    try:
        if "vllm.utils.import_utils" not in sys.modules:
            sys.modules["vllm.utils.import_utils"] = types.ModuleType("import_utils")
        sys.modules["vllm.utils.import_utils"].init_cached_hf_modules = lambda: None

        os.environ["VLLM_USE_V1"] = "0"
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

        if ray_distributed_executor is None:
            return

        try:
            if EngineArgs is not None:
                orig_create_engine_config = EngineArgs.create_engine_config

                def patched_create_engine_config(self, *args, **kwargs):
                    is_tpu = get_resource_name() == "TPU" or os.environ.get("VLLM_USE_V1") == "0"
                    is_multi_host = (
                        self.tensor_parallel_size > 4
                        or int(os.environ.get("NNODES_ROLLOUT", "1")) > 1
                        or os.environ.get("TPU_MULTIHOST_BACKEND") == "ray"
                    )
                    if is_tpu:
                        os.environ["VLLM_USE_V1"] = "0"

                    if getattr(self, "data_parallel_size", 1) <= 1:
                        if hasattr(self, "data_parallel_external_lb"):
                            self.data_parallel_external_lb = False
                        if hasattr(self, "data_parallel_rank"):
                            self.data_parallel_rank = None
                        if hasattr(self, "data_parallel_size_local"):
                            self.data_parallel_size_local = None
                        if hasattr(self, "data_parallel_start_rank"):
                            self.data_parallel_start_rank = None
                        if hasattr(self, "data_parallel_hybrid_lb"):
                            self.data_parallel_hybrid_lb = False

                    if not is_multi_host:
                        if "TPU_MULTIHOST_BACKEND" in os.environ:
                            del os.environ["TPU_MULTIHOST_BACKEND"]
                        if vllm_envs is not None and hasattr(vllm_envs, "TPU_MULTIHOST_BACKEND"):
                            vllm_envs.TPU_MULTIHOST_BACKEND = None
                        if tpu_envs is not None and hasattr(tpu_envs, "TPU_MULTIHOST_BACKEND"):
                            tpu_envs.TPU_MULTIHOST_BACKEND = None

                        vllm_config = orig_create_engine_config(self, *args, **kwargs)

                        logger.info("Single-host TPU rollout detected; using local executor backend.")
                        if hasattr(vllm_config, "scheduler_config") and hasattr(
                            vllm_config.scheduler_config, "async_scheduling"
                        ):
                            vllm_config.scheduler_config.async_scheduling = True
                            logger.info("Enabled async_scheduling on TPU for single-host rollout.")
                    else:
                        os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
                        if vllm_envs is not None and hasattr(vllm_envs, "TPU_MULTIHOST_BACKEND"):
                            vllm_envs.TPU_MULTIHOST_BACKEND = "ray"
                        if tpu_envs is not None and hasattr(tpu_envs, "TPU_MULTIHOST_BACKEND"):
                            tpu_envs.TPU_MULTIHOST_BACKEND = "ray"

                        vllm_config = orig_create_engine_config(self, *args, **kwargs)
                        vllm_config.parallel_config.distributed_executor_backend = "ray"
                        logger.info("Multi-host TPU rollout detected; using Ray distributed executor backend.")
                        if hasattr(vllm_config, "scheduler_config") and hasattr(
                            vllm_config.scheduler_config, "async_scheduling"
                        ):
                            vllm_config.scheduler_config.async_scheduling = False
                            logger.info("Disabled async_scheduling on TPU for multi-host Ray rollout.")
                    return vllm_config

                EngineArgs.create_engine_config = patched_create_engine_config
                if AsyncEngineArgs is not None:
                    AsyncEngineArgs.create_engine_config = patched_create_engine_config
        except Exception as e:
            logger.warning(f"Failed to patch EngineArgs.create_engine_config: {e}")

        try:
            if TPUWorker is not None:
                TPUWorker.reset_encoder_cache = dummy_reset_encoder_cache
                TPUWorker.load_weights_from_ray_registry = load_weights_from_ray_registry
                logger.info(
                    "Patched TPUWorker class with dummy_reset_encoder_cache and load_weights_from_ray_registry."
                )
        except Exception as e:
            logger.warning(f"Failed to patch TPUWorker class directly: {e}")

        _topo_map = getattr(
            ray_distributed_executor,
            "TPU_TOPOLOGY_MAP",
            getattr(ray_distributed_executor, "TPU_MULTIHOST_TOPOLOGY_MAP", None),
        )
        if _topo_map is not None:
            _topo_map.update(get_tpu_topology_map())

        original_driver_environ_setitem = os.environ.__class__.__setitem__

        def patched_driver_environ_setitem(self, key, value):
            if key in (
                "TORCH_TPU_SLICEBUILDER_ADDRESSES",
                "TPU_PROCESS_ADDRESSES",
            ):
                existing = os.environ.get(key)
                # Prevent torch_tpu's single-device fallback ("localhost:...") from clobbering
                # an already-configured multi-worker slicebuilder address list, while allowing
                # explicit worker environment updates to override KubeRay pod-level defaults.
                if existing and "localhost:" not in existing and "localhost:" in str(value):
                    value = existing
            elif key == "LIBTPU_INIT_ARGS":
                value = value.replace("--deepsea_chip_config_name=megachip_tccontrol", "")
            original_driver_environ_setitem(self, key, value)

        os.environ.__class__.__setitem__ = patched_driver_environ_setitem

        try:
            import vllm.v1.executor.ray_utils as v1_ray_utils

            orig_avail_res = v1_ray_utils.available_resources_per_node

            def patched_avail_res(*args, **kwargs):
                res_map = orig_avail_res(*args, **kwargs)
                for node_id, res in res_map.items():
                    res["TPU"] = max(res.get("TPU", 0.0), 4.0)
                return res_map

            v1_ray_utils.available_resources_per_node = patched_avail_res

            orig_init_ray_cluster = v1_ray_utils.initialize_ray_cluster

            def patched_initialize_ray_cluster(parallel_config, *args, **kwargs):
                if parallel_config.placement_group is None:
                    curr_pg = ray.util.get_current_placement_group()
                    if curr_pg is None:
                        pg_name = os.environ.get("VERL_ROLLOUT_PG_NAME")
                        if pg_name:
                            try:
                                curr_pg = ray.util.get_placement_group(pg_name)
                            except Exception:
                                pass
                        if curr_pg is None:
                            try:
                                pgs = ray.util.placement_group_table()
                                for pg_id, pg_info in pgs.items():
                                    state = (
                                        pg_info.get("state")
                                        if isinstance(pg_info, dict)
                                        else getattr(pg_info, "state", None)
                                    )
                                    if hasattr(state, "name"):
                                        state = state.name
                                    name = (
                                        pg_info.get("name")
                                        if isinstance(pg_info, dict)
                                        else getattr(pg_info, "name", None)
                                    )
                                    if (
                                        str(state) in ("CREATED", "1")
                                        and name
                                        and ("rollout" in str(name) or "global" in str(name))
                                    ):
                                        try:
                                            candidate_pg = ray.util.get_placement_group(name)
                                            if candidate_pg is not None:
                                                num_bundles = len(getattr(candidate_pg, "bundle_specs", []))
                                                if num_bundles >= parallel_config.world_size:
                                                    curr_pg = candidate_pg
                                                    break
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    parallel_config.placement_group = curr_pg
                try:
                    return orig_init_ray_cluster(parallel_config, *args, **kwargs)
                except ValueError as e:
                    if "exceeds the total number of available" in str(e) or "placement group" in str(e):
                        logger.warning(f"Bypassed vLLM placement group size validation on multi-node TPU: {e}")
                        return
                    raise

            try:
                import vllm.v1.engine.core as v1_core

                if not getattr(v1_core, "_verl_tpu_patched", False):
                    v1_core._verl_tpu_patched = True
                    global _orig_run_engine_core
                    raw_fn = v1_core.EngineCoreProc.run_engine_core
                    if hasattr(raw_fn, "__func__"):
                        raw_fn = raw_fn.__func__
                    _orig_run_engine_core = raw_fn
                    v1_core.EngineCoreProc._unpatched_run_engine_core = raw_fn
                    v1_core.EngineCoreProc.run_engine_core = staticmethod(_patched_run_engine_core)
                    v1_core.run_engine_core = _patched_run_engine_core
            except Exception as e3:
                logger.warning(f"Failed to patch v1_core.run_engine_core: {e3}")

            try:
                import vllm.v1.executor.ray_executor as v1_ray_executor

                v1_ray_executor.initialize_ray_cluster = patched_initialize_ray_cluster

                executor_cls = getattr(
                    v1_ray_executor, "RayDistributedExecutor", getattr(v1_ray_executor, "RayExecutor", None)
                )
                if executor_cls is not None and hasattr(executor_cls, "_init_workers_ray"):
                    orig_init_workers_ray = executor_cls._init_workers_ray

                    def patched_init_workers_ray(self, placement_group, **ray_remote_kwargs):
                        if not os.environ.get("VLLM_RAY_BUNDLE_INDICES"):
                            indices = [str(i) for i in range(len(placement_group.bundle_specs))]
                            if indices and len(indices) >= self.parallel_config.world_size:
                                os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
                                    indices[: self.parallel_config.world_size]
                                )
                        return orig_init_workers_ray(self, placement_group, **ray_remote_kwargs)

                    executor_cls._init_workers_ray = patched_init_workers_ray
            except Exception as e2:
                logger.warning(f"Failed to patch v1_ray_executor: {e2}")
        except Exception as e:
            logger.warning(f"Failed to patch v1_ray_utils: {e}")

        OriginalRayWorkerWrapper = ray_distributed_executor.RayWorkerWrapper
        original_init_worker = OriginalRayWorkerWrapper.init_worker
        original_wrapper_init = OriginalRayWorkerWrapper.__init__

        def patched_wrapper_init(self, *args, **kwargs):
            patch_vllm_for_tpu()
            try:
                local_ip = ray.util.get_node_ip_address()
                pod_map_str = os.environ.get("TPU_POD_IP_TO_SLICE")
                if pod_map_str:
                    pod_map = json.loads(pod_map_str)
                    current_slice = pod_map.get(local_ip)
                    if current_slice:
                        slice_ips = sorted([ip for ip, sl in pod_map.items() if sl == current_slice])
                        if local_ip in slice_ips:
                            worker_id = slice_ips.index(local_ip)
                            os.environ["TPU_WORKER_ID"] = str(worker_id)
                            os.environ["TPU_WORKER_HOSTNAMES"] = ",".join(slice_ips)
                            logger.info(
                                f"[TPU WORKER FIX] Node {local_ip} (slice {current_slice}) "
                                f"configured TPU_WORKER_ID={worker_id}, hostnames={slice_ips}"
                            )
            except Exception as e:
                logger.warning(f"Failed setting local TPU_WORKER_ID: {e}")
            return original_wrapper_init(self, *args, **kwargs)

        OriginalRayWorkerWrapper.__init__ = patched_wrapper_init

        def patched_setup_device_if_necessary(self):
            patch_vllm_for_tpu()
            if not getattr(self, "compiled_dag_cuda_device_set", False):
                try:
                    from vllm.platforms import current_platform

                    if current_platform.is_cuda() and hasattr(self.worker, "device") and self.worker.device is not None:
                        current_platform.set_device(self.worker.device)
                except Exception:
                    pass
                self.compiled_dag_cuda_device_set = True

        OriginalRayWorkerWrapper.setup_device_if_necessary = patched_setup_device_if_necessary

        def patched_init_worker(self, *args, **kwargs):
            patch_vllm_for_tpu()
            if "vllm.utils.import_utils" not in sys.modules:
                sys.modules["vllm.utils.import_utils"] = types.ModuleType("import_utils")
            sys.modules["vllm.utils.import_utils"].init_cached_hf_modules = lambda: None

            try:
                import vllm.model_executor.model_loader.weight_utils as vllm_weight_utils

                vllm_weight_utils.initialize_dummy_weights = lambda *args, **kwargs: None
            except Exception:
                pass
            try:
                import vllm.model_executor.model_loader.dummy_loader as vllm_dummy_loader

                vllm_dummy_loader.initialize_dummy_weights = lambda *args, **kwargs: None
            except Exception:
                pass

            self.__class__.load_weights_from_ray_registry = load_weights_from_ray_registry
            self.__class__.load_weights_from_state_dict_on_worker = lambda self, sd: load_weights_from_ray_registry(
                self, 0
            )
            self.__class__.reset_encoder_cache = dummy_reset_encoder_cache

            torch.set_grad_enabled(False)

            res = original_init_worker(self, *args, **kwargs)
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.reset_encoder_cache = dummy_reset_encoder_cache
                self.worker.__class__.reset_encoder_cache = dummy_reset_encoder_cache
            return res

        OriginalRayWorkerWrapper.init_worker = patched_init_worker

        def patched_init_workers_ray(self, placement_group, **ray_remote_kwargs):
            import inspect

            RayWorkerWrapper_local = ray_distributed_executor.RayWorkerWrapper
            TPU_TOPOLOGY_MAP_local = getattr(
                ray_distributed_executor,
                "TPU_TOPOLOGY_MAP",
                getattr(ray_distributed_executor, "TPU_MULTIHOST_TOPOLOGY_MAP", {}),
            )

            self.workers = []
            self.pp_tp_workers = []

            if self.parallel_config.ray_workers_use_nsight:
                ray_remote_kwargs = self._configure_ray_workers_use_nsight(ray_remote_kwargs)

            bundle_indices = []
            if vllm_envs is not None and vllm_envs.VLLM_RAY_BUNDLE_INDICES:
                bundle_indices = list(map(int, vllm_envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
                assert len(bundle_indices) == self.parallel_config.world_size, (
                    "VLLM_RAY_BUNDLE_INDICES must have the same size"
                    f" as the world size, but got {bundle_indices=} "
                    f"and {self.parallel_config.world_size=}"
                )
                assert len(set(bundle_indices)) == len(bundle_indices), (
                    f"VLLM_RAY_BUNDLE_INDICES cannot have duplicate values, but got {bundle_indices=}"
                )
            else:
                for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                    if current_platform is not None and bundle.get(current_platform.ray_device_key, 0):
                        bundle_indices.append(bundle_id)
                if not bundle_indices:
                    bundle_indices = list(
                        range(min(len(placement_group.bundle_specs), self.parallel_config.world_size))
                    )

            worker_metadata = []
            driver_ip = get_ip() if get_ip is not None else ""
            num_tpu_per_worker = 1
            for rank, bundle_id in enumerate(bundle_indices):
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_id,
                )
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={current_platform.ray_device_key: num_tpu_per_worker}
                    if current_platform is not None
                    else {},
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper_local).remote(rpc_rank=rank)
                worker_metadata.append(
                    RayWorkerMetaData(worker=worker, created_rank=rank) if RayWorkerMetaData is not None else None
                )

            worker_ips = ray.get([each.worker.get_node_ip.remote() for each in worker_metadata])

            for each, ip in zip(worker_metadata, worker_ips, strict=False):
                each.ip = ip

            logger.info(f"Initialized worker_metadata: {worker_metadata}")

            ip_counts = {}
            for ip in worker_ips:
                ip_counts[ip] = ip_counts.get(ip, 0) + 1

            def sort_by_driver_then_worker_ip(item):
                ip = item.ip
                return (0 if ip == driver_ip else 1, ip_counts[ip], ip)

            sorted_worker_metadata = sorted(worker_metadata, key=sort_by_driver_then_worker_ip)
            start_rank = 0
            for i, item in enumerate(sorted_worker_metadata):
                item.adjusted_rank = i + start_rank
            logger.info(f"Initialized sorted worker_metadata: {sorted_worker_metadata}")

            self.workers = [item.worker for item in sorted_worker_metadata]
            rerank_mapping = {item.created_rank: item.adjusted_rank for item in sorted_worker_metadata}
            self.collective_rpc("adjust_rank", args=(rerank_mapping,))

            worker_node_and_tpu_ids = []
            for worker in self.workers:
                if hasattr(worker, "get_node_and_gpu_ids"):
                    worker_node_and_tpu_ids.append(ray.get(worker.get_node_and_gpu_ids.remote()))
                else:
                    worker_node_and_tpu_ids.append(ray.get(worker.get_node_and_physical_gpu_ids.remote()))

            node_workers = defaultdict(list)
            node_tpus = defaultdict(list)

            for i, (node_id, tpu_ids) in enumerate(worker_node_and_tpu_ids):
                node_workers[node_id].append(i)
                tpu_ids = [int(x) for x in tpu_ids]
                node_tpus[node_id].extend(tpu_ids)
            for node_id, tpu_ids in node_tpus.items():
                node_tpus[node_id] = sorted(tpu_ids)
            logger.info(f"RayDistributedExecutor | node_workers={node_workers} | node_tpus={node_tpus}")

            all_ips = set(worker_ips + [driver_ip])
            n_ips = len(all_ips)
            n_nodes = len(node_workers)

            if n_nodes != n_ips:
                logger.warning(
                    f"Got {n_nodes} nodes but with {n_ips} IP addresses. "
                    "This is not a typical production setup whose "
                    "number of nodes and IPs is equal. This setup may "
                    "lead to unexpected behaviors."
                )

            unique_node_ids = list(node_workers.keys())
            num_nodes = len(unique_node_ids)

            sb_addresses = []
            base_port = int(os.environ.get("TORCH_TPU_BASE_PORT", TPU_ROLLOUT_BASE_PORT))
            for node_id in unique_node_ids:
                w_idx = node_workers[node_id][0]
                host_ip = sorted_worker_metadata[w_idx].ip
                chips_on_node = len(node_workers[node_id])
                for lr in range(chips_on_node):
                    sb_addresses.append(f"{host_ip}:{base_port + lr}")

            sb_addresses_str = ",".join(sb_addresses)
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = sb_addresses_str
            logger.info(f"Constructed TORCH_TPU_SLICEBUILDER_ADDRESSES: {sb_addresses_str}")

            total_chips = len(self.workers)
            topology, host_bounds, chips_per_host_bounds, chips_per_host = _resolve_tpu_topology_bounds(
                total_chips=total_chips,
                num_nodes=num_nodes,
                fallback_map=TPU_TOPOLOGY_MAP_local,
            )

            rank_0_node_id = unique_node_ids[0]
            rank_0_worker_index = node_workers[rank_0_node_id][0]
            master_addr = sorted_worker_metadata[rank_0_worker_index].ip
            master_port = str(get_open_port()) if get_open_port is not None else ""

            all_args_to_update_environment_variables = []
            for i in range(total_chips):
                node_id = worker_node_and_tpu_ids[i][0]
                node_rank = unique_node_ids.index(node_id)
                local_ws = len(node_workers[node_id]) or len(node_tpus[node_id]) or total_chips
                args = {
                    "NNODES": str(num_nodes),
                    "NODE_RANK": str(node_rank),
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": master_port,
                    "TORCH_TPU_TOPOLOGY": topology,
                    "LOCAL_WORLD_SIZE": str(local_ws),
                    "TPU_NUM_HOSTS": str(num_nodes),
                }
                if "TORCH_TPU_XPROF_SESSION_ID" not in os.environ:
                    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = str(time.time_ns())

                args["TORCH_TPU_XPROF_SESSION_ID"] = os.environ["TORCH_TPU_XPROF_SESSION_ID"]
                all_args_to_update_environment_variables.append(args)

            env_vars_to_copy_list = []
            if get_env_vars_to_copy is not None:
                exclude_vars = getattr(self, "WORKER_SPECIFIC_ENV_VARS", None)
                if exclude_vars is None:
                    from vllm.v1.executor import ray_utils as _ray_utils

                    exclude_vars = getattr(_ray_utils, "WORKER_SPECIFIC_ENV_VARS", set())
                env_vars_to_copy_list = get_env_vars_to_copy(
                    exclude_vars=exclude_vars,
                    additional_vars=set(current_platform.additional_env_vars)
                    if current_platform is not None
                    else set(),
                    destination="workers",
                )

            for i, args in enumerate(all_args_to_update_environment_variables):
                for name in env_vars_to_copy_list:
                    if name in os.environ:
                        args[name] = os.environ[name]
                logger.debug(f"RayDistributedExecutor | Worker {i} environment variables before patch: {args}")

            self._env_vars_for_all_workers = all_args_to_update_environment_variables

            try:
                unique_host_ips = [sorted_worker_metadata[node_workers[nid][0]].ip for nid in unique_node_ids]
                host_names_str = ",".join(unique_host_ips)
                for i, worker in enumerate(self.workers):
                    node_id = worker_node_and_tpu_ids[i][0]
                    host_idx = unique_node_ids.index(node_id)
                    local_chip_id = node_workers[node_id].index(i)
                    args = self._env_vars_for_all_workers[i]
                    args["RANK"] = str(i)
                    args["LOCAL_RANK"] = str(local_chip_id)
                    args["TPU_VISIBLE_CHIPS"] = str(local_chip_id)
                    args["TPU_VISIBLE_DEVICES"] = str(local_chip_id)
                    args["TPU_PROCESS_PORT"] = str(base_port + local_chip_id)
                    args["CLOUD_TPU_TASK_ID"] = str(i)
                    args["TPU_WORKER_HOSTNAMES"] = host_names_str
                    args["TPU_HOST_BOUNDS"] = host_bounds
                    args["TPU_PROCESS_BOUNDS"] = host_bounds
                    args["TPU_CHIPS_PER_HOST_BOUNDS"] = chips_per_host_bounds
                    args["TPU_CHIPS_PER_PROCESS_BOUNDS"] = chips_per_host_bounds
                    args["CHIPS_PER_HOST"] = chips_per_host
                    args["TORCH_TPU_TOPOLOGY"] = topology
                    args["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = sb_addresses_str
                    args["TPU_PROCESS_ADDRESSES"] = sb_addresses_str
                    if total_chips > 4 or num_nodes > 1:
                        args["TPU_MULTIHOST_BACKEND"] = "ray"

                    logger.info(
                        f"Configured TPU worker {i} (host {host_idx}, chip {local_chip_id}) env vars: "
                        f"TPU_VISIBLE_DEVICES={local_chip_id}, TPU_PROCESS_PORT={base_port + local_chip_id}, "
                        f"CLOUD_TPU_TASK_ID={i}, TORCH_TPU_TOPOLOGY={topology}"
                    )
            except Exception as patch_err:
                logger.warning(f"Failed to inject TPU worker env vars: {patch_err}")

            self.collective_rpc("update_environment_variables", args=(self._get_env_vars_to_be_updated(),))

            distributed_init_method = (
                get_distributed_init_method(driver_ip, get_open_port())
                if get_distributed_init_method is not None
                else ""
            )

            driver_node_id = ray.get_runtime_context().get_node_id()

            accepts_prev_worker_ip = True
            if TPUWorker is not None:
                try:
                    accepts_prev_worker_ip = "prev_worker_ip" in inspect.signature(TPUWorker.__init__).parameters
                except Exception:
                    pass

            accepts_assigned_physical_gpu_ids = False
            try:
                accepts_assigned_physical_gpu_ids = "assigned_physical_gpu_ids" in inspect.getsource(
                    original_init_worker
                )
            except Exception:
                pass

            all_kwargs = []
            for rank, (node_id, _) in enumerate(worker_node_and_tpu_ids):
                local_rank = node_workers[node_id].index(rank)
                ip = sorted_worker_metadata[rank].ip
                prev_ip = sorted_worker_metadata[rank - 1].ip if rank > 0 else ""

                worker_vllm_config = self.vllm_config

                if (
                    node_id != driver_node_id
                    and getattr(self.vllm_config, "model_config", None)
                    and getattr(self.vllm_config.model_config, "model_weights", None)
                ):
                    worker_vllm_config = copy.deepcopy(self.vllm_config)
                    worker_vllm_config.model_config.model = worker_vllm_config.model_config.model_weights
                    worker_vllm_config.model_config.model_weights = None

                kwargs = dict(
                    vllm_config=worker_vllm_config,
                    local_rank=local_rank,
                    rank=rank,
                    distributed_init_method=distributed_init_method,
                    is_driver_worker=(not self.parallel_config)
                    or (rank % self.parallel_config.tensor_parallel_size == 0),
                    ip=ip,
                )
                if accepts_prev_worker_ip:
                    kwargs["prev_worker_ip"] = prev_ip
                if accepts_assigned_physical_gpu_ids:
                    kwargs["assigned_physical_gpu_ids"] = sorted(node_tpus[node_id])
                all_kwargs.append(kwargs)
            self.collective_rpc("init_worker", args=(all_kwargs,))
            self.collective_rpc("init_device")
            if self.parallel_config.pipeline_parallel_size > 1:
                self.collective_rpc("initialize_pp_transfer_connect")
            self.collective_rpc("load_model")
            if hasattr(self, "pp_tp_workers"):
                self.pp_tp_workers = []
                pp_size = self.parallel_config.pipeline_parallel_size if self.parallel_config else 1
                tp_size = self.parallel_config.tensor_parallel_size if self.parallel_config else len(self.workers)
                for pp_rank in range(pp_size):
                    self.pp_tp_workers.append([])
                    for tp_rank in range(tp_size):
                        rank = (pp_rank * tp_size) + tp_rank
                        if rank < len(self.workers):
                            self.pp_tp_workers[pp_rank].append(self.workers[rank])

        def _detach_zero_copy(output):
            try:
                from vllm.v1.executor.ray_utils import detach_zero_copy_from_model_runner_output
            except ImportError:
                return
            detach_zero_copy_from_model_runner_output(output)

        def patched_execute_dag(
            self,
            scheduler_output,
            grammar_output,
            non_block: bool = False,
        ):
            refs = [worker.execute_model_ray.remote((scheduler_output, grammar_output)) for worker in self.workers]
            if not self.has_connector:
                if not non_block:
                    all_results = ray.get(refs)
                    _detach_zero_copy(all_results[0])
                    return all_results[0]
                from vllm.v1.executor.ray_utils import FutureWrapper

                return FutureWrapper(refs[0])

            assert self.kv_output_aggregator is not None
            if not non_block:
                outputs = ray.get(refs)
                for output in outputs:
                    _detach_zero_copy(output)
                return self.kv_output_aggregator.aggregate(outputs)
            from vllm.v1.executor.ray_utils import FutureWrapper

            return FutureWrapper(refs, self.kv_output_aggregator)

        if ray_distributed_executor is not None:
            ray_distributed_executor.RayDistributedExecutor._init_workers_ray = patched_init_workers_ray
        try:
            import vllm.executor.ray_distributed_executor as vllm_ray_dist_exec

            vllm_ray_dist_exec.RayDistributedExecutor._init_workers_ray = patched_init_workers_ray
        except Exception:
            pass
        try:
            import vllm.v1.executor.ray_executor as vllm_v1_ray_exec

            vllm_v1_ray_exec.RayDistributedExecutor._init_workers_ray = patched_init_workers_ray
            vllm_v1_ray_exec.RayDistributedExecutor._execute_dag = patched_execute_dag
        except Exception:
            pass

        logger.info("Successfully applied all TPU patches and hacks to vLLM & torchtpu-vllm")
    except Exception as e:
        logger.warning(f"Failed to apply TPU patches: {e}")


# Helper helpers to import from torchtpu-vllm and other places inside vllm_async_server
try:
    from vllm_torchtpu.platforms.tpu_platform import get_env_vars_to_copy
except ImportError:
    try:
        from vllm.ray.ray_env import get_env_vars_to_copy
    except ImportError:
        get_env_vars_to_copy = None


def is_tpu_vllm_run() -> bool:
    """Returns True if executing on a Google TPU resource or with V1 explicitly disabled."""
    return get_resource_name() == "TPU" or os.environ.get("VLLM_USE_V1") == "0"


def override_vllm_configs_for_tpu(args_or_config: Any):
    """Enforces VLLM_USE_V1=0 and use_v1=False across dictionary and namespace objects on TPU."""
    os.environ["VLLM_USE_V1"] = "0"
    try:
        import vllm.envs as vllm_envs

        vllm_envs.VLLM_USE_V1 = False
    except Exception:
        pass

    if isinstance(args_or_config, dict):
        args_or_config["use_v1"] = False
        return

    for obj in [args_or_config, getattr(args_or_config, "model_config", None)]:
        if obj is None:
            continue
        for attr in ["use_v1", "_use_v1"]:
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, False)
                except Exception:
                    pass
            if hasattr(obj, "__dict__") and attr in obj.__dict__:
                try:
                    obj.__dict__[attr] = False
                except Exception:
                    pass


async def get_tpu_server_launch_config(workers):
    """
    Asynchronously queries node ID, visible chips, and TPU specific environment
    variables from all TPU workers for launching the server actor on TPU.
    """
    worker_infos = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: (
                    ray.get_runtime_context().get_node_id(),
                    os.environ.get("TPU_VISIBLE_CHIPS", "0"),
                )
            )
            for worker in workers
        ]
    )

    worker_tpu_envs = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith("TPU_")
                    or k.startswith("TORCH_TPU_")
                    or k
                    in (
                        "CLOUD_TPU_TASK_ID",
                        "CHIPS_PER_HOST",
                        "JAX_MEM_FRACTION",
                        "JAX_THREE_G_MEM_ALLOC_ON_FREE",
                        "XLA_PYTHON_CLIENT_PREALLOCATE",
                        "XLA_PYTHON_CLIENT_MEM_FRACTION",
                        "LIBTPU_INIT_ARGS",
                        "TORCH_DYNAMO_RECOMPILE_LIMIT",
                        "SKIP_JAX_PRECOMPILE",
                        "VLLM_ENABLE_V1_MULTIPROCESSING",
                        # Keep the vLLM AOT compile cache disabled in the server
                        # process too: a reloaded artifact degrades the model to
                        # eager execution and reintroduces the unaligned-DUS
                        # crash (b/501165531). See apply_tpu_vllm_patches().
                        "VLLM_DISABLE_COMPILE_CACHE",
                        "VERL_PLATFORM",
                        "XLA_FLAGS",
                    )
                }
            )
            for worker in workers
        ]
    )

    node_id = worker_infos[0][0]
    visible_chips = ",".join([info[1] for info in worker_infos])
    tpu_env_vars = worker_tpu_envs[0] if worker_tpu_envs else {}

    return node_id, visible_chips, tpu_env_vars


# =============================================================================
# Preflight census of leaked vLLM engine processes.
#
# Every rollout engine that shuts down leaves processes in state Z, reparented
# to PID 1, one of them VLLM::EngineCore. Nothing reaps them, so the count grows
# for the lifetime of the pod. Past some point the host stops handing out TPU
# devices and the next job dies ~8k log lines into engine init with an opaque
# "local device count is 0". Naming the host up front turns that into a lead.
#
# Enforcement is off by default. The count correlates with failures (0, 1 and 4
# passed; 5 and ~10 failed) but the threshold rests on too little data to fail a
# good run over, so the census reports unless asked otherwise:
#
#   VERL_TPU_MAX_STALE_ENGINES unset, or -1   (default) report only, never fail
#   VERL_TPU_MAX_STALE_ENGINES=0              fail if any host leaked even one
#   VERL_TPU_MAX_STALE_ENGINES=4              fail if a host leaked more than 4
#
# Negative is the "off" sentinel rather than 0, so 0 stays usable as a real
# threshold. There is deliberately no wait loop: zombies never clear on their
# own, so waiting could only delay the failure.
#
# NOTE: do not name the libtpu fusion CHECK in any message emitted from here. An
# earlier revision did, and log scrapers searching for the crash matched this
# check's output instead.
# =============================================================================


def probe_stale_tpu_engines(self=None):
    """Return a zombie-process census for this host.

    Executed remotely on each TPU worker actor via ``__ray_call__``, hence the
    unused ``self`` parameter and the function-local imports: the callable is
    pickled and must not depend on the caller's module state.

    Returns a dict with ``hostname``, ``zombies`` (every process in state ``Z``)
    and ``engines`` (those whose command name looks like a vLLM EngineCore).

    ``/proc/<pid>/cmdline`` is empty for a zombie -- that is why ``ps`` renders
    them as ``<defunct>`` -- so the name has to come from the ``comm`` field of
    ``/proc/<pid>/stat`` instead.
    """
    import os as _os
    import socket as _socket

    # /proc/<pid>/stat truncates comm to 15 characters, so "VLLM::EngineCore"
    # arrives as "VLLM::EngineCor". Match on the truncated form.
    engine_marker = "EngineCor"

    census = {"hostname": _socket.gethostname(), "zombies": 0, "engines": 0}

    try:
        entries = _os.listdir("/proc")
    except OSError:
        return census

    for entry in entries:
        if not entry.isdigit():
            continue

        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                stat_line = handle.read().decode(errors="replace")
        except OSError:
            # Process exited mid-scan, or /proc is not readable. Not actionable.
            continue

        # comm sits between the first '(' and the last ')' and may itself
        # contain spaces or parentheses, so anchor on the LAST ')'.
        close = stat_line.rfind(")")
        if close == -1:
            continue

        fields = stat_line[close + 1 :].split()
        if not fields or fields[0] != "Z":
            continue

        census["zombies"] += 1

        opened = stat_line.find("(")
        comm = stat_line[opened + 1 : close] if opened != -1 else ""
        if engine_marker in comm:
            census["engines"] += 1

    return census


async def report_stale_tpu_engines(workers) -> None:
    """Log a per-host census of leaked vLLM engine processes before launching servers.

    Reports only, unless ``VERL_TPU_MAX_STALE_ENGINES`` is set to a non-negative
    value, in which case a host holding more than that many leaked engines
    raises before any TPU work starts. Unset or negative disables enforcement;
    ``0`` means "fail if any host has leaked even one". Negative is the off
    sentinel rather than ``0`` so that ``0`` stays usable as a real threshold.

    This never blocks, and deliberately does not wait for the count to come
    down. Waiting would assume the stragglers eventually clear; these are
    zombies, so nothing short of recycling the pod removes them and a wait could
    only delay the failure.

    Args:
        workers: TPU worker actor handles. There is one per chip, so several of
            them report the same host; the census is deduplicated by hostname.

    Raises:
        RuntimeError: Only when enforcement is enabled and a host is over the limit.
    """
    if not is_tpu_vllm_run() or not workers:
        return

    raw_limit = os.environ.get("VERL_TPU_MAX_STALE_ENGINES", "-1")
    try:
        limit = int(raw_limit)
    except ValueError:
        _tpu_preflight_log(f"ignoring malformed VERL_TPU_MAX_STALE_ENGINES={raw_limit!r}, staying report-only")
        limit = -1

    try:
        per_actor = await asyncio.gather(*[worker.__ray_call__.remote(probe_stale_tpu_engines) for worker in workers])
    except Exception as probe_err:
        # The census is a diagnostic aid; never let it be the thing that breaks
        # a run that would otherwise have worked.
        _tpu_preflight_log(f"census failed, continuing without preflight check: {probe_err}")
        return

    by_host = {}
    for census in per_actor:
        if isinstance(census, dict) and census.get("hostname"):
            by_host[census["hostname"]] = census

    if not by_host:
        return

    summary = ", ".join(
        f"{host}: {by_host[host]['engines']} engine(s) / {by_host[host]['zombies']} zombie(s)"
        for host in sorted(by_host)
    )
    _tpu_preflight_log(f"leaked rollout engines across {len(by_host)} host(s) -- {summary}")

    if limit < 0:
        return

    over = sorted(host for host, census in by_host.items() if census["engines"] > limit)
    if not over:
        return

    raise RuntimeError(
        f"[TPU preflight] {len(over)}/{len(by_host)} TPU host(s) exceed "
        f"VERL_TPU_MAX_STALE_ENGINES={limit}: {', '.join(over)}. Leaked engine processes "
        "accumulate for the lifetime of the pod and are never reaped. Past some point the "
        "host stops handing out TPU devices and vLLM engine init dies with 'local device "
        "count is 0' about 8k log lines in. Recycle the workers and resubmit:\n"
        "    kubectl delete pod -l ray.io/cluster=<your-cluster>,ray.io/node-type=worker\n"
        f"Census: {summary}"
    )


def _tpu_preflight_log(message: str, tag: str = "TPU preflight") -> None:
    """Emit a preflight message via both logging and stdout."""
    logging.getLogger(__name__).warning("[%s] %s", tag, message)
    print(f"[{tag}] {message}", flush=True)


def prepare_tpu_server_args(args: dict):
    """Configures TPU-specific CLI/server arguments and environment variables."""
    if not is_tpu_vllm_run():
        return

    args["enable_sleep_mode"] = False
    args["distributed_executor_backend"] = "external_launcher"
    os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
    os.environ["VLLM_USE_RAY_V2_EXECUTOR_BACKEND"] = "0"

    try:
        import vllm_torchtpu.envs as tpu_envs

        tpu_envs.TPU_MULTIHOST_BACKEND = "ray"
        if hasattr(tpu_envs, "__getattr__") and hasattr(tpu_envs.__getattr__, "cache_clear"):
            tpu_envs.__getattr__.cache_clear()
    except Exception as env_err:
        logging.getLogger(__name__).warning(f"Failed to force TPU_MULTIHOST_BACKEND to ray: {env_err}")


async def launch_tpu_vllm_servers(replica) -> None:
    """Launch vLLM rollout server actor for a TPU replica (`vLLMReplica`)."""
    from verl.plugin.platform import get_platform
    from verl.utils.net_utils import is_valid_ipv6_address

    if replica.config.data_parallel_size > 1:
        raise NotImplementedError(
            "actor_rollout_ref.rollout.data_parallel_size="
            f"{replica.config.data_parallel_size} is not supported on TPU. The TPU "
            "distributed runtime requires each compiled program to span the whole "
            "TPU mesh, but a single DP group only spans "
            f"tensor_model_parallel_size={replica.config.tensor_model_parallel_size} "
            "chips. Set actor_rollout_ref.rollout.data_parallel_size=1 instead: verl "
            "then creates one rollout replica per tensor_model_parallel_size chips, "
            "which is equivalent to engine-internal data parallelism."
        )

    await report_stale_tpu_engines(replica.workers)

    node_id, visible_chips, tpu_env_vars = await get_tpu_server_launch_config(replica.workers)

    prefix = replica._get_server_name_prefix() if hasattr(replica, "_get_server_name_prefix") else "vllm_"
    if replica.is_reward_model:
        name = f"{prefix}server_reward_{replica.replica_rank}_0{replica.name_suffix}"
    elif replica.is_teacher_model:
        name = f"{prefix}server_teacher_{replica.replica_rank}_0{replica.name_suffix}"
    else:
        name = f"{prefix}server_{replica.replica_rank}_0{replica.name_suffix}"

    platform_env_vars = get_platform().rollout_env_vars()
    env_vars = {
        **{var: "1" for var in get_platform().ray_noset_envvars()},
        **platform_env_vars,
        **tpu_env_vars,
    }
    if "VERL_PLATFORM" in os.environ:
        env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]

    flags_to_copy = set()
    for flag_var in ("XLA_FLAGS", "LIBTPU_INIT_ARGS"):
        base_value = platform_env_vars.get(flag_var) or tpu_env_vars.get(flag_var)
        extra_value = os.environ.get(f"VERL_TPU_EXTRA_{flag_var}")
        resolved = " ".join(v for v in (base_value, extra_value) if v)
        _tpu_preflight_log(
            f"{flag_var}: base={base_value!r} extra={extra_value!r} -> engine={resolved or None!r}",
            tag="TPU env",
        )
        if resolved:
            env_vars[flag_var] = resolved
            flags_to_copy.add(flag_var)

    if flags_to_copy:
        copy_var = "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"
        existing = env_vars.get(copy_var) or os.environ.get(copy_var) or ""
        names = {tok.strip() for tok in existing.split(",") if tok.strip()}
        env_vars[copy_var] = ",".join(sorted(names | flags_to_copy))
        _tpu_preflight_log(f"{copy_var}={env_vars[copy_var]}", tag="TPU env")

    server = replica.server_class.options(
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False,
        ),
        runtime_env={"env_vars": env_vars},
        name=name,
        max_concurrency=replica.max_concurrency,
    ).remote(
        config=replica.config,
        model_config=replica.model_config,
        rollout_mode=replica.rollout_mode,
        workers=replica.workers,
        replica_rank=replica.replica_rank,
        node_rank=0,
        gpus_per_node=replica.gpus_per_replica_node,
        nnodes=replica.nnodes,
        cuda_visible_devices=visible_chips,
    )
    replica.servers.append(server)

    master_address, master_port, dp_rpc_port = await replica.servers[0].get_master_address.remote()
    await replica.servers[0].launch_server.remote(
        master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
    )

    server_address, server_port = await replica.servers[0].get_server_address.remote()
    replica._server_handle = replica.servers[0]
    replica._server_address = (
        f"[{server_address}]:{server_port}"
        if is_valid_ipv6_address(server_address)
        else f"{server_address}:{server_port}"
    )
