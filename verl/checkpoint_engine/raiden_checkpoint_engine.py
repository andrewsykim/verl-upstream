# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Google LLC
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
import logging
import os
import time
from typing import Any, Generator, Iterable, Optional

import ray
import torch

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
)

logger = logging.getLogger(__name__)


def compute_tensor_stats(items) -> dict:
    """Compute deterministic L1/L2 norms and parameter counts across tensors.
    Reductions are performed directly on device (TPU) to avoid copying full tensor weights across PCIe.
    """
    total_numel = 0
    total_l1 = 0.0
    total_l2_sq = 0.0
    per_tensor = {}

    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            name, p = item
        else:
            name = None
            p = item
        p_local = p.to_local() if hasattr(p, "to_local") else p

        # Perform reductions directly on device (TPU HBM) in float32 for precision and speed
        p_float = p_local.float()
        t_l1 = float(p_float.abs().sum().item())
        t_l2_sq = float(p_float.pow(2).sum().item())
        numel = p_local.numel()

        if name is not None:
            per_tensor[name] = {
                "l1": t_l1,
                "l2": float(t_l2_sq**0.5),
                "l2_sq": t_l2_sq,
                "numel": numel,
                "shape": list(p_local.shape),
                "dtype": str(p_local.dtype),
            }
        total_numel += numel
        total_l1 += t_l1
        total_l2_sq += t_l2_sq

    return {
        "total_numel": total_numel,
        "num_tensors": len(items),
        "l1_norm": total_l1,
        "l2_norm": float(total_l2_sq**0.5),
        "l2_sq": total_l2_sq,
        "per_tensor": per_tensor,
    }


def create_torch_weight_synchronizer(
    device_tensors: list[list[torch.Tensor]],
    local_port: int = 0,
    parallelism: int = 8,
    listener_port: int = 0,
    bind_ip: str = "127.0.0.1",
):
    """Creates an instance of WeightSynchronizer using tpu_sync."""
    from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer

    return WeightSynchronizer(
        device_tensors,
        local_port=local_port,
        parallelism=parallelism,
        listener_port=listener_port,
        bind_ip=bind_ip,
        unsafe_skip_buffer_lock=True,
        auto_h2d=False,
    )


def _unwrap_tensor(t: Any) -> Any:
    """Extract the raw underlying local tensor from nn.Parameter or DTensor wrappers."""
    t = t.to_local().data if hasattr(t, "to_local") else (t.data if hasattr(t, "data") else t)
    return t


def filter_tied_embeddings(named_items: Iterable[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Exclude redundant lm_head weights if embedding tokens are present."""
    items = list(named_items)
    has_embed = any("embed_tokens" in k or "tok_embeddings" in k for k, _ in items)
    if has_embed:
        items = [(k, v) for k, v in items if not (k == "lm_head.weight" or k.endswith(".lm_head.weight"))]
    return items


def validate_and_sanitize_tensors(
    named_tensors: Iterable[tuple[str, Any]],
    device: Optional[torch.device] = None,
) -> list[tuple[str, torch.Tensor]]:
    """Validate and sanitize tensors for zero-copy DMA registration with Raiden.

    Performs physical memory validation:
    1. Drops None, non-tensor objects, zero-element tensors, and unallocated meta tensors.
    2. Unwraps DTensor/nn.Parameter to local tensor buffers.
    3. Ensures tensors physically reside in TPU HBM.
    4. Guarantees memory contiguity (contiguous buffers) required for direct DMA.
    """
    if device is None:
        device = torch.device("tpu")

    sanitized = []
    for item in named_tensors:
        name, p = item[0], item[1]
        if p is None:
            continue
        t = _unwrap_tensor(p)
        if not isinstance(t, torch.Tensor):
            continue

        # Skip 0-element tensors and unallocated meta tensors (prevents C++ nullptr faults)
        if t.numel() == 0 or getattr(t, "is_meta", False):
            continue

        # Ensure tensor physically resides in TPU device memory
        if not (hasattr(t, "device") and str(t.device).startswith("tpu")):
            try:
                t = t.to(device)
            except Exception as e:
                logger.warning(f"Could not move {name} to TPU: {e}")
                continue

        # Ensure contiguous memory layout (required for direct DMA pointer calculation)
        if not t.is_contiguous():
            t = t.contiguous()

        sanitized.append((name, t))

    return sanitized


def _dim0_shard_info(spec) -> Optional[tuple[tuple, int, int]]:
    """Return ``(local_shape, num_shards, shard_index)`` if the local shard is an even dim-0 cut on a 1-D mesh.

    That is the FSDP2 ``Shard(0)`` layout, which Raiden can describe directly: the variable keeps its full
    shape, is split ``num_shards`` ways on dim 0, and this rank holds block ``shard_index``. Uneven cuts
    are excluded because torch gives the remainder to the leading shards while Raiden gives it to the last.
    """
    from verl.workers.engine.spec import BlockPlacement, derive_dtensor_placement

    if spec.mesh is None or spec.mesh.ndim != 1:
        return None
    place, _, _ = derive_dtensor_placement(spec)
    if not isinstance(place, BlockPlacement) or not place.is_flat_contiguous:
        return None
    num_shards = spec.mesh.size(0)
    full_dim0 = int(spec.full_shape[0])
    if num_shards <= 1 or full_dim0 % num_shards:
        return None
    return tuple(place.local_shape), num_shards, int(place.global_offset[0]) // (full_dim0 // num_shards)


def export_local_shards(engine) -> tuple[list[tuple[str, torch.Tensor]], dict[str, tuple[int, int]]]:
    """Export this rank's weights for Raiden without all-gathering FSDP shards.

    Returns ``(named_tensors, shard_info)``. ``shard_info[name] = (num_shards, shard_index)`` for tensors
    exported as a local dim-0 shard; tensors absent from it are full (replicated) on every rank. Tensors
    whose layout Raiden cannot express are all-gathered here, in the same order on every rank.
    """
    from torch.distributed.tensor import DTensor

    from verl.workers.engine.spec import BlockPlacement, derive_dtensor_placement

    gen, _ = engine.get_per_tensor_param_shard()
    items = filter_tied_embeddings((name, (local, spec)) for name, local, spec in gen)

    named, shard_info = [], {}
    for name, (local, spec) in items:
        if spec.place is not None or spec.hf_slots is not None:
            raise NotImplementedError(
                f"Raiden sharded export does not support exporter-defined placements or expert stacks ({name})"
            )
        full_shape = tuple(int(d) for d in spec.full_shape)
        info = _dim0_shard_info(spec)
        if info is not None:
            local_shape, num_shards, shard_index = info
            named.append((name, local.view(local_shape)))
            shard_info[name] = (num_shards, shard_index)
            continue
        place = derive_dtensor_placement(spec)[0] if spec.mesh is not None else 0
        if not isinstance(place, BlockPlacement):
            # unsharded or fully replicated: the local tensor is already the full parameter
            named.append((name, local.view(full_shape)))
            continue
        strides = [1] * len(full_shape)
        for d in range(len(full_shape) - 2, -1, -1):
            strides[d] = strides[d + 1] * full_shape[d + 1]
        dt = DTensor.from_local(
            local.view(place.local_shape),
            spec.mesh,
            spec.placements,
            run_check=False,
            shape=torch.Size(full_shape),
            stride=tuple(strides),
        )
        named.append((name, dt.full_tensor()))
    return named, shard_info


def merge_rank_stats(rank_stats: dict) -> dict:
    """Sum per-tensor partial stats (l1, l2_sq, numel) posted by every trainer rank."""
    per_tensor = {}
    for stats in rank_stats.values():
        for name, t in stats.get("per_tensor", {}).items():
            acc = per_tensor.setdefault(name, {"l1": 0.0, "l2_sq": 0.0, "numel": 0})
            acc["l1"] += t["l1"]
            acc["l2_sq"] += t["l2_sq"]
            acc["numel"] += t["numel"]
    for acc in per_tensor.values():
        acc["l2"] = acc["l2_sq"] ** 0.5
    total_l2_sq = sum(t["l2_sq"] for t in per_tensor.values())
    return {
        "total_numel": sum(t["numel"] for t in per_tensor.values()),
        "num_tensors": len(per_tensor),
        "l1_norm": sum(t["l1"] for t in per_tensor.values()),
        "l2_norm": total_l2_sq**0.5,
        "l2_sq": total_l2_sq,
        "per_tensor": per_tensor,
    }


def setup_raiden_controller() -> tuple[Any, Any, str]:
    """Start embedded RaidenControllerServer on the head node and record its address in TPUWeightRegistry."""
    from tpu_sync.rpc import raiden_controller

    from verl.checkpoint_engine.tpu_weight_registry import get_tpu_weight_registry

    controller = raiden_controller.RaidenController(port=0)
    server = raiden_controller.RaidenControllerServer(controller)
    port = server.start()
    ip = ray.util.get_node_ip_address().strip("[]")
    address = f"{ip}:{port}"
    logger.info(f"RaidenControllerServer started on Headnode: {address}")

    try:
        registry = get_tpu_weight_registry()
        ray.get(registry.set_controller_address.remote(address))
        logger.info(f"Successfully stored RaidenController address ({address}) in TPUWeightRegistry")
    except Exception as reg_err:
        raise RuntimeError(
            f"Failed to store RaidenController address ({address}) in TPUWeightRegistry: {reg_err}"
        ) from reg_err

    return controller, server, address


@CheckpointEngineRegistry.register("raiden")
class RaidenCheckpointEngine(CheckpointEngine):
    """P2P Weight Synchronizer Checkpoint Engine for TPUs using Google Raiden (tpu-sync)."""

    # Receive the training engine in send_weights so each rank can register its local FSDP shard
    # (Raiden reshards to the sampler layout) instead of all-gathering full tensors.
    consumes_training_engine = True

    def __init__(self, bucket_size: int = 0, is_master: bool = False, **kwargs) -> None:
        self.is_master = is_master
        self.bucket_size = bucket_size
        self.backend = "raiden"
        self.verify_parity = kwargs.get("verify_parity", False)
        self.parallelism = kwargs.get("parallelism", 8)
        self.tp_size = kwargs.get("tp_size", kwargs.get("tensor_parallel_size", self.parallelism))
        self._trainer_raiden_ws = None
        self._trainer_chunks = []
        self._controller_addr = None
        self._registered_signature = None
        self._bound_tensors = None
        self._shard_info = {}
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = int(os.environ.get("RANK", "0"))

        from verl.checkpoint_engine.tpu_weight_registry import get_tpu_weight_registry

        self.registry = get_tpu_weight_registry()

    def prepare(self) -> dict:
        return {}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        return {}, {}

    def init_process_group(self, **kwargs):
        pass

    def finalize(self):
        if self._trainer_raiden_ws is not None:
            try:
                self._trainer_raiden_ws.close()
            except Exception:
                pass
            self._trainer_raiden_ws = None
        self._registered_signature = None
        self._bound_tensors = None

    async def _create_and_register(self, valid_weights: list[tuple[str, torch.Tensor]]) -> None:
        """(Re)create the trainer WeightSynchronizer on ``valid_weights`` and register it with the controller."""
        if self._trainer_raiden_ws is not None:
            try:
                self._trainer_raiden_ws.close()
            except Exception:
                pass
            self._trainer_raiden_ws = None

        logger.info(f"Trainer Rank {self.rank}: binding {len(valid_weights)} tensors to WeightSynchronizer ")
        bind_ip = ray.util.get_node_ip_address().strip("[]")
        self._trainer_raiden_ws = create_torch_weight_synchronizer(
            [[t] for _, t in valid_weights],
            local_port=0,
            listener_port=0,
            parallelism=getattr(self, "parallelism", 8),
            bind_ip=bind_ip,
        )

        def _full_shape(name: str, p: torch.Tensor) -> list[int]:
            shape = list(p.shape)
            if name in self._shard_info:
                shape[0] *= self._shard_info[name][0]
            return shape

        # Record global shapes to TPUWeightRegistry so Sampler can look them up
        # TODO(tpu): Move global_shapes registration to a one-time setup step during initialization
        # (e.g., in prepare or build_process_group/model_init) instead of per weight-sync step,
        # since tensor shapes are static across training iterations and only need to be communicated once.
        if self.registry is not None:
            try:
                global_shapes = {name: _full_shape(name, p) for name, p in valid_weights}
                ray.get(self.registry.set_global_shapes.remote(global_shapes))
            except Exception as e:
                logger.warning(f"Could not record global shapes in TPUWeightRegistry: {e}")

        # Build variable metadata protos for each dynamic tensor. Every variable carries its full shape.
        # - Local dim-0 shard: mesh_shape=[N, 1, ...] and global_shard_indices=[i] (this rank holds block i
        #   of N); Raiden reshards it to the sampler layout.
        # - Full tensor (replicated on every rank): mesh_shape=[1] * rank, unsharded.
        variable_protos = []
        from tpu_sync.rpc import raiden_service_pb2

        max_shards = 1
        for idx, (name, p) in enumerate(valid_weights):
            shape = _full_shape(name, p)
            itemsize = p.element_size()
            layout = list(range(len(shape) - 1, -1, -1))
            if name in self._shard_info:
                num_shards, shard_index = self._shard_info[name]
                max_shards = max(max_shards, num_shards)
                extra = {
                    "mesh_shape": [num_shards] + [1] * (len(shape) - 1),
                    "sharding_spec": ["fsdp"] + [""] * (len(shape) - 1),
                    "global_shard_indices": [shard_index],
                }
            else:
                extra = {"mesh_shape": [1] * len(shape), "sharding_spec": [""] * len(shape)}
            variable_protos.append(
                raiden_service_pb2.VariableMetadataProto(
                    name=name,
                    shape=shape,
                    layout=layout,
                    item_size=itemsize,
                    layer_idx=idx,
                    **extra,
                )
            )
        num_sharded = sum(name in self._shard_info for name, _ in valid_weights)
        logger.info(
            f"Trainer Rank {self.rank}: registering {num_sharded} sharded and "
            f"{len(valid_weights) - num_sharded} full tensors"
        )

        # TODO(tpu): Consider passing controller_address directly during orchestration (e.g., via
        # CheckpointEngineManager / actor_wg.update_weights or init_process_group) rather than querying
        # the TPUWeightRegistry actor, eliminating cross-process registry lookups altogether.
        if self._controller_addr is None and self.registry is not None:
            for _ in range(30):
                try:
                    self._controller_addr = ray.get(self.registry.get_controller_address.remote())
                    if self._controller_addr:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.1)

        if self._controller_addr:
            from tpu_sync.rpc import raiden_controller

            try:
                ctrl_client = raiden_controller.RaidenControllerClientFacade(self._controller_addr)
                unit_id = raiden_controller.RaidenId("trainer", str(self.rank), "weights")
                ctrl_client.register_work_unit(
                    unit_id,
                    [f"{bind_ip}:{self._trainer_raiden_ws.local_port}"],
                    f"{bind_ip}:{self._trainer_raiden_ws.listener_port}",
                    mesh_shape=[max_shards, 1],
                    variables=variable_protos,
                    mesh_axes=["fsdp", "tp"],
                )
                logger.info(
                    f"Trainer Rank {self.rank} bound {len(valid_weights)} dynamic tensors and registered "
                    f"directly with RaidenController ({self._controller_addr}): "
                    f"data_port={self._trainer_raiden_ws.local_port}, "
                    f"listener_port={self._trainer_raiden_ws.listener_port}"
                )
            except Exception as reg_err:
                logger.error(
                    f"Trainer Rank {self.rank} failed to register with RaidenController "
                    f"({self._controller_addr}): {reg_err}"
                )
                raise
        else:
            raise RuntimeError(
                f"Trainer Rank {self.rank}: No RaidenController address found in TPUWeightRegistry after timeout"
            )

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None]
        | Iterable[tuple[str, torch.Tensor]]
        | dict[str, torch.Tensor],
        dst_peers: Optional[list[str]] = None,
        global_steps: Optional[int] = None,
        compute_stats: Optional[bool] = None,
        compute_checksum: Optional[bool] = None,
        **kwargs,
    ):
        """Register weights with RaidenController and prepare for coordinated P2P network transfer."""
        if compute_stats is None:
            compute_stats = compute_checksum if compute_checksum is not None else getattr(self, "verify_parity", False)
        step_key = global_steps if global_steps is not None else 0
        logger.info(f"RaidenCheckpointEngine: [Step {step_key}] Start send_weights...")

        try:
            from torch_tpu._internal import sync as torch_tpu_sync

            torch_tpu_sync.synchronize(wait=True)
        except Exception as e:
            logger.warning(f"Could not synchronize via torch_tpu: {e}")
            raise RuntimeError(f"TPU synchronization failed: {e}") from e

        if hasattr(weights, "get_per_tensor_param_shard"):
            # Handed the training engine: register each rank's local FSDP shard, no all-gather.
            named_weights, self._shard_info = export_local_shards(weights)
        else:
            named_weights = filter_tied_embeddings(weights.items() if hasattr(weights, "items") else weights)
            self._shard_info = {}

        # Trainer sends pure canonical un-fused model weights directly
        unfused_weights = {k: _unwrap_tensor(v) for k, v in named_weights}

        sorted_weights = sorted(unfused_weights.items(), key=lambda x: x[0])
        valid_weights = validate_and_sanitize_tensors(sorted_weights, device=torch.device("tpu"))

        torch_tpu_sync.synchronize(wait=True)

        # Reuse the WeightSynchronizer across steps when the tensor signature is unchanged: rebinding keeps
        # the DMA-mapped host buffers, listener/threads, and controller registration, and only swaps the
        # device buffers the D2H reads from. Rebuilding every step re-allocates, pins, and first-touches
        # a model-sized host buffer per rank (~1s/step at 0.6B). Rebuild only on first use or if the
        # (name, shape, dtype, shard) signature changes.
        signature = [(name, tuple(t.shape), t.dtype, self._shard_info.get(name)) for name, t in valid_weights]
        if self._trainer_raiden_ws is not None and signature == self._registered_signature:
            self._trainer_raiden_ws.bind_weights([[t] for _, t in valid_weights])
        else:
            await self._create_and_register(valid_weights)
            self._registered_signature = signature
        # Keep the bound device buffers alive until the next sync: the controller-driven push reads them
        # after send_weights returns.
        self._bound_tensors = [t for _, t in valid_weights]

        # No explicit ws.d2h() here: PushWeightsResharded (triggered by the controller's start_transfer)
        # performs a pipelined D2H overlapped with H2H, using the per-transfer skip_tiling plan. An
        # explicit d2h() before the first transfer has no skip_tiling plan yet and stages the wrong
        # (detiled) layout; it was only ever masked by the push's own D2H.

        if not compute_stats or self.registry is None:
            return
        try:
            if self._shard_info:
                # Every rank posts partial stats: its local shards, plus the full tensors on rank 0 only.
                # The orchestrator sums them per tensor.
                items = [(n, t) for n, t in valid_weights if n in self._shard_info or self.rank == 0]
                trainer_stats = compute_tensor_stats(items)
                trainer_stats["rank"] = self.rank
                self.registry.set_rank_stats.remote(step_key, self.rank, trainer_stats)
            elif self.is_master:
                # Every rank holds the full model: rank 0's stats are the reference.
                trainer_stats = compute_tensor_stats(valid_weights)
                trainer_stats["rank"] = self.rank
                self.registry.set_stats.remote(step_key, trainer_stats)
            else:
                return
            logger.info(
                f"[RAIDEN PARITY] Successfully stored trainer rank {self.rank} stats for step {step_key}: "
                f"L1={trainer_stats['l1_norm']:.4f}, numel={trainer_stats['total_numel']}"
            )
        except Exception as e:
            logger.warning(f"Failed to record trainer rank {self.rank} stats in TPUWeightRegistry: {e}")

    @torch.no_grad()
    def receive_weights(self, global_steps: Optional[int] = None, **kwargs):
        return None


async def update_raiden_weights(
    manager,
    global_steps: Optional[int] = None,
    verify_parity: bool = False,
) -> dict:
    """Orchestrator coordination for Raiden TPU P2P weight synchronization via central RaidenController."""
    t_abort_start = time.perf_counter()
    if global_steps and global_steps > 0:
        try:
            await manager.abort_replicas()
        except Exception as e:
            logger.warning(f"Failed to abort replicas at step {global_steps}: {e}")
    t_abort = time.perf_counter() - t_abort_start
    if hasattr(manager, "config") and hasattr(manager.config, "engine_kwargs"):
        verify_parity = manager.config.engine_kwargs.get("raiden", {}).get(
            "verify_parity", manager.config.engine_kwargs.get("verify_parity", verify_parity)
        )

    t_total_start = time.perf_counter()

    # 1. Trigger Trainer ranks to register their tensors with central RaidenController and record global shapes
    t_init_trainer_start = time.perf_counter()

    # Blocks untill all Trainer ranks complete their send_weights, but do so asynchronously
    # in a background thread so we don't freeze the orchestrator's event loop.
    actor_refs = manager.actor_wg.update_weights(global_steps=global_steps, mode="raiden")
    if actor_refs is not None:
        await asyncio.to_thread(ray.get, actor_refs)
    t_init_trainer = time.perf_counter() - t_init_trainer_start

    parallelism = 8
    if hasattr(manager, "config") and hasattr(manager.config, "engine_kwargs"):
        parallelism = manager.config.engine_kwargs.get("raiden", {}).get(
            "parallelism", manager.config.engine_kwargs.get("parallelism", 8)
        )

    # 2. Initialize and register Sampler rollout workers with central RaidenController
    t_init_sampler_start = time.perf_counter()
    sampler_init_futures = [
        replica.server_handle.collective_rpc.remote(
            method="init_raiden_sync_on_worker", kwargs={"parallelism": parallelism}
        )
        for replica in manager.replicas
    ]
    await asyncio.gather(*sampler_init_futures)
    t_init_sampler = time.perf_counter() - t_init_sampler_start

    # 3. Explicit Registration Barrier on Central RaidenController
    # TODO(tpu): Move worker registration and barrier verification to a one-time setup step during
    # initialization (e.g. in prepare/build_process_group), since shard registration is persistent on the
    # controller and does not need to be repeated on every weight sync iteration.
    t_barrier_start = time.perf_counter()

    # Calculate total rollout workers across replicas and create their Raiden IDs ('0'..'N-1').
    # Examples:
    #   - 1 replica with TP=8 (our case): len(replicas)=1, r.world_size=8 -> 8 workers ['0'..'7'].
    #   - 2 replicas with TP=4 (DP=2): len(replicas)=2, each world_size=4 -> 8 workers ['0'..'7'].
    num_rollout_workers = 0
    for r in manager.replicas:
        if hasattr(r, "world_size") and r.world_size:
            num_rollout_workers += r.world_size
        elif hasattr(r, "workers") and r.workers:
            num_rollout_workers += len(r.workers)
        else:
            num_rollout_workers += 1
    if num_rollout_workers == 0:
        num_rollout_workers = len(manager.replicas)
    sampler_replica_ids = [str(i) for i in range(num_rollout_workers)]
    trainer_replica_ids = [str(i) for i in range(manager.actor_wg.world_size)]

    from tpu_sync.api.common import RaidenId
    from tpu_sync.rpc.raiden_controller import RaidenMemoryType

    src_units = [RaidenId(job_name="trainer", job_replica_id=r_id, data_name="weights") for r_id in trainer_replica_ids]
    dst_units = [RaidenId(job_name="sampler", job_replica_id=r_id, data_name="weights") for r_id in sampler_replica_ids]

    if not hasattr(manager, "raiden_controller") or manager.raiden_controller is None:
        raise RuntimeError(
            "No raiden_controller found on CheckpointEngineManager! "
            "Embedded RaidenControllerServer must be initialized on the head node before weight sync."
        )

    barrier_timeout = 60.0
    while True:
        with manager.raiden_controller._lock:
            registered = set(manager.raiden_controller._registered_shards.keys())
        src_registered = all(u in registered for u in src_units)
        dst_registered = all(u in registered for u in dst_units)
        if src_registered and dst_registered:
            logger.info(
                f"[RAIDEN CONTROLLER] All {len(src_units)} Trainer and {len(dst_units)} Sampler units "
                "verified and registered."
            )
            break
        if time.perf_counter() - t_barrier_start > barrier_timeout:
            missing_src = [u for u in src_units if u not in registered]
            missing_dst = [u for u in dst_units if u not in registered]
            raise RuntimeError(
                f"Timeout ({barrier_timeout}s) waiting for workers to register with RaidenController! "
                f"Missing Trainer: {missing_src}, Missing Sampler: {missing_dst}"
            )
        await asyncio.sleep(0.1)
    t_barrier = time.perf_counter() - t_barrier_start

    # 4. Trigger coordinated P2P network transfers via central RaidenController
    t_transfer_start = time.perf_counter()
    transfer_future = manager.raiden_controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        dst_mem_type=RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        expected_block_count=0,
        parallelism=parallelism,
        req_id=f"verl_step_{global_steps or 0}",
    )
    await transfer_future.wait()
    t_transfer = time.perf_counter() - t_transfer_start

    # 5. Sampler replicas install received weights to TPU HBM via H2D DMA
    t_install_start = time.perf_counter()
    install_futures = [
        replica.server_handle.collective_rpc.remote(method="install_raiden_weights") for replica in manager.replicas
    ]
    await asyncio.gather(*install_futures)
    t_install = time.perf_counter() - t_install_start

    t_total = time.perf_counter() - t_total_start

    # 5. Parity Verification (Optional, default=False)
    if verify_parity:
        try:
            await _verify_parity_async(manager, global_steps)
        except Exception as e:
            logger.warning(f"Failed to execute parity verification: {e}")

    logger.info(
        f"[RAIDEN TELEMETRY | Orchestrator] Step {global_steps} Completed in {t_total:.4f}s:\n"
        f"  * Sampler Quiesce/Pause  : {t_abort:.4f}s\n"
        f"  * Trainer Raiden Init    : {t_init_trainer:.4f}s\n"
        f"  * Sampler Raiden Init    : {t_init_sampler:.4f}s\n"
        f"  * Raiden Barrier Check   : {t_barrier:.4f}s\n"
        f"  * RaidenController P2P   : {t_transfer:.4f}s\n"
        f"  * Sampler H2D DMA        : {t_install:.4f}s\n"
        f"  * Total End-to-End Sync  : {t_total:.4f}s"
    )

    # 6. Resume generation immediately
    await manager.resume_generation_replicas()

    return {}


async def _verify_parity_async(manager, global_steps: Optional[int] = None) -> None:
    """Compare distributed norms between Trainer Rank 0 and Sampler TP workers.

    TODO(tpu): Refactor parity verification to use rank-local scalar partitioning.
    Instead of collecting per-tensor dictionaries across all Sampler workers and checking
    replicated vs sharded heuristics over 200+ tensors, each worker can reduce its model
    locally into scalar metrics (numel, l1, l2_sq) before RPC return:
      - Rank 0 accumulates both sharded tensors and replicated 1D tensors (e.g. RMSNorms).
      - Ranks 1..N-1 accumulate only sharded tensors.
    The orchestrator can then perform verification in O(ranks) pure scalar arithmetic
    rather than O(tensors * ranks) loop aggregation.
    """
    step_key = global_steps if global_steps is not None else 0
    if step_key <= 0:
        return

    try:
        registry = ray.get_actor("TPUWeightRegistry", namespace="verl")
        num_trainer_ranks = manager.actor_wg.world_size
        trainer_entry = None
        for _ in range(25):
            entry = await registry.get_stats.remote(step_key)
            # Sharded export: every trainer rank posts partial stats, usable once all have arrived.
            if entry is not None and ("ranks" not in entry or len(entry["ranks"]) >= num_trainer_ranks):
                trainer_entry = entry
                break
            await asyncio.sleep(0.5)

        sampler_futures = [
            replica.server_handle.collective_rpc.remote(
                method="get_model_weights_stats", kwargs={"include_shards": False}
            )
            for replica in manager.replicas
        ]
        sampler_entries = await asyncio.gather(*sampler_futures)

        if not trainer_entry or not sampler_entries:
            logger.warning(f"[RAIDEN PARITY] Incomplete stats data for step {step_key}")
            return

        # Unpacks and flattens the results collected from all Sampler rollout replicas into a single flat list
        # of worker dictionary objects.
        sampler_workers = [
            w
            for res in sampler_entries
            for w in (res if isinstance(res, list | tuple) else [res])
            if isinstance(w, dict)
        ]
        if not sampler_workers:
            return

        if "ranks" in trainer_entry:
            trainer_master = merge_rank_stats(trainer_entry["ranks"])
        else:
            trainer_master = trainer_entry.get("master", trainer_entry)
        trainer_per_tensor = trainer_master.get("per_tensor", {})
        # Gets master list of all model tensor names (e.g., "model.layers.0.self_attn.qkv_proj.weight").
        all_param_names = list(sampler_workers[0].get("per_tensor", {}).keys()) if sampler_workers else []

        total_trainer_numel = trainer_master.get("total_numel", 0)
        total_trainer_l1 = trainer_master.get("l1_norm", 0.0)
        global_trainer_l2 = trainer_master.get("l2_norm", 0.0)

        total_sampler_l1, total_sampler_l2_sq, total_sampler_numel = 0.0, 0.0, 0
        mismatches = []

        for name in all_param_names:
            s_numels = [w.get("per_tensor", {}).get(name, {}).get("numel", 0) for w in sampler_workers]
            s_l1s = [w.get("per_tensor", {}).get(name, {}).get("l1", 0.0) for w in sampler_workers]
            s_l2_sqs = [
                w.get("per_tensor", {})
                .get(name, {})
                .get("l2_sq", w.get("per_tensor", {}).get(name, {}).get("l2", 0.0) ** 2)
                for w in sampler_workers
            ]

            is_replicated = (
                len(set(s_numels)) == 1
                and (name.endswith("layernorm.weight") or "norm" in name)
                and len(s_numels[0:1]) > 0
                and s_numels[0] < 10000
            )
            if is_replicated:
                s_agg_numel = s_numels[0]
                s_agg_l1 = s_l1s[0]
                s_agg_l2 = s_l2_sqs[0] ** 0.5
            else:
                s_agg_numel = sum(s_numels)
                s_agg_l1 = sum(s_l1s)
                s_agg_l2 = sum(s_l2_sqs) ** 0.5

            t_data = trainer_per_tensor.get(name, {})
            t_agg_numel = t_data.get("numel", 0)
            t_agg_l1 = t_data.get("l1", 0.0)

            total_sampler_numel += s_agg_numel
            total_sampler_l1 += s_agg_l1
            total_sampler_l2_sq += s_agg_l2**2

            delta_numel = abs(s_agg_numel - t_agg_numel)
            delta_l1 = abs(s_agg_l1 - t_agg_l1)
            rel_tol = 1e-3 * max(abs(t_agg_l1), 1.0)

            if delta_numel != 0 or delta_l1 > rel_tol:
                mismatches.append(f"  * {name}: Trainer(L1={t_agg_l1:.4f}) vs Sampler(L1={s_agg_l1:.4f})")

        global_sampler_l2 = total_sampler_l2_sq**0.5
        total_l1_delta = abs(total_sampler_l1 - total_trainer_l1)
        total_l2_delta = abs(global_sampler_l2 - global_trainer_l2)
        total_numel_delta = abs(total_sampler_numel - total_trainer_numel)
        total_rel_tol = 1e-3 * max(abs(total_trainer_l1), 1.0)

        if not mismatches and total_numel_delta == 0 and total_l1_delta <= total_rel_tol:
            logger.info(
                f"[RAIDEN PARITY VERIFIED | Step {step_key}] 100% DISTRIBUTED NORM PARITY CONFIRMED!\n"
                f"  * Global L1 Norm: {total_sampler_l1:.6f} "
                f"(Trainer={total_trainer_l1:.6f}, delta={total_l1_delta:.6f})\n"
                f"  * Global L2 Norm: {global_sampler_l2:.6f} "
                f"(Trainer={global_trainer_l2:.6f}, delta={total_l2_delta:.6f})\n"
                f"  * Total Parameters: {total_sampler_numel} across {len(all_param_names)} tensors"
            )
        else:
            mismatches_summary = "\n".join(mismatches[:10])
            logger.error(
                f"[RAIDEN PARITY MISMATCH | Step {step_key}] Norms do NOT match!\n"
                f"  * Trainer: numel={total_trainer_numel}, L1={total_trainer_l1:.6f}, L2={global_trainer_l2:.6f}\n"
                f"  * Sampler: numel={total_sampler_numel}, L1={total_sampler_l1:.6f}, L2={global_sampler_l2:.6f}\n"
                f"  * Mismatched Tensors ({len(mismatches)} / {len(all_param_names)}):\n"
                f"{mismatches_summary}"
            )
    except Exception as e:
        logger.warning(f"Error during parity verification for step {step_key}: {e}")
