# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""TPU (TorchTPU) environment bootstrap helpers.

TorchTPU deliberately keeps its C++ runtime a "dumb consumer" of the environment:
the launcher is responsible for computing the whole distributed topology and
handing it over through environment variables. ``torchrun`` users get this from a
launcher wrapper; under Ray, verl has to do it itself, which is what this module
provides.

The hard contract enforced by TorchTPU's slice-builder discovery is::

    RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
    TORCH_TPU_SLICEBUILDER_ADDRESSES TORCH_TPU_TOPOLOGY

``TORCH_TPU_SLICEBUILDER_ADDRESSES`` is a comma-separated ``host:port`` list with
**one entry per rank, indexed by RANK**; ``TORCH_TPU_TOPOLOGY`` is the device mesh
of the slice. Everything else below is derived from the variables GKE/KubeRay
already injects into TPU pods (``TPU_WORKER_HOSTNAMES``, ``TPU_HOST_BOUNDS``,
``TPU_CHIPS_PER_HOST_BOUNDS``, ``TPU_ACCELERATOR_TYPE``).

Device-vs-chip accounting: on v7x (Ironwood) each chip exposes **two** PJRT
devices (chiplets), and TorchTPU runs one process per device. A ``tpu7x`` host
with a ``2,2,1`` chip bound therefore hosts 8 ranks and reports a 4-dimensional
topology ``2,2,1,2``.
"""

import logging
import os

logger = logging.getLogger(__name__)

# First port of the contiguous range each rank's slice-builder listens on. Matches
# the range GKE reserves for TorchTPU workers in the Ray TPU images.
TRAINER_SLICEBUILDER_BASE_PORT = 8471
# Rollout engines run in a separate process group on a separate slice; keep their
# slice-builder ports disjoint from the trainer's so a colocated debug run does not
# collide.
ROLLOUT_SLICEBUILDER_BASE_PORT = 8070

# Marks that this process' TorchTPU distributed contract has been derived. Kept in the
# environment, not a module global, so it survives module re-imports in a worker.
_BOOTSTRAPPED_FLAG = "VERL_TPU_DIST_ENV_READY"


def is_v7x(environ: dict | None = None) -> bool:
    """Return True when the local TPU is a v7x (Ironwood) chip.

    v7x is the first generation where one chip exposes two independent PJRT
    devices, which changes both the rank accounting and the topology arity.
    """
    environ = os.environ if environ is None else environ
    return environ.get("TPU_ACCELERATOR_TYPE", "").startswith("tpu7x") or environ.get("VERL_TPU_GENERATION", "") == "v7x"


def devices_per_chip(environ: dict | None = None) -> int:
    """Number of PJRT devices (chiplets) exposed by one physical TPU chip."""
    return 2 if is_v7x(environ) else 1


def _slice_topology(world_size: int, num_hosts: int, environ: dict | None = None) -> str | None:
    """Topology string describing the whole slice, as TorchTPU expects it.

    On v7x this is a 4-tuple ``x,y,z,chiplets_per_chip`` (older generations use a
    3-tuple). torch_tpu ships the authoritative table, keyed off a PCI scan of the
    local host, so prefer it for a single-host slice and only fall back to the
    pre-existing ``TORCH_TPU_TOPOLOGY`` otherwise.
    """
    environ = os.environ if environ is None else environ
    inherited = environ.get("TORCH_TPU_TOPOLOGY", "")
    if inherited:
        return inherited

    if num_hosts == 1:
        try:
            from torch_tpu._internal.utils import hardware  # noqa: PLC0415

            topology = hardware.get_tpu_topology(world_size)
            if topology:
                return topology
        except Exception:
            logger.warning("torch_tpu could not report a topology for %d devices", world_size, exc_info=True)

    if is_v7x(environ):
        if world_size == 32:
            return "2,2,4,2"
        elif world_size == 8:
            return "2,2,1,2"

    return None


def select_tpu_slice_nodes(required_devices_per_node: list[int], device_name: str = "TPU") -> list[dict]:
    """Find a free TPU slice that can accommodate the required devices per node.

    Args:
        required_devices_per_node: List specifying the required TPU devices for each node in the slice.
        device_name: Resource name for TPU devices in Ray (default "TPU").

    Returns:
        List of node dicts from `ray.nodes()` in the selected slice, ordered by `ray.io/tpu-worker-id`.
    """
    import ray

    total_required = sum(required_devices_per_node)
    num_nodes_required = len(required_devices_per_node)
    available = ray._private.state.available_resources_per_node()

    slices: dict[str, list[dict]] = {}
    for node in ray.nodes():
        if not node.get("Alive") or device_name not in node.get("Resources", {}):
            continue
        labels = node.get("Labels") or node.get("labels") or {}
        slice_name = labels.get("ray.io/tpu-slice-name", node["NodeID"])
        slices.setdefault(slice_name, []).append(node)

    def _sort_key(node: dict) -> int:
        labels = node.get("Labels") or node.get("labels") or {}
        return int(labels.get("ray.io/tpu-worker-id", 0))

    for name in sorted(slices):
        nodes = sorted(slices[name], key=_sort_key)
        if len(nodes) < num_nodes_required:
            continue
        if sum(int(n["Resources"][device_name]) for n in nodes) < total_required:
            continue
        is_free = True
        for idx, req in enumerate(required_devices_per_node):
            node_id = nodes[idx]["NodeID"]
            avail = available.get(node_id, {}).get(device_name, 0)
            if avail < req:
                is_free = False
                break
        if not is_free:
            continue
        logger.info("Selected TPU slice %s (%d nodes) for required %s", name, len(nodes), required_devices_per_node)
        return nodes[:num_nodes_required]

    raise RuntimeError(
        f"No free TPU slice with at least {total_required} {device_name} devices across {num_nodes_required} nodes. "
        f"Slices seen: " + ", ".join(f"{name}({len(nodes)} nodes)" for name, nodes in sorted(slices.items()))
    )


def maybe_init_tpu_distributed_env(
    environ: dict | None = None,
    base_port: int = TRAINER_SLICEBUILDER_BASE_PORT,
) -> bool:
    """Populate the ``TORCH_TPU_*`` distributed contract for this process.

    Must be called **before** the TPU runtime is initialized (i.e. before the first
    TPU tensor or ``init_process_group``).

    Returns:
        True if the environment now describes a multi-rank TPU slice, False if this
        is a single-process run or the pod does not expose enough information (in
        which case TorchTPU's own single-host defaults apply).
    """
    environ = os.environ if environ is None else environ

    if environ.get(_BOOTSTRAPPED_FLAG):
        # Already derived for this process. Re-deriving would be wrong, not merely
        # redundant: the bounds written below describe one device per process, so a
        # second pass would read them back as the shape of the whole slice.
        return True

    world_size = int(environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False

    rank = int(environ.get("RANK", "0"))

    hostnames = [h for h in environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    if not hostnames:
        logger.warning("TPU_WORKER_HOSTNAMES is not set; cannot derive the TorchTPU slice topology.")
        return False

    # NOTE: deliberately not derived from TPU_CHIPS_PER_HOST_BOUNDS / TPU_HOST_BOUNDS.
    # torch_tpu's runtime overwrites both with its own one-device-per-process layout
    # ("1,1,1,1" and the slice topology) the moment it initializes, which in a vLLM or
    # torchtitan worker can happen before verl gets here. TPU_WORKER_HOSTNAMES is
    # GKE-owned and is not rewritten.
    num_hosts = len(hostnames)
    if world_size % num_hosts != 0:
        raise ValueError(
            f"TPU world_size={world_size} is not divisible by the {num_hosts} host(s) in the slice "
            f"({', '.join(hostnames)}). Set trainer.nnodes to the number of hosts in the slice and "
            "trainer.n_gpus_per_node to the devices per host."
        )
    devices_per_host = world_size // num_hosts

    topology = _slice_topology(world_size, num_hosts, environ)
    if topology is None:
        logger.warning("Could not determine the TPU slice topology; skipping TorchTPU env bootstrap.")
        return False

    # Derived from RANK rather than read from LOCAL_RANK: verl only populates LOCAL_RANK
    # when RAY_EXPERIMENTAL_NOSET_<visible devices> is set *inside the actor*, and an
    # unset LOCAL_RANK silently reads back as 0 for every rank -- which makes every
    # process on the host open the same chip:
    #   Failed to acquire a TPU device node '/dev/vfio/5' because it is already opened
    # This matches how RayWorkerGroup assigns placement-group bundles
    # (local_rank = rank % local_world_size), so it holds for multi-host slices too.
    local_rank = rank % devices_per_host

    ports = [base_port + i for i in range(devices_per_host)]
    addresses = [f"{host}:{port}" for host in hostnames for port in ports]

    environ["LOCAL_RANK"] = str(local_rank)
    environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(addresses)
    environ["TORCH_TPU_TOPOLOGY"] = topology
    # JAX/libtpu-facing mirrors of the same layout, used by the PJRT client.
    environ["TPU_PROCESS_ADDRESSES"] = ",".join(addresses)
    environ["TPU_PROCESS_PORT"] = str(ports[local_rank])
    # One device per process, expressed with the same arity as the topology.
    environ["TPU_CHIPS_PER_HOST_BOUNDS"] = ",".join(["1"] * len(topology.split(",")))
    environ["TPU_HOST_BOUNDS"] = topology
    # TPU_VISIBLE_DEVICES is TorchTPU's source of truth for device selection; it
    # overwrites TPU_VISIBLE_CHIPS to match (libtpu prioritizes the latter), so both are
    # set here to keep the two consistent for any non-TorchTPU consumer in the process.
    environ["TPU_VISIBLE_DEVICES"] = str(local_rank)
    environ["TPU_VISIBLE_CHIPS"] = str(local_rank)
    environ["CLOUD_TPU_TASK_ID"] = str(rank)
    environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

    environ[_BOOTSTRAPPED_FLAG] = "1"

    logger.info(
        "TorchTPU distributed env: rank=%s local_rank=%s world_size=%s topology=%s slicebuilder=%s",
        rank,
        local_rank,
        world_size,
        topology,
        environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"],
    )
    return True


def rollout_tpu_env_vars() -> dict[str, str]:
    """Environment overrides for TPU rollout (vLLM) engine processes.

    The PJRT runtime preallocates most of the HBM of every device it opens. A
    rollout engine that shares a host with anything else therefore has to be told
    to allocate on demand, and the enhanced launch barrier has to be disabled so
    that several independent TPU runtimes can start on one host without deadlocking
    each other.
    """
    return {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "LIBTPU_INIT_ARGS": "--xla_tpu_use_enhanced_launch_barrier=false",
        # RL rollouts see a wide spread of sequence lengths; the default Dynamo
        # recompile limit trips long before the shapes stabilise.
        "TORCH_DYNAMO_RECOMPILE_LIMIT": "100",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "SKIP_JAX_PRECOMPILE": "1",
        "TPU_SKIP_MDS_QUERY": "true",
        "TPU_ACCELERATOR_TYPE": "tpu7x",
    }
