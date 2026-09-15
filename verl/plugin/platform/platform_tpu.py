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
"""Google Cloud TPU platform implementation (TorchTPU backend).

This platform targets the native PyTorch TPU backend shipped as the ``torch_tpu``
package (device type ``"tpu"``, c10d backend ``"tpu_dist"``), **not** ``torch_xla``.

Enable it explicitly with ``VERL_PLATFORM=tpu``; auto-detection also picks it up when
``torch_tpu`` is importable and TPU chips are visible to the process.

Notes on the TPU execution model that shape this file:

* TPU devices are **not** indexed from Python. Each process drives exactly one TPU
  device, selected before runtime initialization through ``TPU_VISIBLE_CHIPS``, and
  tensors are placed with a plain ``torch.device("tpu")`` (no ``tpu:0``). ``set_device``
  is therefore a no-op and ``current_device`` reports the process' local rank.
* Memory is managed by the PJRT runtime, not by a caching allocator, so
  ``empty_cache`` / allocator settings are no-ops and the memory statistics that
  ``verl.utils.memory_utils`` prints are best-effort zeros.
* The runtime is asynchronous: ``synchronize`` drains pending TPU work via
  ``torch_tpu._internal.sync``.
"""

import logging
import os
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

import torch

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)

# c10d backend registered by torch_tpu for the "tpu" device type.
TPU_DISTRIBUTED_BACKEND = "tpu_dist"

# Ray reports one ``TPU`` resource unit per addressable TPU device (note that an
# Ironwood/v7x chip exposes two devices), and verl runs one worker per device.
TPU_RAY_RESOURCE = "TPU"

_torch_tpu = None


def _ensure_torch_tpu():
    """Import ``torch_tpu`` once and return the module (or ``None``).

    Importing the package registers the ``tpu`` device (PrivateUse1), the ``tpu_dist``
    distributed backend and configures ``libtpu``.
    """
    global _torch_tpu
    if _torch_tpu is not None:
        return _torch_tpu
    try:
        import torch_tpu  # noqa: PLC0415

        _torch_tpu = torch_tpu
    except Exception as e:  # noqa: BLE001 - optional dependency
        logger.debug("torch_tpu is not importable: %s", e)
    return _torch_tpu


_ensure_torch_tpu()


def _patch_device_module(module: ModuleType) -> ModuleType:
    """Fill in the ``torch.cuda``-shaped API that ``torch.tpu`` does not provide.

    verl's device-agnostic helpers (memory reporting, profiling, seeding) call a
    handful of ``torch.cuda`` functions through ``get_torch_device()``. TPU has no
    caching allocator and no device index, so the missing entries are no-ops or
    zeros rather than errors.
    """

    def _noop(*args, **kwargs):
        return None

    def _zero(*args, **kwargs):
        return 0

    def _total_memory(module):
        """HBM visible to this device, in bytes."""
        try:
            return int(module.get_device_properties().total_memory)
        except Exception:  # noqa: BLE001 - runtime may not be up yet
            # v7x (Ironwood): 192GiB per chip, shared by the chip's two devices.
            return 96 * (1024**3)

    def _mem_get_info(*args, _mod=None, **kwargs):
        """``(free, total)`` in bytes, mirroring ``torch.cuda.mem_get_info``.

        TPU exposes usage only through a human-readable summary, so report the whole
        device as free rather than guessing. Callers use this for logging only.
        """
        total = _total_memory(_mod)
        return total, total

    defaults = {
        "empty_cache": _noop,
        "reset_peak_memory_stats": _noop,
        "reset_max_memory_allocated": _noop,
        "memory_allocated": _zero,
        "max_memory_allocated": _zero,
        "memory_reserved": _zero,
        "max_memory_reserved": _zero,
        "memory_stats": lambda *a, **k: {},
        "memory_summary": lambda *a, **k: "",
        "mem_get_info": lambda *a, _m=module, **k: _mem_get_info(*a, _mod=_m, **k),
        # TPU memory is owned by the PJRT runtime; there is no per-process budget knob
        # and no CUDA-IPC-style shared-handle collection.
        "set_per_process_memory_fraction": _noop,
        "ipc_collect": _noop,
        "synchronize": _noop,
        "manual_seed": _noop,
        "manual_seed_all": _noop,
        "set_device": _noop,
        "get_rng_state": lambda *a, **k: torch.empty(0, dtype=torch.uint8),
        "set_rng_state": _noop,
        "is_available": lambda: True,
        "device_count": lambda: int(os.environ.get("WORLD_SIZE", "1")),
        "current_device": lambda: int(os.environ.get("LOCAL_RANK", "0")),
        "get_device_name": lambda *a, **k: os.environ.get("TPU_ACCELERATOR_TYPE", "tpu"),
        # v7x (Ironwood) exposes 192GiB of HBM per chip shared by its two devices.
        "get_device_properties": lambda *a, **k: SimpleNamespace(total_memory=96 * (1024**3)),
    }
    for name, fn in defaults.items():
        # `hasattr` drives torch_tpu's lazy attribute resolution, so this only fills in
        # names the backend genuinely does not implement.
        if not hasattr(module, name):
            setattr(module, name, fn)

    # `set_device` needs replacing rather than filling in. A TPU process is bound to a
    # single device by the runtime before Python starts, and torch_tpu *raises* if asked
    # to switch to any other index. verl calls `set_device(local_rank)` from generic code
    # paths where local_rank is the rank within the node, not within the process, so
    # downgrade the mismatch to a debug log.
    if not getattr(module, "_verl_set_device_patched", False):
        _real_set_device = getattr(module, "set_device", None)

        def _tolerant_set_device(device_index, _real=_real_set_device, _mod=module):
            try:
                current = int(_mod.current_device())
            except Exception:  # noqa: BLE001 - runtime may not be up yet
                current = None
            if _real is not None and (current is None or int(device_index) == current):
                return _real(device_index)
            logger.debug(
                "Ignoring set_device(%s): this process is bound to TPU device %s and TPU "
                "does not support switching devices within a process.",
                device_index,
                current,
            )
            return None

        module.set_device = _tolerant_set_device
        module._verl_set_device_patched = True

    return module


class _UnavailableTpuModule:
    """Fallback device module when TPU hardware/backend is unavailable on the current host."""

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return 0

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def synchronize(*args, **kwargs) -> None:
        pass

    @staticmethod
    def empty_cache() -> None:
        pass


def _resolve_device_module() -> ModuleType:
    if _ensure_torch_tpu() is None:
        return _UnavailableTpuModule
    module = getattr(torch, "tpu", None)
    if module is None:
        try:
            from torch._utils import _get_device_module  # noqa: PLC0415

            module = _get_device_module("tpu")
        except Exception:
            return _UnavailableTpuModule
    return _patch_device_module(module)


@PlatformRegistry.register(platform="tpu")
class PlatformTPU(PlatformBase):
    """Platform backend for Google Cloud TPU via TorchTPU."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "tpu"

    @property
    def vendor_name(self) -> str:
        return "google"

    @property
    def device_module(self) -> ModuleType:
        return _resolve_device_module()

    def is_available(self) -> bool:
        if _ensure_torch_tpu() is None:
            return False
        try:
            module = getattr(torch, "tpu", None)
            if module is not None and hasattr(module, "is_available"):
                return bool(module.is_available())
            return False
        except Exception as e:  # noqa: BLE001
            logger.debug("TPU availability check failed: %s", e)
            return False

    def is_platform_available(self, use_smi_check=False) -> bool:
        """Detect a TPU environment.

        ``use_smi_check`` relaxes the check to "the TPU software stack is installed and
        this pod was given TPU chips", so that CPU-only Ray actors (the driver, rollout
        proxies) on a TPU cluster still resolve to this platform.
        """
        if _ensure_torch_tpu() is None:
            return False
        if use_smi_check:
            return any(key in os.environ for key in ("TPU_ACCELERATOR_TYPE", "TPU_WORKER_HOSTNAMES", "TPU_SKIP_MDS_QUERY"))
        return self.is_available()

    def current_device(self) -> int:
        """Return this process' TPU device index.

        TPU tensors carry no device index; verl only uses this value for logging and
        for ``ray``-assigned local-rank bookkeeping.
        """
        return int(os.environ.get("LOCAL_RANK", "0"))

    def device_count(self) -> int:
        module = getattr(torch, "tpu", None)
        if module is not None and hasattr(module, "device_count"):
            try:
                return int(module.device_count())
            except Exception as e:  # noqa: BLE001
                logger.debug("torch.tpu.device_count() failed: %s", e)
        if not self.is_available():
            return 0
        return int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))

    def set_device(self, device_index: int) -> None:
        """No-op: the TPU device is bound by ``TPU_VISIBLE_CHIPS`` before runtime init."""
        return None

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch_tpu = _ensure_torch_tpu()
        if torch_tpu is None:
            return
        try:
            torch_tpu._internal.sync.synchronize(wait=True)
        except Exception as e:  # noqa: BLE001
            logger.debug("TPU synchronize failed: %s", e)

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        """No-op: TPU memory is owned by the PJRT runtime, not a caching allocator."""
        return None

    def empty_cache(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    def torch_device(self, index: Optional[int] = None) -> torch.device:
        # TPU tensors carry no device index: each process owns exactly one device and
        # `torch.device("tpu", 0)` is rejected by the runtime. A caller-supplied index
        # is therefore meaningless here and deliberately ignored.
        return torch.device("tpu")

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        """Return the c10d backend torch_tpu registered for the ``tpu`` device.

        torch_tpu registers ``tpu_dist`` (and aliases it as the default backend for
        the ``tpu`` device) only once more than one local TPU device is visible, so
        resolve it dynamically rather than hardcoding a name that may not exist.
        """
        try:
            import torch.distributed as dist  # noqa: PLC0415

            registered = {name.lower() for name in getattr(dist.Backend, "backend_list", [])}
            if TPU_DISTRIBUTED_BACKEND in registered:
                return TPU_DISTRIBUTED_BACKEND
            mapped = dist.Backend.default_device_backend_map.get("tpu")
            if mapped:
                return mapped
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not resolve the TPU distributed backend dynamically: %s", e)
        return TPU_DISTRIBUTED_BACKEND

    def visible_devices_envvar(self) -> str:
        return "TPU_VISIBLE_CHIPS"

    # ------------------------------------------------------------------
    # Distributed bootstrap
    # ------------------------------------------------------------------

    def prepare_distributed_env(self) -> None:
        """Derive the TorchTPU slice topology and bind this process to its device."""
        from verl.utils.tpu_utils import maybe_init_tpu_distributed_env  # noqa: PLC0415

        maybe_init_tpu_distributed_env()
        # Touching the device forces torch_tpu to initialize the PJRT client and to
        # register the `tpu_dist` process-group backend, which must happen before
        # `init_process_group` is called.
        if _ensure_torch_tpu() is not None:
            self._parse_absl_flags()
            torch.empty(1, device="tpu")

    @staticmethod
    def _parse_absl_flags() -> None:
        """Mark torch_tpu's absl flags parsed.

        torch_tpu's compiler passes are configured with absl flags. A Ray worker
        never goes through an absl main, so the flag registry stays unparsed and
        the first TPU compilation raises ``UnparsedFlagAccessError``.
        """
        try:
            from torch_tpu._internal.distributed.multiprocessing import (  # noqa: PLC0415
                parse_absl_flags,
            )

            parse_absl_flags()
        except Exception:  # pragma: no cover - best effort, torch_tpu may move it
            logger.warning("Could not parse torch_tpu absl flags", exc_info=True)

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        return TPU_RAY_RESOURCE

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        # One worker owns one whole TPU device: TPU chips cannot be time-shared
        # between processes the way CUDA devices can, so never request a fraction.
        return {"resources": {TPU_RAY_RESOURCE: 1}}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS"]

    def rollout_env_vars(self) -> dict[str, str]:
        from verl.utils.tpu_utils import rollout_tpu_env_vars  # noqa: PLC0415

        return rollout_tpu_env_vars()

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        # There is no cross-process TPU buffer sharing analogous to CUDA IPC; weight
        # transfer goes through host memory (see verl.utils.tpu_utils).
        return False

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        logger.debug("NVTX range (no-op on TPU): %s", msg)
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return None
