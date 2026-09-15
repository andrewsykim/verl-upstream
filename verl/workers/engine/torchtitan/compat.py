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
"""Compatibility helpers for the torchtitan versions verl runs against.

torchtitan is vendored rather than released on a stable cadence, and the TPU
distributions (which ship a torchtitan fork alongside ``torch_tpu``) lag the
upstream API that ``transformer_impl.py`` was written against. This module
isolates the handful of call sites that differ so the engine can support both
without branching on a version string.

Every helper degrades to the *upstream* behaviour when the newer API is present,
so CUDA runs are unaffected.
"""

import dataclasses
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _field_names(cls) -> set[str]:
    """Return the dataclass field names of ``cls`` (empty set if not a dataclass)."""
    if not dataclasses.is_dataclass(cls):
        return set()
    return {f.name for f in dataclasses.fields(cls)}


def build_parallelism_config(parallelism_cls, *, spmd_backend: Optional[str] = None, **kwargs):
    """Construct a ``ParallelismConfig``, dropping ``spmd_backend`` when unsupported.

    ``spmd_backend`` selects torchtitan's DTensor/SPMD lowering and only exists on
    builds that carry the SPMD-types work. Forks without it (notably the TorchTPU
    distribution, which always uses plain DTensor) would raise ``TypeError``.
    """
    supported = _field_names(parallelism_cls)
    if spmd_backend is not None:
        if "spmd_backend" in supported:
            kwargs["spmd_backend"] = spmd_backend
        else:
            logger.warning(
                "This torchtitan build has no ParallelismConfig.spmd_backend; ignoring "
                "spmd_backend=%r and using the build's default DTensor lowering.",
                spmd_backend,
            )
    unknown = set(kwargs) - supported if supported else set()
    for name in sorted(unknown):
        logger.warning("Dropping unsupported ParallelismConfig field %r.", name)
        kwargs.pop(name)
    return parallelism_cls(**kwargs)


def build_optimizer_config(optimizers_container_cls, param_group_config_cls, *, name: str, optimizer_kwargs: dict):
    """Build ``OptimizersContainer.Config`` across the two ``ParamGroupConfig`` shapes.

    Upstream torchtitan describes a param group by the optimizer to build for it
    (``optimizer_name`` + ``optimizer_kwargs``); older/forked builds instead keep a
    single optimizer on the container and let a param group apply *multipliers*
    (``lr_multiplier``, ``weight_decay_multiplier``, ...). Since verl only ever uses
    one catch-all group, the latter reduces to setting the hyper-parameters directly
    on the container.
    """
    config_cls = optimizers_container_cls.Config
    group_fields = _field_names(param_group_config_cls)

    if {"optimizer_name", "optimizer_kwargs"} <= group_fields:
        return config_cls(
            param_groups=[
                param_group_config_cls(
                    pattern=r".*",
                    optimizer_name=name,
                    optimizer_kwargs=dict(optimizer_kwargs),
                )
            ],
        )

    # Flat-container form: hyper-parameters live on the container itself.
    container_fields = _field_names(config_cls)
    betas = optimizer_kwargs.get("betas")
    flat: dict[str, Any] = {
        "name": name,
        "lr": optimizer_kwargs.get("lr"),
        "eps": optimizer_kwargs.get("eps"),
        "weight_decay": optimizer_kwargs.get("weight_decay"),
    }
    if betas is not None:
        flat["beta1"], flat["beta2"] = betas[0], betas[1]
    if "implementation" in container_fields:
        # TPU has no fused/`foreach`-with-capturable AdamW; "foreach" is the only
        # implementation with kernels for every supported accelerator.
        flat["implementation"] = "foreach"
    flat = {k: v for k, v in flat.items() if v is not None and k in container_fields}
    return config_cls(**flat)


def build_lr_scheduler_config(lr_schedulers_container_cls, **kwargs):
    """Build ``LRSchedulersContainer.Config``, dropping fields this build lacks."""
    config_cls = lr_schedulers_container_cls.Config
    supported = _field_names(config_cls)
    if supported:
        dropped = sorted(set(kwargs) - supported)
        for name in dropped:
            logger.warning("Dropping unsupported LRSchedulersContainer.Config field %r.", name)
            kwargs.pop(name)
    return config_cls(**kwargs)


def build_activation_checkpoint_config(mode: str):
    """Return the activation-checkpoint config object for ``mode``.

    Two torchtitan shapes exist:

    * one config class per mode
      (``torchtitan.distributed.activation_checkpoint.{FullAC,SelectiveAC}``), where
      "disabled" is expressed as ``None``; and
    * a single ``ActivationCheckpointConfig(mode=...)`` that includes ``mode="none"``
      and whose ``.mode`` is read unconditionally by ``parallelize_*`` -- passing
      ``None`` there raises ``AttributeError``.
    """
    try:
        from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC  # noqa: PLC0415
    except ImportError:
        from torchtitan.config import ActivationCheckpointConfig  # noqa: PLC0415

        return ActivationCheckpointConfig(mode=mode)

    if mode == "none":
        return None
    return {"selective": SelectiveAC.Config, "full": FullAC.Config}[mode]()


def model_spec_supports_attn_backend(model_registry, flavor: str, attn_backend: str) -> bool:
    """Whether ``model_registry`` accepts ``attn_backend`` for ``flavor``."""
    try:
        model_registry(flavor, attn_backend=attn_backend)
    except Exception:  # noqa: BLE001 - probing an unknown third-party signature
        return False
    return True
