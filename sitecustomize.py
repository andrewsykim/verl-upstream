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
"""Workspace sitecustomize.py for veRL on TPU.

Automatically confines vLLM subprocesses (e.g. EngineCoreProc spawned by multiprocessing)
to the allocated TPU rollout slice when VERL_TPU_ROLLOUT_SLICE_IPS is present in the environment.
"""

import logging
import os

_slice_ips = os.environ.get("VERL_TPU_ROLLOUT_SLICE_IPS")
if _slice_ips:
    allowed_ips = {ip.strip() for ip in _slice_ips.split(",") if ip.strip()}
    if "TORCH_TPU_DP_SIZE" not in os.environ and len(allowed_ips) > 1:
        os.environ["TORCH_TPU_DP_SIZE"] = str(len(allowed_ips))
    try:
        from verl.workers.rollout.vllm_rollout.vllm_tpu_async_server import confine_vllm_to_slice

        confine_vllm_to_slice(allowed_ips)
    except Exception as exc:
        logging.getLogger(__name__).warning("sitecustomize: failed to confine vLLM to slice: %s", exc)

try:
    from verl.workers.rollout.vllm_rollout.utils import patch_vllm_tpu_multihost_dp, patch_vllm_tpu_rpa_vmem

    patch_vllm_tpu_multihost_dp()
    patch_vllm_tpu_rpa_vmem()
except Exception:
    pass

