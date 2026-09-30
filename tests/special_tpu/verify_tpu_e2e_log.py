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
"""Verify the Ray job log of a TPU CI test (smoke: TPU platform / sft: trainer / grpo: RL).

Usage:
    python3 tests/special_tpu/verify_tpu_e2e_log.py <smoke|sft|grpo> <log_file> <smoke_test: 0|1>
"""

import math
import re
import sys


def _floats(pattern: str, text: str) -> list[float]:
    return [float(x) for x in re.findall(pattern, text)]


def verify_smoke(text: str, smoke_test: bool) -> None:
    assert "[TPU smoke] PASSED" in text, "[TPU CI] TPU smoke test did not report PASSED."
    assert '"device": "tpu:0"' in text, "[TPU CI] TPU smoke test did not run on a tpu device."
    assert "[TPU smoke] TPU vs CPU max |diff|" in text, "[TPU CI] TPU smoke test did not run the TPU-vs-CPU check."
    print("[TPU CI] TPU smoke test verified.")


def verify_sft(text: str, smoke_test: bool) -> None:
    losses = _floats(r"train/loss[:\s]+([0-9.eE+-]+)", text)
    val_losses = _floats(r"val/loss[:\s]+([0-9.eE+-]+)", text)
    grad_norms = _floats(r"train/grad_norm[:\s]+([0-9.eE+-]+)", text)
    if not losses or not val_losses:
        sys.exit("[TPU CI] ERROR: Missing train/loss or val/loss metrics in SFT output.")
    if any(not math.isfinite(g) for g in grad_norms):
        sys.exit("[TPU CI] ERROR: Non-finite train/grad_norm in SFT output.")
    assert losses[-1] < losses[0] * 0.80, (
        f"[TPU CI] SFT train/loss did not converge sufficiently: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    assert val_losses[-1] <= val_losses[0] and val_losses[-1] < 0.85, (
        f"[TPU CI] SFT val/loss did not converge sufficiently: {val_losses[0]:.4f} -> {val_losses[-1]:.4f}"
    )
    print(
        f"[TPU CI] SFT convergence verified: train/loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
        f"val/loss {val_losses[0]:.4f} -> {val_losses[-1]:.4f}"
    )


def verify_grpo(text: str, smoke_test: bool) -> None:
    if re.search(r"actor/grad_norm[:\s]+(nan|inf)", text, flags=re.IGNORECASE):
        sys.exit("[TPU CI] ERROR: Non-finite actor/grad_norm detected in GRPO output!")
    grad_norms = _floats(r"actor/grad_norm[:\s]+([0-9.eE+-]+)", text)
    rewards = _floats(r"critic/rewards/mean[:\s]+([0-9.eE+-]+)", text)
    corrs = _floats(r"training/rollout_actor_probs_pearson_corr[:\s]+([0-9.eE+-]+)", text)
    val_accs = _floats(r"val-core/openai/gsm8k/acc/mean@1['\"]?:\s*(?:np\.float64\()?([0-9.eE+-]+)", text)
    if not rewards or not grad_norms:
        sys.exit("[TPU CI] ERROR: Missing critic/rewards/mean or actor/grad_norm in GRPO output.")
    assert max(grad_norms) > 0.0, "[TPU CI] GRPO actor/grad_norm was 0.0 on all steps (no gradient flowed)!"
    min_reward = 0.10 if smoke_test else 0.15
    assert max(rewards) >= min_reward, f"[TPU CI] GRPO best training reward {max(rewards):.4f} < {min_reward}!"
    if corrs:
        assert min(corrs) >= 0.90, (
            f"[TPU CI] Rollout-Actor logprob Pearson correlation dropped below 0.90: min={min(corrs):.4f}"
        )
    if smoke_test:
        # 5 steps at lr=1e-6 cannot move GSM8K accuracy of the 0.6B base model (it scores ~0-3% on
        # 32 prompts before training), so a pass-rate threshold would only measure sampling luck.
        # Require that validation ran; learning signal is covered by the training reward check.
        if not val_accs:
            sys.exit("[TPU CI] ERROR: No val-core/openai/gsm8k/acc/mean@1 metric in GRPO smoke output.")
    elif val_accs:
        assert max(val_accs) >= 0.25, (
            f"[TPU CI] GSM8K test pass rate (val-core/openai/gsm8k/acc/mean@1) {max(val_accs):.4f} < 0.25!"
        )
        if len(val_accs) >= 2:
            assert val_accs[-1] > val_accs[0], (
                f"[TPU CI] GSM8K test pass rate did not improve over full run: {val_accs[0]:.4f} -> {val_accs[-1]:.4f}"
            )
    print(
        f"[TPU CI] GRPO convergence & quality verified: best_reward={max(rewards):.4f}, "
        f"val_acc={val_accs[-1] if val_accs else 'N/A'}, min_corr={min(corrs) if corrs else 'N/A'}, "
        f"max_grad_norm={max(grad_norms):.4f}"
    )


VERIFIERS = {"smoke": verify_smoke, "sft": verify_sft, "grpo": verify_grpo}


def main() -> None:
    test_name, log_path, smoke_test = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
    text = open(log_path, encoding="utf-8", errors="replace").read()
    if test_name != "smoke" and not re.findall(r"step[:\s]+([1-9][0-9]*)", text, flags=re.IGNORECASE):
        sys.exit(f"[TPU CI] ERROR: No training steps logged in {test_name} output.")
    VERIFIERS[test_name](text, smoke_test)


if __name__ == "__main__":
    main()
