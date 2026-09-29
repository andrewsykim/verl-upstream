#!/usr/bin/env bash
# GSM8K repeated-prompt probe with configurable prefix caching. Run on a TPU host
# with the project's vLLM environment active; PYTHON may select its interpreter.
set -euo pipefail

usage() {
    cat <<EOF
Usage: ${0##*/} [--prefix-cache on|off]

  --prefix-cache on|off  Enable or disable prefix caching (default: on).
  -h, --help            Show this help message.
EOF
}

prefix_cache=on
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix-cache)
            if [[ $# -lt 2 ]]; then
                echo "Error: --prefix-cache requires on or off." >&2
                usage >&2
                exit 2
            fi
            prefix_cache="$2"
            shift 2
            ;;
        --prefix-cache=*)
            prefix_cache="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$prefix_cache" in
    on) export E2E_PREFIX_CACHE=1 ;;
    off) export E2E_PREFIX_CACHE=0 ;;
    *)
        echo "Error: --prefix-cache must be on or off (got: $prefix_cache)." >&2
        usage >&2
        exit 2
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export E2E_GSM8K=1
# Match the historical GRPO trigger: 8 prompts x 4 copies, 64-token blocks.
export E2E_MODEL="${E2E_MODEL:-Qwen/Qwen3-0.6B}"
export E2E_QUANTIZATION="${E2E_QUANTIZATION-}"
export E2E_BLOCK_SIZE="${E2E_BLOCK_SIZE:-64}"
export E2E_MAX_MODEL_LEN="${E2E_MAX_MODEL_LEN:-1024}"
export E2E_MAX_NUM_SEQS="${E2E_MAX_NUM_SEQS:-32}"
export E2E_MAX_BATCHED_TOKENS="${E2E_MAX_BATCHED_TOKENS:-8192}"
export E2E_GPU_MEM_UTIL="${E2E_GPU_MEM_UTIL:-0.6}"
export E2E_TP="${E2E_TP:-8}" E2E_EP="${E2E_EP:-0}"
# Use one data-parallel replica for all repeated prompts in this probe.
export VLLM_DP_SIZE=1
export E2E_SEED="${E2E_SEED:-42}"
result_dir="${E2E_GSM8K_RESULT_DIR:-/tmp/hybrid_pool_gsm8k}"
mkdir -p -- "$result_dir"

"${PYTHON:-python3}" -u "$script_dir/hybrid_pool_e2e.py" \
    2>&1 | tee "$result_dir/prefix_${prefix_cache}.log"
