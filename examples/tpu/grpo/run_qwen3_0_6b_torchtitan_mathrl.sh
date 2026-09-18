#!/usr/bin/env bash
# =============================================================================
# GRPO on Google Cloud TPU (v7x / Ironwood) with torchtitan & Math RL Dataset
# =============================================================================
# Combines veRL multi-host TPU7x torchtitan training + vLLM TPU rollout with
# the Math RL recipe (OpenMathInstruct2 + GSM8K eval, math_verify reward scoring,
# validation dumps, rollout dumps, and TensorBoard logging).
#
# Topology assumed by the defaults below: a multi-host v7x slice exposing 8 PJRT
# devices per host (4 chips x 2 chiplets), 4 hosts (32 total PJRT devices).
# Check with:
#   kubectl get node -l ray.io/accelerator-type=TPU-V7X -o json | jq '.items[].metadata.labels'
# and override NUM_TPU / NNODES if your slice differs.
#
# Launch under Ray (the TPU chips live on the Ray worker pods, not the head):
#
#   ray job submit --address "${RAY_ADDRESS}" --working-dir . \
#     --runtime-env-json '{
#       "excludes": [".git", "logs", "*.log", "*.pt", "*.bin", "wheels"],
#       "env_vars": {"VERL_PLATFORM": "tpu"}
#     }' \
#     -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan_mathrl.sh
# =============================================================================

set -xeuo pipefail

# --- Platform & TPU Environment ----------------------------------------------
# Selects verl's TorchTPU platform (device "tpu", c10d backend "tpu_dist"). Ray is also
# told not to pin chips itself: verl derives each rank's device from its rank and hands
# it to torch_tpu via TPU_VISIBLE_DEVICES, and a Ray-set TPU_VISIBLE_CHIPS only adds a
# second, silently-overridden opinion.
unset TPU_VISIBLE_CHIPS || true
unset TPU_VISIBLE_DEVICES || true

export VERL_PLATFORM=${VERL_PLATFORM:-tpu}
export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=${RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS:-1}
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-TPU_ACCELERATOR_TYPE,TPU_NAME,TPU_HOST_BOUNDS,TPU_CHIPS_PER_HOST_BOUNDS,TPU_MULTIHOST_BACKEND,TPU_SKIP_MDS_QUERY,RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS,LIBTPU_INIT_ARGS,SKIP_JAX_PRECOMPILE,TORCH_TPU_DP_SIZE,TORCH_TPU_SLICEBUILDER_ADDRESSES,TORCH_TPU_TOPOLOGY,TPU_PROCESS_ADDRESSES}
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=${VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY:-VERL_,TORCH_TPU_,RAY_,REWARD_}
export PYTHONUNBUFFERED=1
export TPU_SKIP_MDS_QUERY=true
export TPU_ACCELERATOR_TYPE=${TPU_ACCELERATOR_TYPE:-tpu7x}

# Deep transformer stacks blow the default limit while AOT-tracing on TPU.
export TORCH_DYNAMO_RECOMPILE_LIMIT=${TORCH_DYNAMO_RECOMPILE_LIMIT:-100}

# --- Topology & Parallelism ---------------------------------------------------
# PJRT devices per host, NOT chips: a v7x chip exposes two chiplets.
# 2x2x4 TPU7x slice = 16 chips = 32 PJRT devices across 4 hosts (8 devices/host).
NUM_TPU=${NUM_TPU:-8}
NNODES=${NNODES:-4}
TP_SIZE=${TP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-$((NUM_TPU * NNODES))}
EP_SIZE=${EP_SIZE:-1}
export TORCH_TPU_DP_SIZE=${TORCH_TPU_DP_SIZE:-$(( FSDP_SIZE / TP_SIZE ))}

# --- Paths (site-specific / cluster mounts) -----------------------------------
DATA_DIR=${DATA_DIR:-/data/jialei/data/openmathinstruct2}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train_qsplit.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val_1k_qsplit.parquet}
GSM8K_TEST_FILE=${GSM8K_TEST_FILE:-$DATA_DIR/gsm8k_test.parquet}

if [ "${SKIP_DATA_CHECK:-0}" != "1" ]; then
  for f in "${TRAIN_FILE}" "${VAL_FILE}" "${GSM8K_TEST_FILE}"; do
    if [ ! -s "$f" ]; then
      echo "[mathrl] Warning: data file $f is missing or empty locally. Set SKIP_DATA_CHECK=1 if data is only mounted on remote worker nodes."
      test -s "$f" || { echo "[mathrl] ABORT: missing data file $f"; exit 2; }
    fi
  done
fi
# VAL_FILES=${VAL_FILES:-"['${VAL_FILE}','${GSM8K_TEST_FILE}']"}
VAL_FILES=${VAL_FILES:-"['${VAL_FILE}']"}

REWARD_FN_PATH=${REWARD_FN_PATH:-/data/jialei/reward/maxtext_math_reward.py}
MODEL_PATH=${MODEL_PATH:-/data/jialei/assets/hf/Qwen3-0.6B}

LOG_DIR=${LOG_DIR:-$HOME/meta-RL/logs}
CKPT_DIR=${CKPT_DIR:-$HOME/meta-RL/ckpt}
TB_ROOT=${TB_ROOT:-/tmp/tb_local}
TB_MIRROR_ROOT=${TB_MIRROR_ROOT:-/workspace/meta-RL/.home/tensorboard_log}
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${TB_ROOT}"

# --- Scale & Training Knobs ---------------------------------------------------
TOTAL_STEPS=${TOTAL_STEPS:-2}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-50}
SEED=${SEED:-1}
RUN_TAG=${RUN_TAG:-${PRESET:-v5}}

# Preset options: PRESET=stab applies the agreed stability-round recipe
if [ "${PRESET:-}" = "stab" ]; then
  : "${ROLLOUT_TEMPERATURE:=1.0}" "${ROLLOUT_TOP_P:=1.0}" "${ROLLOUT_TOP_K:=-1}"
  : "${REWARD_FMT_WEIGHT:=0}" "${REWARD_OVERLONG_BUFFER:=1024}" "${REWARD_OVERLONG_PENALTY:=1.0}"
  : "${FILTER_OVERLONG_PROMPTS:=True}" "${KL_COEF:=0}" "${KL_TYPE:=low_var_kl}" "${TEST_FREQ:=10}"
  echo "[mathrl] PRESET=stab: T=1 top_p=1 top_k=-1 fmt_w=0 overlong=1024/1.0 filter_overlong_prompts=True KL=0 test_freq=10"
fi

# --- Sequence Lengths & Budgets -----------------------------------------------
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

# --- Reward Worker Knobs ------------------------------------------------------
export REWARD_MV_POOL=${REWARD_MV_POOL:-1}
export REWARD_MV_PROCS=${REWARD_MV_PROCS:-4}
export REWARD_MV_TIMEOUT=${REWARD_MV_TIMEOUT:-5}
export REWARD_MATH_VERIFY_MAX_CHARS=${REWARD_MATH_VERIFY_MAX_CHARS:-400}
export REWARD_FMT_WEIGHT=${REWARD_FMT_WEIGHT:-0.1}
export REWARD_OVERLONG_BUFFER=${REWARD_OVERLONG_BUFFER:-0}
export REWARD_OVERLONG_PENALTY=${REWARD_OVERLONG_PENALTY:-1.0}
export REWARD_MAX_RESP_LEN=${MAX_RESPONSE_LENGTH}

PROJECT_NAME=${PROJECT_NAME:-trackA_phase0_tpu}
W=$(( NNODES * NUM_TPU ))
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0p6b_mathrl_tpu_${RUN_TAG}_seed${SEED}_${NNODES}n${W}t_$(date +%Y%m%d_%H%M)}
TB_DIR=${TB_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
TB_MIRROR=${TB_MIRROR_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
VAL_DUMP_DIR=${LOG_DIR}/${EXPERIMENT_NAME}/val_dump
ROLLOUT_DUMP_DIR=${LOG_DIR}/${EXPERIMENT_NAME}/rollout_dump
mkdir -p "${TB_DIR}" "${TB_MIRROR}" "${VAL_DUMP_DIR}" "${ROLLOUT_DUMP_DIR}"

# --- Pre-flight Checks --------------------------------------------------------
if [ "${SKIP_REWARD_PREFLIGHT:-0}" != "1" ] && [ -f "${REWARD_FN_PATH}" ]; then
  python3 - "${REWARD_FN_PATH}" <<'PYEOF'
import importlib.metadata as md, importlib.util, json, os, sys, threading, warnings
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("r", sys.argv[1]); r = importlib.util.module_from_spec(spec)
try:
  spec.loader.exec_module(r)
except Exception as e:
  sys.exit(f"[mathrl] ABORT: reward import failed: {type(e).__name__}: {e}")
fmt_w = float(os.environ.get("REWARD_FMT_WEIGHT", "0.1")); buf = int(os.environ.get("REWARD_OVERLONG_BUFFER", "0"))
pen = float(os.environ.get("REWARD_OVERLONG_PENALTY", "1.0")); mx = int(os.environ.get("REWARD_MAX_RESP_LEN", "8192"))
gt = json.dumps(["\\frac{1}{2}", "\\frac{1}{2}"]); comp = "<reasoning>x</reasoning><answer>1/2</answer>"
res = {}
t = threading.Thread(target=lambda: res.__setitem__("s", r.compute_score("x", comp, gt, extra_info={"index": 0, "response_len": 100})))
t.start(); t.join()
o = res.get("s", {}); st = r.mv_stats()
print(f"[mathrl] math-verify version = {md.version('math-verify')}; reward workers = {st['idle']}/{st['cfg_procs']} idle, pool={st['pool']}, "
      f"timeout={st['cfg_timeout_s']}s; knobs fmt_w={fmt_w} overlong={buf}/{pen}; short correct answer -> {o}")
checks = [("acc == 1", o.get("acc") == 1.0), ("fmt == 1", o.get("fmt") == 1.0), ("length_penalty == 0", o.get("length_penalty") == 0.0),
          (f"score == 1 + fmt_w ({1.0 + fmt_w})", o.get("score") is not None and abs(o["score"] - (1.0 + fmt_w)) < 1e-9)]
if buf > 0:
  o2 = r.compute_score("x", comp, gt, extra_info={"index": 0, "response_len": mx})
  checks.append((f"length_penalty at cap == -{pen}", abs(o2.get("length_penalty", 0.0) + pen) < 1e-9))
  checks.append((f"score at cap == 1 + fmt_w - pen", abs(o2["score"] - (1.0 + fmt_w - pen)) < 1e-9))
bad = [name for name, ok in checks if not ok]
if bad:
  sys.exit(f"[mathrl] ABORT: reward pre-flight failed: {bad}")
if os.environ.get("REWARD_MV_POOL", "1") == "1" and (not st["pool"] or st["idle"] != st["cfg_procs"]):
  sys.exit("[mathrl] ABORT: math_verify worker pool not healthy")
print("[mathrl] reward pre-flight OK:", ", ".join(n for n, _ in checks))
PYEOF
elif [ ! -f "${REWARD_FN_PATH}" ]; then
  echo "[mathrl] Notice: custom reward file ${REWARD_FN_PATH} not found locally; skipping local reward verification."
fi

# --- Background TB Mirror & Process Traps -------------------------------------
( while true; do sleep 300; cp -r "${TB_DIR}/." "${TB_MIRROR}/" 2>/dev/null || true; done ) &
TB_SYNC_PID=$!

GUARD_PID=""; DRIVER_PID=""
on_signal() {
  echo "[mathrl] caught signal -- stopping driver ${DRIVER_PID:-<none>}"
  [ -n "${DRIVER_PID}" ] && kill -TERM "${DRIVER_PID}" 2>/dev/null || true
}
trap 'on_signal; exit 130' INT
trap 'on_signal; exit 143' TERM
on_exit() {
  if [ -n "${DRIVER_PID}" ] && kill -0 "${DRIVER_PID}" 2>/dev/null; then
    echo "[mathrl] exit: driver ${DRIVER_PID} still alive -- sending TERM"; kill -TERM "${DRIVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 "$(( ${DRIVER_KILL_GRACE:-30} / 2 ))"); do kill -0 "${DRIVER_PID}" 2>/dev/null || break; sleep 2; done
    kill -0 "${DRIVER_PID}" 2>/dev/null && { echo "[mathrl] exit: driver did not stop -- KILL"; kill -KILL "${DRIVER_PID}" 2>/dev/null || true; }
  fi
  kill ${TB_SYNC_PID} ${GUARD_PID} 2>/dev/null || true
  cp -r "${TB_DIR}/." "${TB_MIRROR}/" 2>/dev/null || true
  test -f "${TB_DIR}/COLLAPSE_ABORT.txt" && { echo "[mathrl] RUN ABORTED BY COLLAPSE GUARD:"; cat "${TB_DIR}/COLLAPSE_ABORT.txt"; }
  test -f "${TB_DIR}/COLLAPSE_WARN.txt" && { echo "[mathrl] guard warnings:"; cat "${TB_DIR}/COLLAPSE_WARN.txt"; }
  return 0
}
trap on_exit EXIT

# --- torchtitan Parallelism & Engine Setup ------------------------------------
# Pure FSDP2 across the slice with TP=1 and maximum DP (${FSDP_SIZE} shards).

# TPU only has "sdpa": flex/flex_flash need FlexAttention's Triton/CUDA kernels and
# varlen needs FlashAttention, neither of which has a TPU backend.
ATTN_TYPE=${ATTN_TYPE:-sdpa}
# Activation-checkpoint recompute happens on the autograd backward thread, which does
# not carry the TPU device binding; keep it off until that is resolved.
AC_MODE=${AC_MODE:-none}
# torch.compile/Dynamo on TPU recompiles per distinct shape and is not yet reliable
# for this path -- run eager.
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}

# --- Batch & Recipe Variables -------------------------------------------------
train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH:-256}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
rollout_n=${ROLLOUT_N:-8}
kl_loss_coef=${KL_COEF:-0.0}
kl_loss_type=${KL_TYPE:-low_var_kl}
if awk "BEGIN{exit !(${kl_loss_coef} > 0)}"; then use_kl_loss=True; else use_kl_loss=False; fi
clip_ratio_low=0.2
clip_ratio_high=0.28
temperature=${ROLLOUT_TEMPERATURE:-0.8}
top_p=${ROLLOUT_TOP_P:-0.95}
top_k=${ROLLOUT_TOP_K:-50}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
actor_lr=${ACTOR_LR:-1e-6}
filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS:-False}

# For TPU stability, rollout logprob calculation defaults to False to avoid dynamic-shape
# JIT compilation hangs in vllm_torchtpu gather_logprobs/TopK custom-call kernels.
CALCULATE_LOG_PROBS=${CALCULATE_LOG_PROBS:-False}

echo "[accounting] W=${W} batch=${train_batch_size}x${rollout_n} updates/rollout=$((train_batch_size / ppo_mini_batch_size)) (mini=${ppo_mini_batch_size}) seed=${SEED}"
echo "[recipe] lr=${actor_lr} kl_coef=${kl_loss_coef}(${kl_loss_type},use_kl=${use_kl_loss}) mini=${ppo_mini_batch_size} clip=${clip_ratio_low}/${clip_ratio_high} T=${temperature} top_p=${top_p} top_k=${top_k} cap=${MAX_RESPONSE_LENGTH} n=${rollout_n} fmt_w=${REWARD_FMT_WEIGHT} overlong=${REWARD_OVERLONG_BUFFER}/${REWARD_OVERLONG_PENALTY} filter_overlong_prompts=${filter_overlong_prompts}"
echo "[mathrl] steps=${TOTAL_STEPS} test_freq=${TEST_FREQ} save_freq=${SAVE_FREQ}"
echo "[mathrl] tensorboard -> ${TB_DIR}"
echo "[mathrl] checkpoints -> ${CKPT_DIR}/${EXPERIMENT_NAME}"
echo "[mathrl] val dumps   -> ${VAL_DUMP_DIR}"
echo "[mathrl] train dumps -> ${ROLLOUT_DUMP_DIR}"

common_params=(
    model_engine=torchtitan
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${train_batch_size}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts="${filter_overlong_prompts}"
    data.truncation='error'
    data.shuffle="${DATA_SHUFFLE:-True}"
    data.seed="${SEED}"
    data.dataloader_num_workers="${DATALOADER_NUM_WORKERS:-8}"
    actor_rollout_ref.model.path="${MODEL_PATH}"
    # sdpa attention takes no packed-sequence mask, so every row must hold one sequence.
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.actor.optim.lr="${actor_lr}"
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]'
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.min_lr_factor=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}"
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}"
    actor_rollout_ref.actor.kl_loss_type="${kl_loss_type}"
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}"
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}"
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra"]'
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${FSDP_SIZE}"
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TP_SIZE}"
    actor_rollout_ref.actor.torchtitan.expert_parallel_size="${EP_SIZE}"
    actor_rollout_ref.actor.torchtitan.attn_type="${ATTN_TYPE}"
    actor_rollout_ref.actor.torchtitan.activation_checkpoint="${AC_MODE}"
    actor_rollout_ref.actor.torchtitan.use_torch_compile="${USE_TORCH_COMPILE}"
    actor_rollout_ref.actor.torchtitan.param_offload=False
    actor_rollout_ref.actor.torchtitan.optimizer_offload=False
    actor_rollout_ref.actor.torchtitan.max_seq_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.ref.torchtitan.data_parallel_shard_size="${FSDP_SIZE}"
    actor_rollout_ref.ref.torchtitan.attn_type="${ATTN_TYPE}"
    actor_rollout_ref.ref.torchtitan.activation_checkpoint="${AC_MODE}"
    actor_rollout_ref.ref.torchtitan.use_torch_compile="${USE_TORCH_COMPILE}"
    actor_rollout_ref.ref.torchtitan.max_seq_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.name=vllm_tpu
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}"
    actor_rollout_ref.rollout.data_parallel_size="${FSDP_SIZE}"
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.n="${rollout_n}"
    actor_rollout_ref.rollout.n_gpus_per_node="${NUM_TPU}"
    # vLLM-on-TPU has no sleep/wake_up: the KV cache stays resident for the whole run.
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.temperature="${temperature}"
    actor_rollout_ref.rollout.top_p="${top_p}"
    actor_rollout_ref.rollout.top_k="${top_k}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${max_num_batched_tokens}"
    actor_rollout_ref.rollout.calculate_log_probs="${CALCULATE_LOG_PROBS}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.n=1
    trainer.balance_batch=True
    trainer.logger='["console","tensorboard"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${NUM_TPU}"
    trainer.nnodes="${NNODES}"
    trainer.device=tpu
    trainer.save_freq="${SAVE_FREQ}"
    trainer.default_local_dir="${CKPT_DIR}/${EXPERIMENT_NAME}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}"
    trainer.log_val_generations=10
    trainer.validation_data_dir="${VAL_DUMP_DIR}"
    trainer.rollout_data_dir="${ROLLOUT_DUMP_DIR}"
    trainer.resume_mode=disable
    trainer.total_epochs=100
    trainer.total_training_steps="${TOTAL_STEPS}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR='${TB_DIR}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.EXPERIMENT_NAME='${EXPERIMENT_NAME}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MV_POOL='${REWARD_MV_POOL}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MV_PROCS='${REWARD_MV_PROCS}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MV_TIMEOUT='${REWARD_MV_TIMEOUT}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MATH_VERIFY_MAX_CHARS='${REWARD_MATH_VERIFY_MAX_CHARS}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_FMT_WEIGHT='${REWARD_FMT_WEIGHT}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_OVERLONG_BUFFER='${REWARD_OVERLONG_BUFFER}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_OVERLONG_PENALTY='${REWARD_OVERLONG_PENALTY}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MAX_RESP_LEN='${REWARD_MAX_RESP_LEN}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOGPROB_FIXTURE_DIR='${LOGPROB_FIXTURE_DIR:-}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOGPROB_FIXTURE_STEP='${LOGPROB_FIXTURE_STEP:-1}'"
)

if [ "${USE_CUSTOM_REWARD:-1}" = "1" ] && [ -n "${REWARD_FN_PATH:-}" ]; then
  common_params+=(
    reward.custom_reward_function.path="${REWARD_FN_PATH}"
    reward.custom_reward_function.name=compute_score
    custom_reward_function.path="${REWARD_FN_PATH}"
    custom_reward_function.name=compute_score
  )
fi

python3 -m verl.trainer.main_ppo "${common_params[@]}" "$@" &
DRIVER_PID=$!
echo "[mathrl] driver pid ${DRIVER_PID}"

# Optional collapse guard
GUARD_LOG=${LOG_DIR}/${EXPERIMENT_NAME}/collapse_guard.log; mkdir -p "$(dirname "${GUARD_LOG}")"
if [ "${COLLAPSE_GUARD:-0}" = "1" ] && [ -f "${SCRIPTS_DIR:-$(dirname "$0")}/collapse_guard.py" ]; then
  python3 "${SCRIPTS_DIR:-$(dirname "$0")}/collapse_guard.py" --tb "${TB_DIR}" --rollout "${ROLLOUT_DUMP_DIR}" --pid "${DRIVER_PID}" --poll 60 > "${GUARD_LOG}" 2>&1 &
  GUARD_PID=$!
  echo "[mathrl] collapse guard pid ${GUARD_PID} (watching driver ${DRIVER_PID}) -> ${GUARD_LOG}"
fi

set +e
wait "${DRIVER_PID}"; DRIVER_RC=$?
while kill -0 "${DRIVER_PID}" 2>/dev/null; do wait "${DRIVER_PID}"; DRIVER_RC=$?; done
set -e
echo "[mathrl] driver exited with rc=${DRIVER_RC}"
exit ${DRIVER_RC}
