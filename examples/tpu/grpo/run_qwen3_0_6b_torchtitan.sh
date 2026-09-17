#!/usr/bin/env bash
# GRPO on Google Cloud TPU (v7x / Ironwood) with torchtitan as the training engine.
#
# Topology assumed by the defaults below: a single-host v7x slice exposing 8 PJRT
# devices (4 chips x 2 chiplets), one verl worker per device. Check with
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
#     -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh

set -xeuo pipefail

# --- Platform -----------------------------------------------------------------
# Selects verl's TorchTPU platform (device "tpu", c10d backend "tpu_dist"). Ray is also
# told not to pin chips itself: verl derives each rank's device from its rank and hands
# it to torch_tpu via TPU_VISIBLE_DEVICES, and a Ray-set TPU_VISIBLE_CHIPS only adds a
# second, silently-overridden opinion.
# Ensure TPU_VISIBLE_CHIPS and TPU_VISIBLE_DEVICES are never inherited by Ray workers
unset TPU_VISIBLE_CHIPS || true
unset TPU_VISIBLE_DEVICES || true

export VERL_PLATFORM=${VERL_PLATFORM:-tpu}
export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=${RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS:-1}
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-TPU_ACCELERATOR_TYPE,TPU_NAME,TPU_HOST_BOUNDS,TPU_CHIPS_PER_HOST_BOUNDS,TPU_MULTIHOST_BACKEND,TPU_SKIP_MDS_QUERY,RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS,LIBTPU_INIT_ARGS,SKIP_JAX_PRECOMPILE,TORCH_TPU_DP_SIZE,TORCH_TPU_SLICEBUILDER_ADDRESSES,TORCH_TPU_TOPOLOGY,TPU_PROCESS_ADDRESSES}
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=${VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY:-VERL_,TORCH_TPU_,RAY_}
export PYTHONUNBUFFERED=1
export TPU_SKIP_MDS_QUERY=true
export TPU_ACCELERATOR_TYPE=${TPU_ACCELERATOR_TYPE:-tpu7x}
export TORCH_TPU_DP_SIZE=${TORCH_TPU_DP_SIZE:-4}

# Deep transformer stacks blow the default limit while AOT-tracing on TPU.
export TORCH_DYNAMO_RECOMPILE_LIMIT=${TORCH_DYNAMO_RECOMPILE_LIMIT:-100}

# --- Topology -----------------------------------------------------------------
# PJRT devices per host, NOT chips: a v7x chip exposes two.
NUM_TPU=${NUM_TPU:-8}
NNODES=${NNODES:-4}

# --- Assets -------------------------------------------------------------------
# Defaults point at the GCS-fuse mount shared by every pod in the cluster, so no
# rank downloads anything at startup.
MODEL_PATH=${MODEL_PATH:-/data/jialei/assets/hf/Qwen3-0.6B}
TRAIN_FILES=${TRAIN_FILES:-/data/jialei/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-/data/jialei/data/gsm8k/test.parquet}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}

# --- torchtitan parallelism ---------------------------------------------------
# Pure FSDP2 across the slice. TP would need the TPU mesh to match the physical
# 2x2x1 torus, so leave it at 1 until that is validated.
FSDP_SIZE=${FSDP_SIZE:-$((NUM_TPU * NNODES))}
TP_SIZE=${TP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}

# TPU only has "sdpa": flex/flex_flash need FlexAttention's Triton/CUDA kernels and
# varlen needs FlashAttention, neither of which has a TPU backend.
ATTN_TYPE=${ATTN_TYPE:-sdpa}
# Activation-checkpoint recompute happens on the autograd backward thread, which does
# not carry the TPU device binding; keep it off until that is resolved.
AC_MODE=${AC_MODE:-none}
# torch.compile/Dynamo on TPU recompiles per distinct shape and is not yet reliable
# for this path -- run eager.
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}

TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-10}
VERL_EXP_NAME=${VERL_EXP_NAME:-qwen3-0.6b-torchtitan-tpu7x}

# Keep the global batch an exact multiple of the device count so every rank gets the
# same number of tokens: TPU recompiles on every new shape, and a ragged tail would
# also desynchronize the collectives.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((NUM_TPU * NNODES * 4))}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$((NUM_TPU * NNODES))}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_N=${ROLLOUT_N:-4}

common_params=(
    model_engine=torchtitan
    algorithm.adv_estimator=grpo
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.seed=42
    actor_rollout_ref.model.path="${MODEL_PATH}"
    # sdpa attention takes no packed-sequence mask, so every row must hold one sequence.
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.min_lr_factor=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${FSDP_SIZE}"
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TP_SIZE}"
    actor_rollout_ref.actor.torchtitan.expert_parallel_size="${EP_SIZE}"
    actor_rollout_ref.actor.torchtitan.attn_type="${ATTN_TYPE}"
    actor_rollout_ref.actor.torchtitan.activation_checkpoint="${AC_MODE}"
    actor_rollout_ref.actor.torchtitan.use_torch_compile="${USE_TORCH_COMPILE}"
    # TPU has no caching allocator to reclaim, and host<->HBM copies are expensive;
    # offloading costs far more than it saves for a 0.6B model.
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
    actor_rollout_ref.rollout.tensor_model_parallel_size="${NUM_TPU}"
    actor_rollout_ref.rollout.data_parallel_size="${NNODES}"
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.n_gpus_per_node="${NUM_TPU}"
    # vLLM-on-TPU has no sleep/wake_up: the KV cache stays resident for the whole run.
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.enforce_eager=True
    algorithm.kl_ctrl.kl_coef=0.001
    trainer.logger=['console','file']
    trainer.project_name='verl_grpo_tpu'
    trainer.experiment_name="${VERL_EXP_NAME}"
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-False}
    trainer.n_gpus_per_node="${NUM_TPU}"
    trainer.nnodes="${NNODES}"
    trainer.device=tpu
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}"
)

python3 -m verl.trainer.main_ppo "${common_params[@]}" "$@"
