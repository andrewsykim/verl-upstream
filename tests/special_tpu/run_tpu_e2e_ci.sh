#!/usr/bin/env bash
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
#
# End-to-End CI Runner for TPU v6e on GKE KubeRay.
#
# Every invocation runs one CI tier and owns the full lifecycle of its own RayCluster:
#   1. provision  - render tests/special_tpu/gke/raycluster-ci.yaml into a uniquely named,
#                   Kueue-queued RayCluster whose TPU workers are single-host v6e-4 (2x2)
#                   subslices of the v6e-8 (2x4) node pools (4/8-chip tiers) or a whole 1-chip
#                   ct6e-standard-1t node (1-chip tier). Kueue keeps it suspended until TPU
#                   quota is free, so concurrent CI runs queue (FIFO) instead of fighting for hosts.
#   2. run        - submit each of the tier's tests as a Ray job and verify its metrics.
#   3. tear down  - delete the RayCluster on exit, whether the tests passed, failed, timed
#                   out or were cancelled.
#
# Tiers (add new tests to the matching tier in run_tier):
#   v6e-1chip - 1 x v6e-1 (whole ct6e-standard-1t node). TPU platform tests:
#               test_tpu_platform (torch_tpu import / to(device) / matmul / autograd / TPU-vs-CPU).
#   v6e-4chip - 1 x v6e-4 subslice. Trainer tests:
#               test_trainer_sft (Qwen3-0.6B GSM8K SFT, TorchTitan FSDP2).
#   v6e-8chip - 2 x v6e-4 subslices. RL tests:
#               test_rl_grpo (Qwen3-0.6B GSM8K GRPO, 4-chip trainer on tpu-group-0 + 4-chip
#               vLLM rollout on tpu-group-1).
# Note: multi-chip ICI sessions smaller than a full v6e host (e.g. 2 chips) are rejected by
# libtpu (START_SESSION failed), so a single host (4 chips) is the smallest multi-chip unit.
#
# Usage:
#   bash tests/special_tpu/run_tpu_e2e_ci.sh <v6e-1chip|v6e-4chip|v6e-8chip>
#   bash tests/special_tpu/run_tpu_e2e_ci.sh teardown [tier]   # delete this run's clusters

set -euo pipefail

MODE="${1:?usage: $0 <v6e-1chip|v6e-4chip|v6e-8chip|teardown> [tier]}"
export CLUSTER_NAME="${CLUSTER_NAME:-jialeic-ci-v6e8-2s-spot}"
export REGION="${REGION:-us-central2}"
export PROJECT="${PROJECT:-tpu-pytorch}"
export RAY_NAMESPACE="${RAY_NAMESPACE:-default}"
# Each tier has its own Kueue LocalQueue: ${KUEUE_QUEUE_PREFIX}-{1chip,4chip,8chip}.
export KUEUE_QUEUE_PREFIX="${KUEUE_QUEUE_PREFIX:-verl-tpu-ci}"
KUEUE_QUEUE=""
TIER_NAME=""
export TPU_CI_IMAGE="${TPU_CI_IMAGE:-us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20260928-tsync0927}"
export SMOKE_TEST="${SMOKE_TEST:-1}"
export PORT_FORWARD_PORT="${PORT_FORWARD_PORT:-$((28000 + RANDOM % 1000))}"
# Maximum time to wait in the Kueue queue before giving up.
export QUEUE_TIMEOUT_MINS="${QUEUE_TIMEOUT_MINS:-120}"
# CI RayClusters older than this are considered leaked (e.g. runner pod crashed) and reaped.
export STALE_CLUSTER_SECONDS="${STALE_CLUSTER_SECONDS:-10800}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${GITHUB_RUN_ID:-local$(date +%s)}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"

PF_PID=""
CURRENT_RAY_JOB_ID=""
RAY_CLUSTER_NAME=""
HEAD_POD=""
WORKER_REPLICAS=1
WORKER_TPU_CHIPS=4

log() {
    echo "[TPU CI $(date +%H:%M:%S)] $*"
}

connect_gke_cluster() {
    if [[ -n "${KUBERNETES_SERVICE_HOST:-}" ]]; then
        unset KUBECONFIG
        if ! command -v kubectl >/dev/null 2>&1; then
            curl -sLO https://dl.k8s.io/release/v1.31.0/bin/linux/amd64/kubectl
            chmod +x kubectl
            mv kubectl /usr/local/bin/kubectl
        fi
        log "Running inside GKE pod; using in-cluster Kubernetes service account."
    else
        log "Connecting to GKE cluster ${CLUSTER_NAME} (${REGION}, ${PROJECT})"
        gcloud container clusters get-credentials "${CLUSTER_NAME}" \
            --region "${REGION}" \
            --project "${PROJECT}" \
            --dns-endpoint
    fi
}

delete_ray_clusters() {
    # Delete CI RayClusters matching a label selector without blocking on pod termination.
    local selector="$1"
    kubectl delete raycluster -n "${RAY_NAMESPACE}" -l "${selector}" --wait=false --ignore-not-found || true
}

cleanup() {
    local exit_code=$?
    set +e
    if [[ -n "${CURRENT_RAY_JOB_ID}" && -n "${HEAD_POD}" ]]; then
        log "Stopping active Ray job ${CURRENT_RAY_JOB_ID}..."
        kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- \
            ray job stop --address http://127.0.0.1:8265 "${CURRENT_RAY_JOB_ID}" >/dev/null 2>&1
        CURRENT_RAY_JOB_ID=""
    fi
    if [[ -n "${PF_PID}" ]] && kill -0 "${PF_PID}" 2>/dev/null; then
        kill "${PF_PID}" 2>/dev/null
    fi
    if [[ -n "${RAY_CLUSTER_NAME}" ]]; then
        log "Tearing down RayCluster ${RAY_CLUSTER_NAME} (exit code ${exit_code})..."
        kubectl delete raycluster -n "${RAY_NAMESPACE}" "${RAY_CLUSTER_NAME}" --wait=false --ignore-not-found
        RAY_CLUSTER_NAME=""
    fi
    exit "${exit_code}"
}

ensure_kueue_queue() {
    # Bootstrap only: create the Kueue objects if this tier's LocalQueue does not exist yet. The
    # queues and quotas are shared by every PR, so an existing setup is never modified from CI;
    # changes to kueue-tpu-ci.yaml are rolled out by an admin with `kubectl apply -f`.
    local queue="${KUEUE_QUEUE_PREFIX}-${MODE#v6e-}"
    if kubectl get localqueue -n "${RAY_NAMESPACE}" "${queue}" >/dev/null 2>&1; then
        log "Kueue LocalQueue ${queue} exists."
    else
        log "Kueue LocalQueue ${queue} not found; applying tests/special_tpu/gke/kueue-tpu-ci.yaml."
        kubectl apply -f tests/special_tpu/gke/kueue-tpu-ci.yaml
    fi
}

reap_stale_clusters() {
    local stale
    stale="$(kubectl get raycluster -n "${RAY_NAMESPACE}" -l verl-ci/managed=true -o json | python3 -c '
import datetime, json, sys
max_age = int(sys.argv[1])
now = datetime.datetime.now(datetime.timezone.utc)
for item in json.load(sys.stdin)["items"]:
    created = datetime.datetime.fromisoformat(item["metadata"]["creationTimestamp"].replace("Z", "+00:00"))
    if (now - created).total_seconds() > max_age:
        print(item["metadata"]["name"])
' "${STALE_CLUSTER_SECONDS}")"
    for name in ${stale}; do
        log "Reaping leaked CI RayCluster ${name} (older than ${STALE_CLUSTER_SECONDS}s)..."
        kubectl delete raycluster -n "${RAY_NAMESPACE}" "${name}" --wait=false --ignore-not-found || true
    done
}

provision_ray_cluster() {
    # shape: "v6e-4" = 2x2 single-host subslice of a 2x4 node pool (4 chips per worker);
    #        "v6e-1" = whole 1x1 node from the ct6e-standard-1t pool (1 chip per worker).
    local tier="$1" shape="$2" replicas="$3" head_cpu="$4" head_mem="$5" worker_cpu="$6" worker_mem="$7"
    local node_topology torch_topology annotation_key annotation_value slicebuilder
    WORKER_REPLICAS="${replicas}"
    case "${shape}" in
        v6e-4)
            WORKER_TPU_CHIPS=4 node_topology=2x4 torch_topology=2,2,1
            annotation_key=cloud.google.com/gke-tpu-slice-topology annotation_value=2x2
            slicebuilder=localhost:8471,localhost:8472,localhost:8473,localhost:8474
            ;;
        v6e-1)
            # No subslice annotation: the pod owns the whole 1-chip node.
            WORKER_TPU_CHIPS=1 node_topology=1x1 torch_topology=1,1,1
            annotation_key=verl-ci/tpu-slice-topology annotation_value=1x1
            slicebuilder=localhost:8471
            ;;
        *)
            log "ERROR: unknown worker shape ${shape}" >&2
            exit 1
            ;;
    esac
    TIER_NAME="${tier}"
    KUEUE_QUEUE="${KUEUE_QUEUE_PREFIX}-${tier#v6e-}"
    # Keep the name short: KubeRay derives pod/service names from it (63-char limit).
    RAY_CLUSTER_NAME="$(echo "verl-ci-${tier}-${RUN_ID}-${RUN_ATTEMPT}" | tr '[:upper:]_' '[:lower:]-' | cut -c1-40 | sed 's/-*$//')"

    log "Provisioning RayCluster ${RAY_CLUSTER_NAME} (${replicas} x ${shape}, queue=${KUEUE_QUEUE}, image=${TPU_CI_IMAGE})"
    RAY_CLUSTER_NAME="${RAY_CLUSTER_NAME}" TIER="${tier}" HEAD_CPU="${head_cpu}" HEAD_MEMORY="${head_mem}" \
        WORKER_CPU="${worker_cpu}" WORKER_MEMORY="${worker_mem}" WORKER_REPLICAS="${replicas}" CI_RUN_ID="${RUN_ID}" \
        KUEUE_QUEUE="${KUEUE_QUEUE}" WORKER_TPU_CHIPS="${WORKER_TPU_CHIPS}" WORKER_NODE_TOPOLOGY="${node_topology}" \
        SLICE_ANNOTATION_KEY="${annotation_key}" SLICE_ANNOTATION_VALUE="${annotation_value}" \
        TORCH_TPU_TOPOLOGY="${torch_topology}" SLICEBUILDER_ADDRESSES="${slicebuilder}" \
        CI_OWNER="${GITHUB_REPOSITORY:-local}/${GITHUB_REF_NAME:-$(hostname)}#${RUN_ID}" \
        python3 - tests/special_tpu/gke/raycluster-ci.yaml <<'PY' | kubectl apply -f -
import os
import string
import sys

keys = ["RAY_CLUSTER_NAME", "RAY_NAMESPACE", "KUEUE_QUEUE", "TIER", "CI_OWNER", "CI_RUN_ID", "HEAD_CPU",
        "HEAD_MEMORY", "WORKER_CPU", "WORKER_MEMORY", "WORKER_REPLICAS", "WORKER_TPU_CHIPS",
        "WORKER_NODE_TOPOLOGY", "SLICE_ANNOTATION_KEY", "SLICE_ANNOTATION_VALUE", "TORCH_TPU_TOPOLOGY",
        "SLICEBUILDER_ADDRESSES"]
values = {k: os.environ[k] for k in keys}
values["IMAGE"] = os.environ["TPU_CI_IMAGE"]
print(string.Template(open(sys.argv[1]).read()).substitute(values))
PY
}

wait_for_ray_cluster() {
    # 1. Wait for Kueue admission (RayCluster un-suspended).
    local deadline=$((SECONDS + QUEUE_TIMEOUT_MINS * 60))
    local last_report=-1000
    while (( SECONDS < deadline )); do
        local suspended
        suspended="$(kubectl get raycluster -n "${RAY_NAMESPACE}" "${RAY_CLUSTER_NAME}" -o jsonpath='{.spec.suspend}')"
        if [[ "${suspended}" != "true" ]]; then
            log "RayCluster ${RAY_CLUSTER_NAME} admitted by Kueue."
            break
        fi
        if (( SECONDS - last_report >= 60 )); then
            log "Queued in Kueue (${KUEUE_QUEUE}); waiting for TPU quota. Queue state:"
            kubectl get localqueue -n "${RAY_NAMESPACE}" "${KUEUE_QUEUE}" --no-headers 2>/dev/null || true
            kubectl get raycluster -n "${RAY_NAMESPACE}" -l "verl-ci/managed=true,verl-ci/tier=${TIER_NAME}" \
                -o custom-columns=NAME:.metadata.name,SUSPENDED:.spec.suspend,STATE:.status.state --no-headers || true
            last_report=${SECONDS}
        fi
        sleep 10
    done
    if [[ "$(kubectl get raycluster -n "${RAY_NAMESPACE}" "${RAY_CLUSTER_NAME}" -o jsonpath='{.spec.suspend}')" == "true" ]]; then
        log "ERROR: RayCluster not admitted within ${QUEUE_TIMEOUT_MINS} minutes." >&2
        exit 1
    fi

    # 2. Wait for head + TPU worker pods to be Running and all containers ready.
    local want_pods=$((1 + WORKER_REPLICAS))
    local want_tpu="0.0/$((WORKER_TPU_CHIPS * WORKER_REPLICAS)).0 TPU"
    # Generous: the 1-chip pool scales from zero (node boot + image pull).
    deadline=$((SECONDS + 1500))
    while (( SECONDS < deadline )); do
        local ready_pods
        ready_pods="$(kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" --no-headers 2>/dev/null \
            | awk '{split($2, r, "/"); if (r[1] == r[2] && $3 == "Running") c++} END {print c+0}')"
        if [[ "${ready_pods}" -ge "${want_pods}" ]]; then
            break
        fi
        log "Ready pods: ${ready_pods}/${want_pods} (1 head + ${WORKER_REPLICAS} TPU worker(s))."
        sleep 10
    done
    kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -o wide
    if [[ "${ready_pods}" -lt "${want_pods}" ]]; then
        log "ERROR: RayCluster ${RAY_CLUSTER_NAME} pods not ready within 25 minutes of admission." >&2
        kubectl get events -n "${RAY_NAMESPACE}" --sort-by=.lastTimestamp 2>/dev/null \
            | grep "${RAY_CLUSTER_NAME}" | tail -n 20 >&2 || true
        exit 1
    fi

    HEAD_POD="$(kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=head" \
        -o jsonpath='{.items[0].metadata.name}')"

    # 3. Wait for the TPU workers to register their chips with Ray.
    deadline=$((SECONDS + 300))
    while (( SECONDS < deadline )); do
        local tpu_usage
        tpu_usage="$(kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- ray status 2>/dev/null \
            | awk '/[0-9.]+\/[0-9.]+ TPU([[:space:]]|$)/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); print $0; exit}' || true)"
        log "Ray TPU status: ${tpu_usage:-<initializing>} (want ${want_tpu})"
        if [[ "${tpu_usage}" == "${want_tpu}" ]]; then
            return 0
        fi
        sleep 10
    done
    log "ERROR: Ray cluster ${RAY_CLUSTER_NAME} did not register $((WORKER_TPU_CHIPS * WORKER_REPLICAS)) TPU chips." >&2
    kubectl get pods -n "${RAY_NAMESPACE}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -o wide >&2
    exit 1
}

connect_ray_dashboard() {
    local svc="${RAY_CLUSTER_NAME}-head-svc"
    local deadline=$((SECONDS + 60))
    # Runner pods live in the same GKE cluster: talk to the head service directly.
    while (( SECONDS < deadline )); do
        if curl -sf "http://${svc}.${RAY_NAMESPACE}.svc.cluster.local:8265/api/version" >/dev/null 2>&1; then
            export RAY_ADDRESS="http://${svc}.${RAY_NAMESPACE}.svc.cluster.local:8265"
            log "Using in-cluster Ray head service: ${RAY_ADDRESS}"
            return 0
        fi
        if [[ -z "${KUBERNETES_SERVICE_HOST:-}" ]]; then
            break
        fi
        sleep 5
    done

    log "Starting port-forward to svc/${svc} on localhost:${PORT_FORWARD_PORT}..."
    kubectl port-forward -n "${RAY_NAMESPACE}" "svc/${svc}" "${PORT_FORWARD_PORT}:8265" >/dev/null 2>&1 &
    PF_PID=$!
    deadline=$((SECONDS + 60))
    while (( SECONDS < deadline )); do
        if curl -sf "http://127.0.0.1:${PORT_FORWARD_PORT}/api/version" >/dev/null 2>&1; then
            export RAY_ADDRESS="http://127.0.0.1:${PORT_FORWARD_PORT}"
            log "Port-forward established at ${RAY_ADDRESS}"
            return 0
        fi
        sleep 1
    done
    log "ERROR: Failed to reach the Ray head service." >&2
    exit 1
}

submit_and_verify_ray_job() {
    local test_name="$1"
    local timeout_mins="$2"
    local test_env_json="$3"
    shift 3
    local entrypoint=("$@")

    local sub_id="ci_${test_name}_$(date +%Y%m%d_%H%M%S)"
    local log_file="/tmp/${sub_id}.log"

    local runtime_env
    runtime_env="$(TEST_ENV="${test_env_json}" python3 -c '
import json, os
env_vars = {
    "PYTHONPATH": ".",
    "PYTHONUNBUFFERED": "1",
    "VERL_PLATFORM": "tpu",
    "SMOKE_TEST": os.environ.get("SMOKE_TEST", "1"),
    "RAY_memory_monitor_refresh_ms": "0",
    "RAY_memory_usage_threshold": "0.99",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
}
env_vars.update(json.loads(os.environ["TEST_ENV"]))
print(json.dumps({
    "excludes": [".git", "logs", "*.log", "*.pt", "*.bin", "__pycache__", ".ruff_cache", ".mypy_cache"],
    "env_vars": env_vars,
}))
')"

    log "Submitting ${test_name^^} job ${sub_id}: ${entrypoint[*]}"
    ray job submit \
        --address "${RAY_ADDRESS}" \
        --submission-id "${sub_id}" \
        --working-dir "${REPO_ROOT}" \
        --runtime-env-json "${runtime_env}" \
        --no-wait \
        -- "${entrypoint[@]}"
    CURRENT_RAY_JOB_ID="${sub_id}"

    # Poll via kubectl exec so transient dashboard/port-forward drops never kill the CI run.
    local deadline=$((SECONDS + timeout_mins * 60))
    local final_status="TIMEOUT"
    while (( SECONDS < deadline )); do
        local status_out
        status_out="$(kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- \
            ray job status --address http://127.0.0.1:8265 "${sub_id}" 2>/dev/null \
            | grep -E "Job '.*' (succeeded|failed|was stopped)|Status message|Status for job" || true)"
        log "${sub_id}: $(tail -n 1 <<<"${status_out}")"
        if [[ -z "${status_out}" ]] && ! kubectl get pod -n "${RAY_NAMESPACE}" "${HEAD_POD}" >/dev/null 2>&1; then
            log "ERROR: Ray head pod ${HEAD_POD} is gone (evicted or preempted)." >&2
            final_status="HEAD_LOST"
            break
        fi
        local status_lower="${status_out,,}"
        if [[ "${status_lower}" == *"' succeeded"* || "${status_lower}" == *"': succeeded"* ]]; then
            final_status="SUCCEEDED"
            break
        elif [[ "${status_lower}" =~ \':?\ (failed|stopped|was\ stopped) ]]; then
            final_status="FAILED"
            break
        fi
        sleep 20
    done
    if [[ "${final_status}" == "TIMEOUT" ]]; then
        log "Job ${sub_id} exceeded ${timeout_mins} minutes; stopping it."
        kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- \
            ray job stop --address http://127.0.0.1:8265 "${sub_id}" >/dev/null 2>&1 || true
    fi
    CURRENT_RAY_JOB_ID=""

    log "Fetching full job logs for ${sub_id}..."
    kubectl exec -n "${RAY_NAMESPACE}" "${HEAD_POD}" -c ray-head -- \
        ray job logs --address http://127.0.0.1:8265 "${sub_id}" > "${log_file}" 2>&1 || true
    tail -n 150 "${log_file}"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        echo "log_file=${log_file}" >> "${GITHUB_OUTPUT}"
    fi

    if [[ "${final_status}" != "SUCCEEDED" ]]; then
        log "ERROR: Job ${sub_id} ended with status ${final_status}" >&2
        exit 1
    fi

    python3 tests/special_tpu/verify_tpu_e2e_log.py "${test_name}" "${log_file}" "${SMOKE_TEST}"
    log "${test_name^^} E2E test PASSED!"
}

# ---------------------------------------------------------------------------------------------
# Tests. Each runs as one Ray job on the tier's RayCluster; add new tests as functions here and
# call them from the matching tier in run_tier.
# ---------------------------------------------------------------------------------------------

test_tpu_platform() {
    # torch_tpu import / host<->device copy / bf16 matmul / autograd / TPU-vs-CPU MLP on 1 chip.
    submit_and_verify_ray_job smoke 15 '{}' python3 tests/special_tpu/tpu_smoke_test.py
}

test_trainer_sft() {
    # Qwen3-0.6B GSM8K SFT, TorchTitan FSDP2 on one v6e-4 subslice.
    submit_and_verify_ray_job sft 30 \
        '{"NNODES_TRAINER": "1", "N_CHIPS_TRAINER": "4"}' \
        bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
}

test_rl_grpo() {
    # Qwen3-0.6B GSM8K GRPO: 4-chip TorchTitan trainer on tpu-group-0 + 4-chip vLLM rollout on
    # tpu-group-1 (see TPUPlatform.auto_assign_accelerator_type).
    # Disable Qwen3 <think>: with thinking on, 512-token smoke rollouts are all clipped before
    # "#### <answer>" and the reward is ~0. 8 prompts x 4 samples per step keeps the
    # training-reward check stable.
    submit_and_verify_ray_job grpo 40 \
        '{"NNODES_TRAINER": "1", "N_CHIPS_TRAINER": "4", "NNODES_ROLLOUT": "1", "N_CHIPS_ROLLOUT": "4"}' \
        bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        trainer.val_before_train=True \
        data.val_max_samples=32 \
        data.train_batch_size=8 \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.rollout.n=4
}

# ---------------------------------------------------------------------------------------------
# CI tiers: one RayCluster per tier, sized by chip count, running that tier's tests in order.
# ---------------------------------------------------------------------------------------------

run_tier() {
    local tier="$1"
    case "${tier}" in
        v6e-1chip)
            # TPU platform tests: a whole ct6e-standard-1t (1x1) node.
            provision_ray_cluster "${tier}" v6e-1 1 2 8Gi 32 128Gi
            wait_for_ray_cluster
            connect_ray_dashboard
            test_tpu_platform
            ;;
        v6e-4chip)
            # Trainer tests: one v6e-4 (2x2) subslice.
            provision_ray_cluster "${tier}" v6e-4 1 4 16Gi 96 400Gi
            wait_for_ray_cluster
            connect_ray_dashboard
            test_trainer_sft
            ;;
        v6e-8chip)
            # RL tests: two v6e-4 subslices (trainer slice + rollout slice).
            provision_ray_cluster "${tier}" v6e-4 2 4 16Gi 96 400Gi
            wait_for_ray_cluster
            connect_ray_dashboard
            test_rl_grpo
            ;;
        *)
            echo "Usage: $0 <v6e-1chip|v6e-4chip|v6e-8chip|teardown> [tier]" >&2
            exit 1
            ;;
    esac
}

connect_gke_cluster

if [[ "${MODE}" == "teardown" ]]; then
    # Safety net for cancelled workflows: delete every cluster created by this CI run.
    selector="verl-ci/managed=true,verl-ci/run-id=${RUN_ID}"
    if [[ -n "${2:-}" ]]; then
        selector="${selector},verl-ci/tier=${2}"
    fi
    log "Deleting RayClusters matching ${selector}"
    delete_ray_clusters "${selector}"
    exit 0
fi

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ensure_kueue_queue
reap_stale_clusters
run_tier "${MODE}"
