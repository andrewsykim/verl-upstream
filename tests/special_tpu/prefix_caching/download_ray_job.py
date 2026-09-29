# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Download a finished prefix-cache probe and export its prompt/response records."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import urlopen

_PROMPT_END = ' Let\'s think step by step and output the final answer after "####".'
_MARKER = re.compile(r"^(?:PROMPT|OUTPUT|GSM8K_DONE)(?: [^\n]*)?$", re.MULTILINE)
_PROMPT = re.compile(r"PROMPT row=(\d+) tokens=(\d+)")
_OUTPUT = re.compile(r"OUTPUT round=(\d+) row=(\d+) sample=(\d+) cached_tokens=(\d+) finish=(\S+)")
_DONE = re.compile(r"GSM8K_DONE cached_tokens=(\d+)")
_ENGINE_LOG = re.compile(
    r"^(?:\((?:Worker[^ ]*|EngineCore[^ ]*) pid=\d+\)|(?:INFO|WARNING|ERROR) \d\d-\d\d )", re.MULTILINE
)


def _configuration(text):
    lines = re.findall(r"^GSM8K (.+)$", text, re.MULTILINE)
    if len(lines) != 1:
        raise ValueError("Expected exactly one GSM8K configuration line")
    config = {}
    for field in lines[0].split():
        key, separator, value = field.partition("=")
        if not separator or key in config:
            raise ValueError(f"Malformed or duplicate GSM8K configuration field: {field}")
        config[key] = value
    required = {
        "model",
        "prefix_cache",
        "block_size",
        "seed",
        "cacheable_prompts",
        "temperature",
        "requested_temperature",
    }
    if missing := required - config.keys():
        raise ValueError(f"Missing GSM8K configuration fields: {sorted(missing)}")
    if config["prefix_cache"] not in ("True", "False"):
        raise ValueError("Invalid prefix_cache setting in GSM8K configuration")
    config["prefix_cache"] = config["prefix_cache"] == "True"
    try:
        for key in ("block_size", "seed", "cacheable_prompts"):
            config[key] = int(config[key])
        for key in ("temperature", "requested_temperature"):
            config[key] = float(config[key])
            if not isfinite(config[key]) or config[key] < 0:
                raise ValueError(f"Invalid {key}")
    except ValueError as error:
        raise ValueError(f"Invalid numeric GSM8K configuration: {error}") from error
    if config["block_size"] <= 0 or not 0 <= config["cacheable_prompts"] <= 8:
        raise ValueError("Invalid block size or cacheable prompt count")

    engines = re.findall(r"^.*Initializing a V1 LLM engine .*with config: (.+)$", text, re.MULTILINE)
    if len(engines) != 1:
        raise ValueError("Expected exactly one vLLM engine configuration line with TP, PP, and DP sizes")
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
        match = re.search(rf"(?:^|, ){key}=(\d+)(?:,|$)", engines[0])
        if match is None or int(match[1]) <= 0:
            raise ValueError(f"Missing or invalid {key} in engine configuration")
        config[key] = int(match[1])
    if version := re.search(r"Initializing a V1 LLM engine \(v([^)]*)\)", text):
        config["vllm_version"] = version[1]
    return config


def parse_log(text):
    """Return (configuration, generation records, summary), or raise ValueError.

    This accepts the probe's current format: eight prompts, four samples per
    prompt, and one cold round numbered zero. It removes only the newline added
    by print() from each answer. The raw job log remains the source of truth;
    an interleaved engine message in an answer is rejected as ambiguous.
    """
    config = _configuration(text)
    markers = list(_MARKER.finditer(text))
    expected_count = 8 + 32 + 1
    if len(markers) != expected_count:
        raise ValueError(f"Expected 8 prompts, 32 outputs, and GSM8K_DONE; found {len(markers)} record markers")

    prompts = {}
    records = []
    for index, marker in enumerate(markers):
        header = marker[0]
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        if text[marker.end() : marker.end() + 1] != "\n":
            raise ValueError(f"Incomplete record header: {header}")
        body = text[marker.end() + 1 : end]
        if index < 8:
            match = _PROMPT.fullmatch(header)
            if match is None or int(match[1]) != index:
                raise ValueError(f"Expected PROMPT row={index}; got {header}")
            # Worker warnings may follow the final prompt while generate() runs.
            # The appended instruction gives an exact end boundary for the prompt.
            boundary = body.find(_PROMPT_END + "\n")
            if boundary < 0:
                raise ValueError(f"Missing complete prompt text for row {index}")
            prompt = body[: boundary + len(_PROMPT_END)]
            tokens = int(match[2])
            if tokens <= 0:
                raise ValueError(f"Invalid prompt token count for row {index}")
            prompts[index] = (prompt, tokens)
        elif index < 40:
            match = _OUTPUT.fullmatch(header)
            if match is None:
                raise ValueError(f"Missing or malformed OUTPUT record: {header}")
            round_index, row, sample, hits = map(int, match.group(1, 2, 3, 4))
            expected_row, expected_sample = divmod(index - 8, 4)
            if (round_index, row, sample) != (0, expected_row, expected_sample):
                raise ValueError(f"Missing, duplicate, or reordered OUTPUT record: {header}")
            if not body.endswith("\n") or _ENGINE_LOG.search(body):
                raise ValueError(f"Incomplete output or interleaved engine log for row {row}, sample {sample}")
            prompt, tokens = prompts[row]
            if hits > tokens or hits % config["block_size"] != 0:
                raise ValueError(f"Invalid cached token count for row {row}, sample {sample}: {hits}")
            if not config["prefix_cache"] and hits:
                raise ValueError("Prefix caching is disabled but an output reports cached tokens")
            records.append(
                {
                    "input": prompt,
                    "output": body[:-1],
                    "step": round_index,
                    "prompt_row": row,
                    "sample_index": sample,
                    "prompt_tokens": tokens,
                    "cached_tokens": hits,
                    "finish_reason": match[5],
                    "source": "ray_job_logs",
                }
            )
        else:
            match = _DONE.fullmatch(header)
            if match is None:
                raise ValueError("Missing GSM8K_DONE completion marker")
            total_hits = int(match[1])

    if sum(tokens > config["block_size"] for _, tokens in prompts.values()) != config["cacheable_prompts"]:
        raise ValueError("Logged cacheable prompt count disagrees with prompt token counts")
    if sum(record["cached_tokens"] for record in records) != total_hits:
        raise ValueError("GSM8K_DONE cached token total disagrees with output records")
    summary = {
        "num_prompts": len(prompts),
        "num_outputs": len(records),
        "samples_per_prompt": 4,
        "cached_tokens": total_hits,
        "outputs_with_cache_hits": sum(record["cached_tokens"] > 0 for record in records),
        "finish_reasons": dict(Counter(record["finish_reason"] for record in records)),
    }
    return config, records, summary


# Run inside the execution pod using only the standard library. Nothing is
# written on the pod: the archive is streamed to the local collector.
REMOTE_COLLECT = r"""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

job, config = json.loads(sys.argv[1]), json.loads(sys.argv[2])
submission = job["submission_id"]
sessions = {p.parent.parent.resolve() for p in Path("/tmp/ray").glob(
    "session_*/logs/job-driver-" + submission + ".log"
)}
if len(sessions) != 1:
    raise RuntimeError(f"Expected one retained session for {submission}, found {sessions}")
session = sessions.pop()
driver = session / "logs" / f"job-driver-{submission}.log"
files = [(driver, "ray_logs/rollout/" + driver.name)]
warnings = []
supervisor = session / "logs/jobs" / f"supervisor-{submission}.log"
if supervisor.is_file():
    files.append((supervisor, "ray_logs/rollout/jobs/" + supervisor.name))
else:
    warnings.append("The job supervisor log is no longer available.")

package = Path(job["runtime_env"]["working_dir"]).name.removesuffix(".zip")
working_dir = session / "runtime_resources/working_dir_files" / package
for name in ("hybrid_pool_e2e.py", "hybrid_pool_gsm8k.py", config["launcher"]):
    source = working_dir / name
    if source.is_file():
        files.append((source, "launch/" + name))
    else:
        warnings.append(f"Submitted source no longer available: {source}")

mode = "on" if config["prefix_cache"] else "off"
result_dir = job["runtime_env"].get("env_vars", {}).get("E2E_GSM8K_RESULT_DIR", "/tmp/hybrid_pool_gsm8k")
result_dir = Path(result_dir)
if not result_dir.is_absolute():
    result_dir = working_dir / result_dir
probe_log = result_dir / f"prefix_{mode}.log"
driver_bytes = driver.read_bytes()
snapshots = {driver: (driver_bytes, driver.stat().st_mtime)}
if probe_log.is_file():
    payload = probe_log.read_bytes()
    mtime = probe_log.stat().st_mtime
    if payload and job["start_time"] / 1000 - 2 <= mtime <= job["end_time"] / 1000 + 5 \
            and driver_bytes.endswith(payload):
        files.append((probe_log, f"prefix_{mode}.log"))
        snapshots[probe_log] = (payload, mtime)
    else:
        warnings.append("The shared probe log differs from this job; it may have been overwritten. Use ray-job.log.")
else:
    warnings.append("The shared probe log is no longer available. Use ray-job.log.")

manifest = {"session": str(session), "working_dir": str(working_dir), "files": [], "warnings": warnings}
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    for source, target in files:
        if source in snapshots:
            payload, mtime = snapshots[source]
        else:
            payload, mtime = source.read_bytes(), source.stat().st_mtime
        info = tarfile.TarInfo(target)
        info.size = len(payload)
        info.mtime = mtime
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
        manifest["files"].append({"archive_path": target, "source_path": str(source),
                                  "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    info = tarfile.TarInfo("download-manifest-rollout.json")
    info.size = len(payload)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))
"""


def get_json(url):
    """Read a JSON response from the Ray dashboard."""
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def write_json(path, value):
    """Write a readable JSON artifact."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def kubectl_json(command, *args):
    """Run a read-only kubectl command and decode its JSON output."""
    completed = subprocess.run([*command, *args, "-o", "json"], check=True, capture_output=True, timeout=60)
    return json.loads(completed.stdout)


def slug(value):
    """Convert a metadata value to one safe directory-name component."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip(".-") or "unknown"


def download_pod_files(command, pod, container, job, config, run):
    """Stream, extract, and verify artifacts from the execution pod."""
    archive_path = run / "downloaded-artifacts.tar.gz"
    partial = archive_path.with_suffix(".gz.partial")
    with partial.open("wb") as output:
        completed = subprocess.run(
            [
                *command,
                "exec",
                pod,
                "-c",
                container,
                "--",
                "python3",
                "-c",
                REMOTE_COLLECT,
                json.dumps(job),
                json.dumps(config),
            ],
            stdout=output,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    if completed.returncode:
        raise RuntimeError(f"Pod download failed: {completed.stderr.decode(errors='replace')}")
    with tarfile.open(partial, "r:gz") as archive:
        archive.extractall(run, filter="data")
    manifest = json.loads((run / "download-manifest-rollout.json").read_text())
    for item in manifest["files"]:
        payload = (run / item["archive_path"]).read_bytes()
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Downloaded artifact failed checksum verification: {item['archive_path']}")
    archived_log = run / "ray_logs/rollout" / f"job-driver-{job['submission_id']}.log"
    # The Ray API reads in text mode, normalizing progress-bar carriage returns.
    # Verify the same text while retaining the pod's original bytes and hash.
    if archived_log.read_text(encoding="utf-8") != (run / "ray-job.log").read_text(encoding="utf-8"):
        raise RuntimeError("The pod driver log differs from the downloaded Ray API log.")
    partial.replace(archive_path)
    return manifest


def main():
    """Collect one finished probe without submitting or modifying any Ray job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_id", help="Ray submission ID, for example raysubmit_DKk2dPpHaiKwSrYb")
    parser.add_argument("--address", default="http://127.0.0.1:23334", help="Ray dashboard URL")
    parser.add_argument("--context", help="Kubernetes context (default: current context)")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--container", help="Execution container (default: discover ray-worker or ray-head)")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "logs")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.submission_id):
        parser.error("submission_id must contain only letters, numbers, underscores, or hyphens")
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or later is required")

    endpoint = f"{args.address.rstrip('/')}/api/jobs/{quote(args.submission_id, safe='')}"
    job = get_json(endpoint)
    if job.get("submission_id") != args.submission_id:
        raise RuntimeError("The dashboard returned a different submission ID.")
    if job["status"] != "SUCCEEDED":
        raise RuntimeError(f"Job is {job['status']}; this exporter requires a successful, complete GSM8K probe.")
    logs = get_json(endpoint + "/logs")["logs"]
    config, records, summary = parse_log(logs)

    collected = datetime.now(timezone.utc)
    model = slug(config["model"].rsplit("/", 1)[-1].lower())
    mode = "on" if config["prefix_cache"] else "off"
    tp, dp = config.get("tensor_parallel_size", "unknown"), config.get("data_parallel_size", "unknown")
    job_id = job.get("job_id") or "no-ray-job-id"
    name = (
        f"prefix_cache_gsm8k_{model}_tp{tp}-dp{dp}-gen_prefix-{mode}_"
        f"{args.submission_id}_{slug(job_id)}_{collected:%Y%m%d-%H%M%S}"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = args.output_root.resolve() / name
    run.mkdir()  # Refuse to overwrite a prior export with the same name.
    print(f"Collecting into {run}", flush=True)
    (run / "launch").mkdir()
    (run / "ray-job.log").write_text(logs, encoding="utf-8")
    write_json(run / "ray-job.json", job)
    write_json(run / "resolved-config.json", config)
    write_json(run / "summary.json", summary)
    write_json(run / "launch/runtime-env.json", job["runtime_env"])
    (run / "launch/entrypoint.sh").write_text("#!/usr/bin/env bash\n" + job["entrypoint"] + "\n", encoding="utf-8")
    write_json(
        run / "launch/launch.json",
        {
            "submission_id": args.submission_id,
            "ray_job_id": job.get("job_id"),
            "entrypoint": job["entrypoint"],
            "start_time_ms": job["start_time"],
            "end_time_ms": job["end_time"],
        },
    )
    generations = run / "generations/rollout"
    generations.mkdir(parents=True)
    for step in sorted({record["step"] for record in records}):
        with (generations / f"{step}.jsonl").open("w", encoding="utf-8") as output:
            for record in records:
                if record["step"] == step:
                    record["uid"] = f"{args.submission_id}:{step}:{record['prompt_row']}:{record['sample_index']}"
                    record["submission_id"] = args.submission_id
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")

    context = (
        args.context
        or subprocess.check_output(
            ["kubectl", "config", "current-context"],
            text=True,
            timeout=30,
        ).strip()
    )
    command = ["kubectl", "--context", context, "--namespace", args.namespace, "--request-timeout=30s"]
    driver_ip = urlparse(job.get("driver_agent_http_address") or "").hostname
    pods = kubectl_json(command, "get", "pods")["items"]
    candidates = [pod for pod in pods if driver_ip and pod.get("status", {}).get("podIP") == driver_ip]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot find the execution pod at {driver_ip}. Job log and JSONL saved in {run}")
    pod = candidates[0]
    pod_name = pod["metadata"]["name"]
    containers = [c["name"] for c in pod["spec"]["containers"] if c["name"] in {"ray-worker", "ray-head"}]
    if not args.container and len(containers) != 1:
        raise RuntimeError("Cannot identify the Ray container; specify --container.")
    container = args.container or containers[0]
    container_spec = next(c for c in pod["spec"]["containers"] if c["name"] == container)
    container_status = next(
        (c for c in pod["status"].get("containerStatuses", []) if c["name"] == container),
        {},
    )
    write_json(
        run / "collection-source.json",
        {
            "collected_at_utc": collected.isoformat(),
            "ray_address": args.address,
            "context": context,
            "namespace": args.namespace,
            "pod": pod_name,
            "pod_uid": pod["metadata"]["uid"],
            "pod_ip": driver_ip,
            "node": pod["spec"].get("nodeName"),
            "container": container,
            "container_image": container_spec["image"],
            "container_image_id": container_status.get("imageID"),
            "scope": "This probe's Ray driver/supervisor logs, retained submitted scripts and matching probe log.",
            "jsonl_source": "Reconstructed from PROMPT/OUTPUT records in ray-job.log; input is the logged user prompt.",
        },
    )
    launcher_args = shlex.split(job["entrypoint"])
    launchers = [Path(arg).name for arg in launcher_args if arg.endswith(".sh")]
    if len(launchers) != 1 or launchers[0] not in {
        "run_hybrid_pool_gsm8k_prefix_cache.sh",
        "run_hybrid_pool_gsm8k_prefix_on.sh",
    }:
        raise RuntimeError(f"Unrecognized probe launcher: {job['entrypoint']}")
    download = download_pod_files(command, pod_name, container, job, {**config, "launcher": launchers[0]}, run)
    probe_log = run / f"prefix_{mode}.log"
    (run / f"{name}.log").write_bytes((probe_log if probe_log.is_file() else run / "ray-job.log").read_bytes())
    write_json(
        run / "download-manifest.json",
        {
            "source_pod": pod_name,
            "archive": "downloaded-artifacts.tar.gz",
            **download,
        },
    )
    files = []
    for path in sorted(run.rglob("*")):
        if path.is_file():
            payload = path.read_bytes()
            files.append(
                {
                    "path": str(path.relative_to(run)),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    write_json(
        run / f"{name}_manifest.json",
        {
            "submission_id": args.submission_id,
            "collected_at_utc": collected.isoformat(),
            "files": files,
            "summary": summary,
            "warnings": download["warnings"],
        },
    )
    for warning in download["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Saved {len(records)} prompt/response records to {generations}", flush=True)
    print(f"Artifacts: {run}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as error:
        sys.exit(f"Error: {error}")
