#!/usr/bin/env python3
import argparse
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""


def first_line(text):
    if not text:
        return ""
    return text.splitlines()[0].strip()


def build_run_id(git_sha):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = git_sha[:8] if git_sha else "nogitsha"
    return f"{ts}_{short}"


def main():
    parser = argparse.ArgumentParser(description="Collect run-level environment manifest.")
    parser.add_argument("--results-root", required=True, help="Path to data/results")
    parser.add_argument("--run-id", default="", help="Optional run id override")
    parser.add_argument("--emit-shell", action="store_true", help="Emit shell assignments")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    git_sha = run_cmd(["git", "rev-parse", "HEAD"])
    run_id = args.run_id or build_run_id(git_sha)

    gpu_name = first_line(run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]))
    gpu_uuid = first_line(run_cmd(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]))
    driver_version = first_line(run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]))
    cuda_version = first_line(run_cmd(["nvcc", "--version"]))
    if "release" in cuda_version:
        idx = cuda_version.find("release")
        cuda_version = cuda_version[idx:].replace(",", "").strip()

    build_type = os.environ.get("CMAKE_BUILD_TYPE", "")
    cuda_arch = os.environ.get("CMAKE_CUDA_ARCHITECTURES", "")

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "git_sha": git_sha,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "build_type": build_type,
        "cuda_arch": cuda_arch,
    }

    run_dir = results_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    if args.emit_shell:
        print(f'RUN_ID="{run_id}"')
        print(f'MANIFEST_PATH="{manifest_path}"')
        print(f'RUN_DIR="{run_dir}"')
    else:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
