#!/usr/bin/env python3
"""
Sync Fleet Gold Virtual Environment (SSOT Parity Tool)
=====================================================
Target Runtimes: Hermes Agent v0.23.1-fleet-gold (ubsrvagent & fleet nodes)
Enforces:
  1. Strict blacklist enforcement (torch, torchvision, torchaudio, triton, nvidia-*, cuda-*)
  2. Target venv & Python version validation (Python >= 3.10)
  3. Idempotent dependency synchronization via `uv pip install`
  4. Machine-readable manifest generation
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

FORBIDDEN_PATTERNS = [
    re.compile(r"^torch($|[<>=!~@])", re.IGNORECASE),
    re.compile(r"^torchvision($|[<>=!~@])", re.IGNORECASE),
    re.compile(r"^torchaudio($|[<>=!~@])", re.IGNORECASE),
    re.compile(r"^triton($|[<>=!~@])", re.IGNORECASE),
    re.compile(r"^nvidia[-_].*", re.IGNORECASE),
    re.compile(r"^cuda[-_].*", re.IGNORECASE),
]

DEFAULT_REQUIREMENTS_PATH = Path("/mnt/data/runtimes/hermes-agent-runtime-v0.23.1-fleet-gold/fleet-gold-requirements.txt")
DEFAULT_TARGET_VENV = Path("/home/srvadm/.hermes/installs/329a495c03e775a4/environments/de3ebfb122c748d6b2902c1bf394a838/venv")


def find_python_executable(target: Path) -> Path:
    if target.is_file() and os.access(target, os.X_OK):
        return target
    # If directory, look for bin/python3 or bin/python
    candidates = [
        target / "bin" / "python3",
        target / "bin" / "python",
        target / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(f"Could not locate Python executable in target path: {target}")


def validate_python_version(python_bin: Path) -> tuple[int, ...]:
    cmd = [str(python_bin), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        ver_str = proc.stdout.strip()
        parts = tuple(int(p) for p in ver_str.split("."))
        if parts < (3, 10):
            raise RuntimeError(f"Python version must be >= 3.10, found: {ver_str}")
        return parts
    except Exception as e:
        raise RuntimeError(f"Failed to validate python version for {python_bin}: {e}") from e


def audit_requirements_blacklist(req_path: Path) -> list[str]:
    violations = []
    if not req_path.is_file():
        raise FileNotFoundError(f"Requirements file not found: {req_path}")

    with open(req_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            # Extract package name (first token before version specifiers)
            pkg_name = re.split(r"[<>=!~@;\s]", clean)[0].strip()
            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(pkg_name):
                    violations.append(f"Line {line_num}: {clean} matches forbidden pattern {pattern.pattern}")
    return violations


def audit_installed_blacklist(python_bin: Path) -> list[str]:
    violations = []
    cmd = ["uv", "pip", "list", "--python", str(python_bin), "--format", "json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
        installed = json.loads(proc.stdout)
        for item in installed:
            name = item.get("name", "")
            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(name):
                    violations.append(f"Installed package '{name}' ({item.get('version')}) violates blacklist pattern {pattern.pattern}")
    except Exception as e:
        violations.append(f"Failed to audit installed packages: {e}")
    return violations


def run_sync(python_bin: Path, req_path: Path) -> dict:
    cmd = ["uv", "pip", "install", "-r", str(req_path), "--python", str(python_bin)]
    start_time = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    duration = time.time() - start_time
    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_sec": round(duration, 3),
        "success": proc.returncode == 0,
    }


def export_manifest(python_bin: Path, output_file: Path | None = None) -> dict:
    cmd = ["uv", "pip", "list", "--python", str(python_bin), "--format", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
    packages = json.loads(proc.stdout)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "python_executable": str(python_bin),
        "package_count": len(packages),
        "packages": packages,
    }

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Synchronize Fleet Gold venv dependencies safely and idempotently.")
    parser.add_argument("--venv", type=Path, default=DEFAULT_TARGET_VENV, help="Target virtualenv path or python binary")
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS_PATH, help="Path to fleet-gold-requirements.txt")
    parser.add_argument("--output-manifest", type=Path, default=None, help="Path to write machine-readable manifest JSON")
    parser.add_argument("--preflight-only", action="store_true", help="Perform pre-flight checks and blacklist audit without installing")
    args = parser.parse_args()

    print(f"[*] Locating Python binary in target: {args.venv}...")
    python_bin = find_python_executable(args.venv)
    print(f"[+] Target Python: {python_bin}")

    ver = validate_python_version(python_bin)
    print(f"[+] Target Python Version validated: {ver[0]}.{ver[1]}.{ver[2]} (>= 3.10)")

    print(f"[*] Auditing requirements file for blacklisted packages: {args.requirements}...")
    req_violations = audit_requirements_blacklist(args.requirements)
    if req_violations:
        print("[!] FATAL: Blacklist violations found in requirements file:")
        for v in req_violations:
            print(f"    - {v}")
        sys.exit(2)
    print("[+] Requirements file is 100% clean of blacklisted packages.")

    print(f"[*] Auditing target venv for existing blacklisted packages...")
    inst_violations = audit_installed_blacklist(python_bin)
    if inst_violations:
        print("[!] FATAL: Target virtualenv already contains blacklisted packages:")
        for v in inst_violations:
            print(f"    - {v}")
        sys.exit(3)
    print("[+] Target virtualenv is 100% clean of blacklisted packages.")

    if args.preflight_only:
        print("[+] Preflight check completed successfully.")
        sys.exit(0)

    print(f"[*] Starting synchronization via `uv pip install`...")
    sync_res = run_sync(python_bin, args.requirements)
    if not sync_res["success"]:
        print(f"[!] Synchronization FAILED (exit code {sync_res['exit_code']}):")
        print(sync_res["stderr"])
        sys.exit(sync_res["exit_code"] or 1)

    print(f"[+] Synchronization SUCCEEDED in {sync_res['duration_sec']}s.")
    if sync_res["stdout"]:
        print(sync_res["stdout"].strip())

    print(f"[*] Generating installed packages manifest...")
    manifest = export_manifest(python_bin, args.output_manifest)
    print(f"[+] Manifest created: {manifest['package_count']} installed packages.")
    if args.output_manifest:
        print(f"[+] Saved manifest to {args.output_manifest}")

    print("[SUCCESS] Fleet Gold venv synchronization complete.")


if __name__ == "__main__":
    main()
