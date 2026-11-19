#!/usr/bin/env bash
# ==============================================================================
# Fleet Gold Virtual Environment Synchronization Wrapper
# Target: Hermes Agent v0.23.1-fleet-gold
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/sync_fleet_gold_venv.py"
REQUIREMENTS_FILE="${RUNTIME_ROOT}/fleet-gold-requirements.txt"
DEFAULT_VENV="/home/srvadm/.hermes/installs/329a495c03e775a4/environments/de3ebfb122c748d6b2902c1bf394a838/venv"
TARGET_VENV="${1:-$DEFAULT_VENV}"
OUTPUT_MANIFEST="${2:-/home/srvadm/.hermes/tasks/task_20261116_fleet_gold_requirements/evidence/installed_manifest.json}"

echo "[*] Fleet Gold Venv Sync initiated"
echo "[*] Runtime root: ${RUNTIME_ROOT}"
echo "[*] Target venv:  ${TARGET_VENV}"
echo "[*] Requirements: ${REQUIREMENTS_FILE}"
echo "[*] Manifest out: ${OUTPUT_MANIFEST}"

python3 "${PYTHON_SCRIPT}" \
    --venv "${TARGET_VENV}" \
    --requirements "${REQUIREMENTS_FILE}" \
    --output-manifest "${OUTPUT_MANIFEST}"

echo "[+] Sync wrapper completed successfully."
