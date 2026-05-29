#!/usr/bin/env bash
# Export the exact installed inference environment after successful preflight/smoke testing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_PATH="${1:-${REPO_ROOT}/requirements.inference.lock.txt}"

cd "${REPO_ROOT}"
python -m hairport.preflight
python -m pip freeze --all | LC_ALL=C sort > "${OUTPUT_PATH}"

echo "Wrote exact Python environment lock: ${OUTPUT_PATH}"
echo "Pinned external module revisions:"
git -C "${REPO_ROOT}/modules/CodeFormer" rev-parse HEAD
git -C "${REPO_ROOT}/modules/MV-Adapter" rev-parse HEAD
git -C "${REPO_ROOT}/modules/SHeaP" rev-parse HEAD
