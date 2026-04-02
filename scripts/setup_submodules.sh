#!/usr/bin/env bash
# scripts/setup_submodules.sh — Clone / initialise external dependencies
# into the modules/ directory.
#
# Usage:
#   bash scripts/setup_submodules.sh        # fresh clone
#   bash scripts/setup_submodules.sh --pull  # pull latest on every repo
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODULES_DIR="${REPO_ROOT}/modules"
PULL="${1:-}"

mkdir -p "${MODULES_DIR}"

clone_or_update() {
    local name="$1" url="$2" branch="${3:-}"
    local dest="${MODULES_DIR}/${name}"
    if [[ -d "${dest}/.git" ]]; then
        if [[ "${PULL}" == "--pull" ]]; then
            echo ">> Pulling ${name}..."
            git -C "${dest}" pull --ff-only
        else
            echo ">> ${name} already cloned, skipping. Use --pull to update."
        fi
    else
        echo ">> Cloning ${name}..."
        if [[ -n "${branch}" ]]; then
            git clone --depth 1 --branch "${branch}" "${url}" "${dest}"
        else
            git clone --depth 1 "${url}" "${dest}"
        fi
    fi
}

# ── CodeFormer ──────────────────────────────────────────────────────
clone_or_update "CodeFormer" "https://github.com/sczhou/CodeFormer.git"

# ── MV-Adapter ─────────────────────────────────────────────────────
clone_or_update "MV-Adapter" "https://github.com/huanngzh/MV-Adapter.git"

# ── Hi3DGen ────────────────────────────────────────────────────────
clone_or_update "Hi3DGen" "https://github.com/Stable-X/Hi3DGen.git"

# ── SHeaP ──────────────────────────────────────────────────────────
clone_or_update "SHeaP" "https://github.com/deepmancer/SHeaP.git"

# ── Install editable packages where needed ─────────────────────────
echo ""
echo ">> Installing SHeaP in editable mode..."
pip install -e "${MODULES_DIR}/SHeaP" 2>/dev/null || echo "   (SHeaP install skipped — run manually if needed)"

echo ""
echo ">> Installing CodeFormer dependencies..."
if [[ -f "${MODULES_DIR}/CodeFormer/requirements.txt" ]]; then
    pip install -r "${MODULES_DIR}/CodeFormer/requirements.txt" 2>/dev/null || echo "   (some deps may already be installed)"
fi

echo ""
echo "Done! Modules directory:"
ls -1 "${MODULES_DIR}"
