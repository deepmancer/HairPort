#!/usr/bin/env bash
# scripts/setup_submodules.sh — Clone / initialise external dependencies
# into the modules/ directory.
#
# Usage:
#   bash scripts/setup_submodules.sh        # fresh clone
#   bash scripts/setup_submodules.sh --pull  # pull latest on every repo
set -euo pipefail

eval "$(conda shell.bash hook)"


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
clone_or_update "CodeFormer" "git@github.com:deepmancer/CodeFormer.git"

# ── MV-Adapter ─────────────────────────────────────────────────────
clone_or_update "MV-Adapter" "git@github.com:deepmancer/MV-Adapter.git"
# ── MV-Adapter downloads ───────────────────────────────────────────
echo ">> Setting up MV-Adapter checkpoints..."
mkdir -p "${MODULES_DIR}/MV-Adapter/checkpoints"
[[ -f "${MODULES_DIR}/MV-Adapter/checkpoints/RealESRGAN_x2plus.pth" ]] || wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O "${MODULES_DIR}/MV-Adapter/checkpoints/RealESRGAN_x2plus.pth"
[[ -f "${MODULES_DIR}/MV-Adapter/checkpoints/big-lama.pt" ]] || wget https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt -O "${MODULES_DIR}/MV-Adapter/checkpoints/big-lama.pt"
# ── MV-Adapter LoRAs ───────────────────────────────────────────────
echo ">> Downloading MV-Adapter LoRAs..."
cd "${MODULES_DIR}/MV-Adapter"
if [[ ! -d "loras" ]]; then
    [[ -f "loras.zip" ]] || gdown 1zmEPR-w7PFaboZLrJ6biT3Vy9YGUF8tg -O loras.zip
    unzip -q loras.zip -d loras
    rm -f loras.zip
else
    echo ">> MV-Adapter loras/ already exists, skipping."
fi
cd "${REPO_ROOT}"


# ── Hi3DGen ────────────────────────────────────────────────────────
clone_or_update "Hi3DGen" "git@github.com:deepmancer/Hi3DGen.git"
cd "${MODULES_DIR}/Hi3DGen"
rm -rf NiRNE 
git clone git@github.com:lzt02/NiRNE.git
conda activate hairport && python download_nirne_weights.py
cd "${REPO_ROOT}"

# ── SHeaP ──────────────────────────────────────────────────────────
clone_or_update "SHeaP" "git@github.com:deepmancer/SHeaP.git"
cd "${MODULES_DIR}/SHeaP"
conda activate hairport && python convert_flame.py --flame_base_dir "${REPO_ROOT}/assets/"
cd "${REPO_ROOT}"

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
