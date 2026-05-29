#!/usr/bin/env bash
# scripts/setup_submodules.sh — Clone / initialise external dependencies
# into the modules/ directory.
#
# Usage:
#   bash scripts/setup_submodules.sh
# The module revisions below are intentionally immutable for paper inference.
set -euo pipefail

eval "$(conda shell.bash hook)"


REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODULES_DIR="${REPO_ROOT}/modules"
mkdir -p "${MODULES_DIR}"

clone_at_commit() {
    local name="$1" url="$2" commit="$3"
    local dest="${MODULES_DIR}/${name}"
    if [[ -d "${dest}/.git" ]]; then
        echo ">> Verifying pinned ${name} revision..."
    else
        echo ">> Cloning ${name}..."
        git clone "${url}" "${dest}"
    fi
    git -C "${dest}" fetch origin "${commit}" --depth 1
    git -C "${dest}" checkout --detach "${commit}"
    test "$(git -C "${dest}" rev-parse HEAD)" = "${commit}"
}

# ── CodeFormer ──────────────────────────────────────────────────────
clone_at_commit "CodeFormer" "https://github.com/deepmancer/CodeFormer.git" \
    "8180c3e9000fbd9d63d22e0df2bb5f991e5a2d01"

# ── MV-Adapter ─────────────────────────────────────────────────────
clone_at_commit "MV-Adapter" "https://github.com/deepmancer/MV-Adapter.git" \
    "849c1a2babdc76c01cbe6158493d750088a3f250"

# ── SHeaP ───────────────────────────────────────────
clone_at_commit "SHeaP" "https://github.com/deepmancer/SHeaP.git" \
    "cde7b7a9f0ba28e8250d6fc7100fd985be483134"

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


cd "${REPO_ROOT}"

# ── SHeaP downloads ──────────────────────────────────────────────────────────
cd "${MODULES_DIR}/SHeaP"
conda activate hairport && python convert_flame.py --flame_base_dir "${REPO_ROOT}/assets/"
cd "${REPO_ROOT}"

# ── Install editable packages where needed ─────────────────────────
echo ""
echo ">> Installing SHeaP in editable mode..."
pip install -e "${MODULES_DIR}/SHeaP"

echo ""
echo ">> Installing CodeFormer dependencies..."
if [[ -f "${MODULES_DIR}/CodeFormer/requirements.txt" ]]; then
    pip install -r "${MODULES_DIR}/CodeFormer/requirements.txt"
fi

echo ""
echo "Done! Modules directory:"
ls -1 "${MODULES_DIR}"
