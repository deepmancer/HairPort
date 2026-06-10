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

# ── PEAR (head/body fitting backend) ────────────────────────────────
# PEAR is a tracked git submodule; init it (or update to the pinned commit).
echo ">> Initializing PEAR submodule..."
git -C "${REPO_ROOT}" submodule update --init --recursive modules/PEAR

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

# ── HairPort + PEAR data assets ─────────────────────────────────────────────────
# Pipeline landmark embeddings (assets/landmarks) and PEAR runtime models
# (modules/PEAR/assets/{FLAME,SMPLX}) ship together in hairport_data.zip on the
# Hugging Face Hub. By downloading you accept the SMPL-X / FLAME license terms.
echo ">> Downloading HairPort + PEAR data assets..."
if [[ ! -f "${MODULES_DIR}/PEAR/assets/SMPLX/SMPLX_NEUTRAL_2020.npz" ]]; then
    python "${REPO_ROOT}/scripts/download_assets.py"
else
    echo ">> Data assets already present, skipping."
fi

# ── pytorch3d (required by PEAR's renderer / mesh ops) ──────────────────────────
# Build against the active torch's CUDA toolkit and a supported host compiler:
echo ">> Installing pytorch3d (CUDA build)..."
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" \
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo ""
echo ">> Installing CodeFormer dependencies..."
if [[ -f "${MODULES_DIR}/CodeFormer/requirements.txt" ]]; then
    pip install -r "${MODULES_DIR}/CodeFormer/requirements.txt"
fi

echo ""
echo "Done! Modules directory:"
ls -1 "${MODULES_DIR}"
