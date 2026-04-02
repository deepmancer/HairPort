#!/usr/bin/env bash
# install.sh — Full installation script for HairPort
# Usage: bash scripts/install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Prerequisites ──────────────────────────────────────────────────
npm install -g gltfpack
git lfs install

# ── Clone ──────────────────────────────────────────────────────────
if [[ ! -d "HairPort/.git" ]]; then
    git clone git@github.com:deepmancer/HairPort.git
else
    echo ">> HairPort already cloned, skipping."
fi

# ── Blender ────────────────────────────────────────────────────────
if ! command -v blender &>/dev/null; then
    BLENDER_ARCHIVE="blender-4.0.2-linux-x64.tar.xz"
    if [[ ! -f "${BLENDER_ARCHIVE}" ]]; then
        wget https://download.blender.org/release/Blender4.0/${BLENDER_ARCHIVE}
    fi
    tar -xf "${BLENDER_ARCHIVE}"
    sudo mv blender-4.0.2-linux-x64 /usr/local/blender-4.0.2
    sudo ln -sf /usr/local/blender-4.0.2/blender /usr/local/bin/blender
fi

# ── Conda environment ─────────────────────────────────────────────
eval "$(conda shell.bash hook)"
if ! conda info --envs | grep -q "^hairport "; then
    conda create -n hairport python=3.10 -y
else
    echo ">> Conda env 'hairport' already exists, skipping."
fi
conda activate hairport

# ── PyTorch (CUDA 12.8) ───────────────────────────────────────────
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Core Python tooling ───────────────────────────────────────────
python -m pip install -U pip setuptools wheel packaging

# ── Python dependencies ───────────────────────────────────────────
pip install \
    accelerate \
    safetensors \
    Pillow \
    scipy \
    opencv-python \
    mediapipe==0.10.20 \
    roma \
    huggingface_hub \
    scikit-image \
    imageio \
    plotly \
    dash \
    psutil \
    ninja \
    ftfy \
    regex \
    nltk \
    matplotlib \
    open3d \
    wandb \
    pandas \
    gdown \
    tensorboard \
    onnxruntime-gpu

pip install trimesh pyrender

# ── Diffusers, Transformers, Flash Attention (from source) ────────
PACKAGES_DIR="${REPO_ROOT}/local_packages"
mkdir -p "${PACKAGES_DIR}"
pushd "${PACKAGES_DIR}"

[[ -d diffusers/.git ]]        || git clone --recursive git@github.com:huggingface/diffusers.git
[[ -d transformers/.git ]]     || git clone --recursive git@github.com:huggingface/transformers.git

pip install diffusers/ transformers/ \
    git+https://github.com/huggingface/peft \
    sentence-transformers

MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="9.0" pip install flash-attn --no-build-isolation

popd

# ── easy_dwpose (from source) ────────────────────────────────────
[[ -d "${PACKAGES_DIR}/easy_dwpose/.git" ]] || \
    git clone git@github.com:deepmancer/easy_dwpose.git "${PACKAGES_DIR}/easy_dwpose"
pip install -e "${PACKAGES_DIR}/easy_dwpose"

# ── BEN2, rembg, mediapipe ────────────────────────────────────────
pip install git+https://github.com/PramaLLC/BEN2.git "rembg[gpu]"

# ── Chumpy (for SHeaP) ───────────────────────────────────────────────
pip install chumpy --no-build-isolation                                                           

# ── Jupyter ────────────────────────────────────────────────────────
conda install -c conda-forge jupyter notebook jupyterlab \
    nb_conda_kernels nb_conda ipykernel -y
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# ── PyTorch3D & nvdiffrast ────────────────────────────────────────
pip install fvcore iopath
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0" \
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0" \
    pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"

# ── Blender Python module ─────────────────────────────────────────
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
pip install fake-bpy-module

# ── Pin numpy last (some packages may override it) ────────────────
pip install numpy==1.26.4

# ── Setup submodules (CodeFormer, MV-Adapter, Hi3DGen, SHeaP) ────
# bash "${SCRIPT_DIR}/setup_submodules.sh"

# ── Install HairPort itself ───────────────────────────────────────
cd "${REPO_ROOT}"
pip install -e .

echo ""
echo "HairPort installation complete!"
