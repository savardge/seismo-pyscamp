#!/bin/bash
#SBATCH --job-name=pyscamp-build
#SBATCH --partition=glados-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#
# Create or update the `seismo-pyscamp` conda environment on Glados (with its
# CUDA 12.0 toolchain) and build pyscamp from GitHub HEAD *with* the CUDA
# backend, then verify
# that pyscamp actually sees the GPU. Runs on a GPU node, not the login node:
# the compile is heavy and the verification needs a card.
#
# Usage (from the repo on Glados):
#   cd ~/seismo-pyscamp/scripts_frs/glados && mkdir -p logs && sbatch build_pyscamp.slurm
#
# Environment overrides:
#   SEISCAMP_REPO   path of the seismo-pyscamp checkout (default ~/seismo-pyscamp)
#   SCAMP_REF       SCAMP git ref to build (default: master HEAD)
set -euo pipefail

REPO="${SEISCAMP_REPO:-$HOME/seismo-pyscamp}"
ENV_NAME=seismo-pyscamp
SCAMP_REF="${SCAMP_REF:-master}"

echo "=== $(date) on $(hostname) | repo $REPO | env $ENV_NAME ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# --- conda environment (includes the CUDA 12.0 toolchain) --------------------
# CONDA_OVERRIDE_CUDA tells the solver which CUDA the driver supports in case
# it cannot detect it; `conda env update` also upgrades an env created from an
# earlier version of the file.
module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_OVERRIDE_CUDA=12.0
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "conda env $ENV_NAME exists, updating it from environment-glados.yml"
    conda env update -n "$ENV_NAME" -f "$REPO/environment-glados.yml"
else
    echo "creating conda env $ENV_NAME from environment-glados.yml"
    conda env create -f "$REPO/environment-glados.yml"
fi
# conda's compiler activation scripts read variables that may be unset, so
# the unbound-variable check is lifted around the activation only.
set +u
conda activate "$ENV_NAME"
set -u
# gxx_linux-64 exports CC/CXX to the conda gcc 12 on activation; point nvcc at it.
export CUDAHOSTCXX="$CXX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
echo "python: $(which python) ($(python --version))"
echo "cmake:  $(cmake --version | head -1)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo "host:   $($CXX --version | head -1)"

# --- build pyscamp -----------------------------------------------------------
# FORCE_CUDA=1 makes the build fail loudly instead of silently falling back to
# CPU. CMAKE_CUDA_ARCHITECTURES=61 targets the GTX 1080 Ti (Pascal GP102) only;
# without it SCAMP compiles every architecture its CUDA version supports (16 for
# CUDA 11.x), which is slow and has crashed nvcc elsewhere.
export FORCE_CUDA=1
export PYSCAMP_ADD_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=61 \
  -DCMAKE_CUDA_COMPILER=$CUDACXX -DCMAKE_CUDA_HOST_COMPILER=$CXX \
  -DCUDAToolkit_ROOT=$CONDA_PREFIX/targets/x86_64-linux"
pip install --no-cache-dir --force-reinstall --no-deps \
    "git+https://github.com/zpzim/SCAMP.git@${SCAMP_REF}"

# --- verify ------------------------------------------------------------------
python - <<'PY'
import time, numpy as np, pyscamp
print("pyscamp", pyscamp.__version__, "| gpu_supported:", pyscamp.gpu_supported())
assert pyscamp.gpu_supported(), "pyscamp was built without CUDA"
rng = np.random.default_rng(0)
x = rng.standard_normal(2_000_000)
for prec in ("single", "double"):
    t = time.time()
    mp, _ = pyscamp.selfjoin(x, 50, pearson=True, precision=prec, gpus=[0])
    print(f"2e6-sample self-join, {prec}: {time.time()-t:6.1f} s, max CC {mp.max():.4f}")
import seiscamp
print("seiscamp import OK from", seiscamp.__file__)
PY
echo "=== build finished $(date) ==="
