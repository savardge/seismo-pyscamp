#!/bin/bash
#SBATCH --job-name=pyscamp-build
#SBATCH --partition=glados-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#
# Create the `seismo-pyscamp` conda environment on Glados (if it does not exist
# yet) and build pyscamp from GitHub HEAD *with* the CUDA backend, then verify
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

# --- toolchain ---------------------------------------------------------------
# cuda/11.2 is the newest CUDA module on Glados (SCAMP needs >= 11.0). Its nvcc
# refuses host compilers newer than gcc 10, which is why gcc/10.4.0 is loaded
# and why environment-glados.yml carries no conda compiler.
module purge
module load cuda/11.2 gcc/10.4.0
export CC=gcc CXX=g++ CUDAHOSTCXX=g++
echo "nvcc: $(nvcc --version | tail -1)"
echo "gcc:  $(gcc --version | head -1)"

# --- conda environment -------------------------------------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "conda env $ENV_NAME exists, reusing it"
else
    echo "creating conda env $ENV_NAME from environment-glados.yml"
    conda env create -f "$REPO/environment-glados.yml"
fi
conda activate "$ENV_NAME"
echo "python: $(which python) ($(python --version))"
echo "cmake:  $(cmake --version | head -1)"

# --- build pyscamp -----------------------------------------------------------
# FORCE_CUDA=1 makes the build fail loudly instead of silently falling back to
# CPU. CMAKE_CUDA_ARCHITECTURES=61 targets the GTX 1080 Ti (Pascal GP102) only;
# without it SCAMP compiles every architecture its CUDA version supports (16 for
# CUDA 11.x), which is slow and has crashed nvcc elsewhere.
export FORCE_CUDA=1
export PYSCAMP_ADD_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=61"
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
