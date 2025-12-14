#!/bin/bash
# =============================================================================
# TPU Environment Setup Script for GRPO Gemma Training
# =============================================================================
# Usage: bash setup_tpu.sh
# =============================================================================

set -e

echo "=============================================="
echo "  TPU Environment Setup for GRPO Training"
echo "=============================================="

# Step 1: System update
echo "[1/6] Updating system packages..."
sudo apt-get update -qq

# Step 2: Install Python dependencies
echo "[2/6] Installing Python packages..."
pip install --upgrade pip

# Step 3: Install JAX for TPU
echo "[3/6] Installing JAX for TPU..."
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Step 4: Install ML libraries
echo "[4/6] Installing ML libraries..."
pip install \
    flax \
    optax \
    orbax-checkpoint \
    grain-nightly \
    humanize \
    kagglehub \
    tensorflow-datasets \
    tqdm \
    qwix \
    tunix

# Step 5: Install Jupyter
echo "[5/6] Installing Jupyter..."
pip install jupyter jupyterlab ipywidgets

# Step 6: Verify TPU
echo "[6/6] Verifying TPU connection..."
python3 -c "
import jax
devices = jax.devices()
print(f'Available devices: {len(devices)}')
for d in devices:
    print(f'  - {d}')
if any('TPU' in str(d) for d in devices):
    print('TPU detected successfully!')
else:
    print('WARNING: No TPU detected. Please check your TPU VM setup.')
"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Set your Kaggle credentials:"
echo "     export KAGGLE_USERNAME='your_username'"
echo "     export KAGGLE_KEY='your_key'"
echo ""
echo "  2. (Optional) Set Wandb API key:"
echo "     export WANDB_API_KEY='your_key'"
echo ""
echo "  3. Start Jupyter Lab:"
echo "     jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo ""
echo "  4. Open grpo-demo-local.ipynb and run!"
echo ""

