#!/usr/bin/env bash
# Setup script for rl-irs-tax-code on Apple M4 Max
# Usage: bash scripts/setup.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== rl-irs-tax-code Environment Setup ==="
echo "Project dir: $PROJECT_DIR"
echo "Platform: Apple Silicon (arm64)"
echo ""

# Create venv using Homebrew Python 3.14 explicitly
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment with Python 3.14..."
  /opt/homebrew/bin/python3 -m venv "$VENV_DIR"
fi

# Activate using python3.14 (NOTE: default python3 in .venv points to uv's 3.12)
PYTHON="$VENV_DIR/bin/python3.14"
PIP="$VENV_DIR/bin/pip"

echo "Upgrading pip..."
"$PYTHON" -m pip install --upgrade pip

echo "Installing requirements..."
"$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"

echo "Installing bitsandbytes (supports Apple Silicon M4)..."
"$PYTHON" -m pip install bitsandbytes

echo "Installing MLX (Apple Silicon native ML framework)..."
"$PYTHON" -m pip install mlx mlx-lm

echo ""
echo "=== Setup Complete ==="
echo ""
echo "IMPORTANT: To activate the environment, use:"
echo "  source .venv/bin/activate"
echo "  # Then use 'python3.14' not 'python3' to get the right interpreter"
echo ""
echo "Or use directly:"
echo "  .venv/bin/python3.14 your_script.py"
echo ""
echo "Verifying installation..."
"$PYTHON" -c "
import torch, transformers, trl, peft, datasets, accelerate
import mlx.core as mx
import bitsandbytes as bnb
print('torch:', torch.__version__, '| MPS:', torch.backends.mps.is_available())
print('transformers:', transformers.__version__)
print('trl:', trl.__version__)
print('peft:', peft.__version__)
print('datasets:', datasets.__version__)
print('accelerate:', accelerate.__version__)
print('mlx device:', mx.default_device())
print('bitsandbytes:', bnb.__version__)
print('ALL OK')
"
