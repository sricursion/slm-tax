#!/usr/bin/env bash
# Bootstrap a GPU node (ROCm or CUDA) for slm-tax training.
# Usage: bash scripts/setup_node.sh [rocm|cuda]
set -euo pipefail
FLAVOR="${1:-rocm}"

cd "$(dirname "$0")/.."

echo "=== python env ==="
python3 -m pip install --upgrade pip -q

if [ "$FLAVOR" = "rocm" ]; then
    # MI300X: ROCm PyTorch wheels
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 -q || \
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.1 -q
else
    python3 -m pip install torch -q
fi
python3 -m pip install -q transformers trl peft datasets accelerate

echo "=== gpu check ==="
python3 - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda/hip available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("vram GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)
EOF

echo "=== benchmark clone ==="
[ -d external/tax-calc-bench ] || git clone --depth 1 https://github.com/column-tax/tax-calc-bench external/tax-calc-bench

echo "=== datasets (seeded, deterministic) ==="
[ -f data/train_20k.jsonl ] || python3 scripts/generate_data.py --n 20000 --seed 1000 --out data/train_20k.jsonl
[ -f data/val_2k.jsonl ] || python3 scripts/generate_data.py --n 2000 --seed 5000000 --out data/val_2k.jsonl

echo "=== setup complete ==="
