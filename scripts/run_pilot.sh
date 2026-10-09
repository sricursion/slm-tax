#!/usr/bin/env bash
# One-shot pilot: setup -> train -> eval, all detachable. Artifacts stay small
# and get collected under /root/slm/artifacts/ for immediate scp off-node.
# Usage (on the GPU node): nohup bash scripts/run_pilot.sh > /root/pilot_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="$HOME/venv"
PY="$VENV/bin/python"

echo "=== [1/4] setup ==="
if ! $PY -c "import torch" 2>/dev/null; then
    sudo apt-get install -y -qq python3-venv python3-pip 2>/dev/null \
        || apt-get install -y -qq python3-venv python3-pip
    python3 -m venv "$VENV"
    $PY -m pip install -U pip -q
    if command -v nvidia-smi >/dev/null 2>&1; then
        IDX_LIST="cu124 cu121"
    else
        IDX_LIST="rocm7.0 rocm6.4"
    fi
    for idx in $IDX_LIST; do
        $PY -m pip install torch --index-url https://download.pytorch.org/whl/$idx -q && \
        $PY -c "import torch; assert torch.cuda.is_available()" && break || $PY -m pip uninstall -y torch -q
    done
fi
$PY -m pip install -q transformers trl peft datasets accelerate lxml
$PY -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print(torch.cuda.get_device_name(0))"
[ -d external/tax-calc-bench ] || git clone --depth 1 https://github.com/column-tax/tax-calc-bench external/tax-calc-bench
[ -f data/train_20k_v7.jsonl ] || $PY scripts/generate_data.py --n 20000 --seed 1000 --out data/train_20k_v7.jsonl
[ -f data/val_2k_v7.jsonl ] || $PY scripts/generate_data.py --n 2000 --seed 5000000 --out data/val_2k_v7.jsonl

echo "=== preflight (fail fast before burning train time) ==="
$PY scripts/preflight.py

echo "=== [2/4] train ==="
# GPU utilization trace (30s samples) so usage is auditable after the fact
mkdir -p artifacts
[ -s artifacts/gpu_trace.csv ] || bash scripts/gpu_sample.sh --header > artifacts/gpu_trace.csv
( bash scripts/gpu_sample.sh --loop 15 >> artifacts/gpu_trace.csv ) &
GPU_TRACE_PID=$!
trap "kill $GPU_TRACE_PID 2>/dev/null || true" EXIT
BSZ="${BSZ:-$($PY -c "import torch; print(8 if torch.cuda.get_device_properties(0).total_memory/1e9 > 70 else 2)")}"
ACCUM=$((32 / BSZ))
echo "batch size $BSZ x accum $ACCUM"
$PY scripts/train_sft.py --model Qwen/Qwen3-1.7B --data data/train_20k_v7.jsonl \
    --lora --lr 2e-4 --epochs 1 --save-steps 250 --seq-len 10240 --bsz "$BSZ" --accum "$ACCUM" \
    --out runs/pilot7-1.7b

echo "=== [3/4] eval (saves generations for post-run taxonomy) ==="
$PY scripts/eval_model.py --model runs/pilot7-1.7b --benchmark \
    --synthetic data/val_2k_v7.jsonl --limit 200 --batch-size 8 --max-new-tokens 3584 \
    --save-texts artifacts/pilot7-gen-texts

echo "=== failure taxonomy ==="
$PY scripts/analyze_outputs.py artifacts/pilot7-gen-texts || true

echo "=== [4/4] collect artifacts ==="
mkdir -p artifacts
cp runs/pilot7-1.7b/adapter_model.safetensors artifacts/ 2>/dev/null || true
cp runs/pilot7-1.7b/adapter_config.json artifacts/ 2>/dev/null || true
cp runs/pilot7-1.7b/eval_results.json artifacts/ 2>/dev/null || true
kill $GPU_TRACE_PID 2>/dev/null || true
ls -la artifacts/
echo "PILOT_RUN_DONE"
