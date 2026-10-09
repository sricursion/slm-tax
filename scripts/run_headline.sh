#!/usr/bin/env bash
# Headline run (E3): Qwen3-4B LoRA on 100k oracle examples, then benchmark eval.
# Launch only after E1 (run_pilot.sh) clears >31.4% strict. Best on H100/MI300X.
# Usage: nohup bash scripts/run_headline.sh > /root/headline_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="$HOME/venv"
PY="$VENV/bin/python"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
NTRAIN="${NTRAIN:-100000}"
RUN="${RUN:-headline-4b}"

echo "=== [1/4] setup ==="
if ! $PY -c "import torch" 2>/dev/null; then
    sudo apt-get install -y -qq python3-venv python3-pip 2>/dev/null \
        || apt-get install -y -qq python3-venv python3-pip
    python3 -m venv "$VENV"
    $PY -m pip install -U pip -q
    if command -v nvidia-smi >/dev/null 2>&1; then IDX_LIST="cu124 cu121"; else IDX_LIST="rocm7.0 rocm6.4"; fi
    for idx in $IDX_LIST; do
        $PY -m pip install torch --index-url https://download.pytorch.org/whl/$idx -q && \
        $PY -c "import torch; assert torch.cuda.is_available()" && break || $PY -m pip uninstall -y torch -q
    done
fi
$PY -m pip install -q transformers trl peft datasets accelerate lxml
$PY -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
[ -d external/tax-calc-bench ] || git clone --depth 1 https://github.com/column-tax/tax-calc-bench external/tax-calc-bench
[ -f data/train_${NTRAIN}_v5.jsonl ] || $PY scripts/generate_data.py --n "$NTRAIN" --seed 1000 --out data/train_${NTRAIN}_v5.jsonl
[ -f data/val_2k_v5.jsonl ] || $PY scripts/generate_data.py --n 2000 --seed 5000000 --out data/val_2k_v5.jsonl
$PY scripts/preflight.py

echo "=== [2/4] train (Qwen3-4B LoRA, ${NTRAIN} examples) ==="
mkdir -p artifacts
[ -s artifacts/gpu_trace.csv ] || bash scripts/gpu_sample.sh --header > artifacts/gpu_trace.csv
( bash scripts/gpu_sample.sh --loop 15 >> artifacts/gpu_trace.csv ) &
TRACE=$!; trap "kill $TRACE 2>/dev/null || true" EXIT
# 4B needs more memory than 1.7B; smaller per-device batch, more accum (eff 32).
BSZ="${BSZ:-$($PY -c "import torch; print(4 if torch.cuda.get_device_properties(0).total_memory/1e9 > 120 else 1)")}"
ACCUM=$((32 / BSZ))
echo "batch size $BSZ x accum $ACCUM"
$PY scripts/train_sft.py --model "$MODEL" --data data/train_${NTRAIN}_v5.jsonl \
    --lora --lr 2e-4 --epochs 2 --seq-len 10240 --bsz "$BSZ" --accum "$ACCUM" --out runs/$RUN

echo "=== [3/4] eval (unconstrained; save texts for taxonomy) ==="
$PY scripts/eval_model.py --model runs/$RUN --benchmark \
    --synthetic data/val_2k_v5.jsonl --limit 500 --batch-size 8 --max-new-tokens 3072 \
    --save-texts artifacts/${RUN}-gen-texts
$PY scripts/analyze_outputs.py artifacts/${RUN}-gen-texts || true

echo "=== [4/4] collect ==="
cp runs/$RUN/adapter_model.safetensors artifacts/ 2>/dev/null || true
cp runs/$RUN/adapter_config.json artifacts/ 2>/dev/null || true
cp runs/$RUN/eval_results.json artifacts/${RUN}_eval.json 2>/dev/null || true
kill $TRACE 2>/dev/null || true
ls -la artifacts/
echo "HEADLINE_RUN_DONE"
