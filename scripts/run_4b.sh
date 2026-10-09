#!/usr/bin/env bash
# Budget-bounded Qwen3-4B run (E3). Measured 1.7B throughput (31.7s/step at
# bsz8xaccum4) scales to ~75s/step at 4B/bsz4xaccum8, so a full epoch (625
# steps) would be 11-15h. This runs a bounded fraction with frequent
# checkpoints so we can evaluate whatever lands inside the GPU budget.
#
# Usage: STEPS_EPOCH=0.5 nohup bash scripts/run_4b.sh > /root/run4b.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="$HOME/venv"; PY="$VENV/bin/python"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
RUN="${RUN:-headline-4b}"
EPOCHS="${EPOCHS:-0.5}"          # 0.5 epoch of 20k = ~313 steps ~5.5-7.4h
DATA="${DATA:-data/train_20k_v7.jsonl}"

echo "=== [1/4] setup ==="
$PY -m pip install -q transformers trl peft datasets accelerate lxml 2>/dev/null || true
$PY -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
[ -f "$DATA" ] || $PY scripts/generate_data.py --n 20000 --seed 1000 --out "$DATA"
[ -f data/val_2k_v7.jsonl ] || $PY scripts/generate_data.py --n 2000 --seed 5000000 --out data/val_2k_v7.jsonl
$PY scripts/preflight.py

echo "=== [2/4] train 4B (epochs=$EPOCHS) ==="
mkdir -p artifacts
[ -s artifacts/gpu_trace_4b.csv ] || bash scripts/gpu_sample.sh --header > artifacts/gpu_trace_4b.csv
( bash scripts/gpu_sample.sh --loop 15 >> artifacts/gpu_trace_4b.csv ) &
TRACE=$!; trap "kill $TRACE 2>/dev/null || true" EXIT
$PY scripts/train_sft.py --model "$MODEL" --data "$DATA" \
    --lora --lr 2e-4 --epochs "$EPOCHS" --seq-len 10240 --bsz 4 --accum 8 \
    --save-steps 100 --out runs/$RUN

echo "=== [3/4] eval ==="
$PY scripts/eval_model.py --model runs/$RUN --tokenizer "$MODEL" --benchmark \
    --synthetic data/val_2k_v7.jsonl --limit 150 --batch-size 4 --max-new-tokens 3072 \
    --save-texts artifacts/${RUN}-gen-texts --out /root/${RUN}_eval.json
$PY scripts/analyze_outputs.py artifacts/${RUN}-gen-texts || true

echo "=== [4/4] collect ==="
cp runs/$RUN/adapter_model.safetensors artifacts/ 2>/dev/null || true
cp runs/$RUN/adapter_config.json artifacts/ 2>/dev/null || true
kill $TRACE 2>/dev/null || true
echo RUN4B_DONE
