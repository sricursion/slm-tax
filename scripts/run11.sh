#!/usr/bin/env bash
# Run 11+ — the fork experiment and its seed replicas (E1/E2).
# Qwen3-4B, v9 fixed data, 20k examples, 1 FULL epoch, config otherwise
# identical to Run 8 (LoRA r=32 a=64 all-linear, lr 2e-4 cosine warmup 3%,
# eff batch 32, seq-len 10240, bf16, completion-only loss, thinking off).
#
# Usage: SEED=1000 nohup bash scripts/run11.sh > /root/run11_s1000.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/root/venv/bin/python
SEED="${SEED:-1000}"
RUN="run11_s${SEED}"
DATA="data/train_20k_v9_s${SEED}.jsonl"

echo "=== [1/4] data + preflight (seed $SEED) ==="
[ -f "$DATA" ] || $PY scripts/generate_data.py --n 20000 --seed "$SEED" --out "$DATA"
[ -f data/val_1100_v9.jsonl ] || $PY scripts/generate_data.py --n 1100 --seed 5000000 --out data/val_1100_v9.jsonl
$PY scripts/preflight.py

echo "=== [2/4] train (Qwen3-4B, 20k, 1 epoch, seed $SEED) ==="
mkdir -p artifacts
[ -s artifacts/gpu_trace.csv ] || bash scripts/gpu_sample.sh --header > artifacts/gpu_trace.csv
( bash scripts/gpu_sample.sh --loop 15 >> artifacts/gpu_trace.csv ) &
TRACE=$!; trap "kill $TRACE 2>/dev/null || true" EXIT
$PY scripts/train_sft.py --model Qwen/Qwen3-4B --data "$DATA" \
    --lora --lr 2e-4 --epochs 1 --seq-len 10240 --bsz 4 --accum 8 \
    --save-steps 200 --seed "$SEED" --out "runs/$RUN"
echo "=== TRAIN_SAVED $RUN ==="

echo "=== [3/4] eval: benchmark (51) + synthetic (1100) ==="
$PY scripts/eval_model.py --model "runs/$RUN" --tokenizer Qwen/Qwen3-4B \
    --benchmark --batch-size 24 --no-stop --max-new-tokens 4096 \
    --save-texts "artifacts/${RUN}-bench-texts" --out "artifacts/${RUN}_bench_eval.json"
$PY scripts/eval_model.py --model "runs/$RUN" --tokenizer Qwen/Qwen3-4B \
    --synthetic data/val_1100_v9.jsonl --limit 1100 --batch-size 24 --no-stop \
    --max-new-tokens 4096 \
    --save-texts "artifacts/${RUN}-syn-texts" --out "artifacts/${RUN}_syn_eval.json"

echo "=== [4/4] taxonomy ==="
$PY scripts/analyze_outputs.py "artifacts/${RUN}-bench-texts" || true
echo "RUN11_DONE $RUN"
