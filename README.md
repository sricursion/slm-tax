# slm-tax

Training and evaluation pipeline for small language models (Qwen3-1.7B/4B)
fine-tuned to compute US federal tax returns. Models are trained on
synthetic returns labeled by an executable TY2024 Form-1040 oracle and
evaluated on [TaxCalcBench](https://github.com/column-tax/tax-calc-bench)
TY2024 with the benchmark's own evaluator.

## Results

| System | Benchmark strict | Synthetic strict (n=1,100) | Complex (n=681) |
| --- | --- | --- | --- |
| 1.7B specialist | 11.8% | 16.9% | 7.0% |
| 4B specialist | **31.4%** | 19.7% | 9.5% |
| 4B + corrected supervision | 27.5% | **28.9%** | **17.9%** |
| Best open-weight baseline (~1T MoE) | 31.4% | — | — |
| Frontier (cited) | 62.75% | — | — |

The 4B specialist matches the strongest open-weight model measured at
~1/250th the parameters. A pre-registered intervention shows that making
the supervision derive—rather than state—every quantity nearly doubles
complex-case accuracy at fixed scale; a 1,100-case stratified evaluation
shows the benchmark's scale effect largely reflects distribution-shift
robustness.

## Layout

```
src/slm_tax/          oracle (engine.py), data generator, trace rendering, scorer
scripts/              preflight gates, data generation, training, evaluation,
                      failure taxonomy, paired statistics, figures
docs/oracle-notes.md  reverse-engineered scoring conventions
```

## Reproduce

```bash
git clone https://github.com/sricursion/slm-tax && cd slm-tax
git clone --depth 1 https://github.com/column-tax/tax-calc-bench external/tax-calc-bench
pip install -e . && pip install transformers trl peft datasets accelerate lxml

python scripts/preflight.py                                   # must be ALL GREEN
python scripts/generate_data.py --n 20000 --seed 1000 --out data/train.jsonl
SEED=1000 bash scripts/run11.sh                               # train + evaluate
python scripts/analyze_outputs.py artifacts/run11_s1000-bench-texts
```

Datasets regenerate deterministically from seeds; every generated case is
checked against an outcome signature and a SHA-256 hash of each benchmark
input. One 80 GB+ GPU suffices; use `--max-new-tokens 4096` when evaluating
models trained on the corrected traces.

MIT License.
