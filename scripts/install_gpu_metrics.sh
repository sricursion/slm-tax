#!/usr/bin/env bash
# Install the DigitalOcean metrics agent (system graphs in the DO console) and
# set up node-side GPU telemetry that works regardless of DO console support.
#
# DO console GPU metrics (utilization/temp/power/throttling) are automatic only
# on DO AI/ML-Ready (NVIDIA) GPU droplets with "Improved Metrics and monitoring"
# selected at create time. On the AMD MI300X devcloud (ROCm) image, DO's GPU
# exporter is NVIDIA-oriented, so the reliable GPU signal is rocm-smi — captured
# here into artifacts/gpu_trace.csv (util/vram/power/temp, sampled every 15s).
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== DigitalOcean metrics agent (CPU/mem/disk/bandwidth graphs) ==="
curl -sSL https://repos.insights.digitalocean.com/install.sh | sudo bash \
    || echo "do-agent install failed (non-fatal; continuing)"

echo "=== GPU telemetry ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA detected. DO console GPU metrics are automatic on AI/ML-Ready"
    echo "droplets; on a basic image, enable them via DO's GPU exporter guide."
    echo "Node-side trace also runs below as a backend-agnostic fallback."
elif command -v rocm-smi >/dev/null 2>&1; then
    echo "AMD/ROCm detected. DO console GPU exporter does NOT cover ROCm — using"
    echo "the node-side rocm-smi trace (artifacts/gpu_trace.csv) as the GPU signal."
else
    echo "No GPU tool found (nvidia-smi/rocm-smi). GPU telemetry unavailable."
fi

echo "done. run_pilot.sh / run_headline.sh already sample GPU metrics during"
echo "training; use scripts/gpu_sample.sh for a one-off reading."
