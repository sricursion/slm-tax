#!/usr/bin/env bash
# Emit one CSV line of GPU telemetry: timestamp,util%,vram%,power_W,temp_C
# Works on both NVIDIA (nvidia-smi) and AMD/ROCm (rocm-smi). Backend-agnostic
# so the same trace format is comparable across node types.
#
# Usage:
#   bash scripts/gpu_sample.sh              # one sample
#   bash scripts/gpu_sample.sh --loop 15    # sample every 15s (Ctrl-C to stop)
#   bash scripts/gpu_sample.sh --header     # print the CSV header
set -uo pipefail

sample() {
    local ts util vram power temp
    ts=$(date -u +%H:%M:%S)
    if command -v nvidia-smi >/dev/null 2>&1; then
        read -r util vram_used vram_total power temp <<<"$(nvidia-smi \
            --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '%,' )"
        vram=$(awk -v u="${vram_used:-0}" -v t="${vram_total:-1}" 'BEGIN{printf "%.0f", (t>0? u*100/t : 0)}')
    elif command -v rocm-smi >/dev/null 2>&1; then
        util=$(rocm-smi --showuse 2>/dev/null | grep -oE 'GPU use \(%\): *[0-9]+' | grep -oE '[0-9]+$' | head -1)
        # "GPU Memory Allocated (VRAM%)" is the portable field; "Memory Activity"
        # is N/A on MI300X VF. Fall back to computing from showmeminfo bytes.
        vram=$(rocm-smi --showmemuse 2>/dev/null | grep -oE 'VRAM%\): *[0-9]+' | grep -oE '[0-9]+$' | head -1)
        if [ -z "${vram:-}" ]; then
            vram=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '
                /Total Memory/ {t=$NF} /Total Used Memory/ {u=$NF}
                END {if (t>0) printf "%.0f", u*100/t}')
        fi
        power=$(rocm-smi --showpower 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
        temp=$(rocm-smi --showtemp 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
    fi
    echo "${ts},${util:-NA},${vram:-NA},${power:-NA},${temp:-NA}"
}

case "${1:-}" in
    --header) echo "time_utc,gpu_util_pct,vram_pct,power_w,temp_c" ;;
    --loop)
        interval="${2:-15}"
        while true; do sample; sleep "$interval"; done
        ;;
    *) sample ;;
esac
