#!/bin/bash
set -euo pipefail

OUTBASE=/workspace/wave/July1/report
mkdir -p "$OUTBASE"
LOG="$OUTBASE/bench_16384.log"

for K in 1024 8192; do
  for variant in baseline opt0_nounroll opt1_nounroll opt2_nounroll opt2_unroll; do
    outdir="$OUTBASE/${variant}_16384_K${K}"
    mkdir -p "$outdir"

    case $variant in
      baseline)
        TEST="test_baseline_8wave_pingpong_mxfp_gemm"
        EXTRA=""
        ;;
      opt0_nounroll)
        TEST="test_dbuf_8wave_pingpong_mxfp_gemm"
        EXTRA="WAVE_MXFP4_VARIANT=opt0 WAVE_ENABLE_UNROLL=0"
        ;;
      opt1_nounroll)
        TEST="test_dbuf_8wave_pingpong_mxfp_gemm"
        EXTRA="WAVE_MXFP4_VARIANT=opt1 WAVE_ENABLE_UNROLL=0"
        ;;
      opt2_nounroll)
        TEST="test_dbuf_8wave_pingpong_mxfp_gemm"
        EXTRA="WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=0"
        ;;
      opt2_unroll)
        TEST="test_dbuf_8wave_pingpong_mxfp_gemm"
        EXTRA="WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1"
        ;;
    esac

    echo "[$(date '+%H:%M:%S')] START: ${variant}_16384_K${K}" | tee -a "$LOG"
    env WAVE_CACHE_ON=0 $EXTRA \
      rocprofv3 \
        --kernel-trace \
        --kernel-include-regex gemm \
        --stats TRUE \
        --output-format csv \
        -d "$outdir" \
        -- python /workspace/wave/examples/python/7.1_schedule.py \
             --test "$TEST" \
             --shape 16384,16384,$K \
             --block 256,256,256 >> "$LOG" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE: ${variant}_16384_K${K} exit=$?" | tee -a "$LOG"
  done
done

echo "[$(date '+%H:%M:%S')] ALL DONE" | tee -a "$LOG"
