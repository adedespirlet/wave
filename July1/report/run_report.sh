#!/bin/bash
# Incremental optimization sweep: 2048x2048, tile 256x256, K=1024 and K=8192
# Run from inside docker at /workspace/wave

OUTBASE="/workspace/wave/July1/report"
LOG="$OUTBASE/bench.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

run_variant() {
    local name="$1"
    local test="$2"
    local shape="$3"
    local block="$4"
    local envvars="${5:-}"
    local outdir="$OUTBASE/$name"

    log "=== START: $name | shape=$shape block=$block ==="
    mkdir -p "$outdir"

    local cmd=(
        rocprofv3
        --kernel-trace
        --kernel-include-regex gemm
        --stats TRUE
        --output-format csv
        -d "$outdir"
        --
        python /workspace/wave/examples/python/7.1_schedule.py
        --test "$test"
        --shape "$shape"
        --block "$block"
    )

    if [ -n "$envvars" ]; then
        env WAVE_CACHE_ON=0 $envvars "${cmd[@]}" >> "$LOG" 2>&1
    else
        env WAVE_CACHE_ON=0 "${cmd[@]}" >> "$LOG" 2>&1
    fi

    local rc=$?
    if [ $rc -eq 0 ]; then
        log "  OK: $name (exit 0)"
    else
        log "  FAILED: $name (exit $rc)"
    fi
}

log "=== SWEEP START: 2048x2048, tile=256x256, K=1024 and K=8192 ==="

# K=1024
run_variant "baseline_K1024"      "test_baseline_8wave_pingpong_mxfp_gemm" "2048,2048,1024" "256,256,256" ""
run_variant "opt0_nounroll_K1024" "test_dbuf_8wave_pingpong_mxfp_gemm"     "2048,2048,1024" "256,256,256" "WAVE_MXFP4_VARIANT=opt0 WAVE_ENABLE_UNROLL=0"
run_variant "opt1_nounroll_K1024" "test_dbuf_8wave_pingpong_mxfp_gemm"     "2048,2048,1024" "256,256,256" "WAVE_MXFP4_VARIANT=opt1 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_nounroll_K1024" "test_dbuf_8wave_pingpong_mxfp_gemm"     "2048,2048,1024" "256,256,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_unroll_K1024"   "test_dbuf_8wave_pingpong_mxfp_gemm"     "2048,2048,1024" "256,256,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1"

# K=8192
run_variant "baseline_K8192"      "test_baseline_8wave_pingpong_mxfp_gemm" "2048,2048,8192" "256,256,256" ""
run_variant "opt0_nounroll_K8192" "test_dbuf_8wave_pingpong_mxfp_gemm"     "2048,2048,8192" "256,256,256" "WAVE_MXFP4_VARIANT=opt0 WAVE_ENABLE_UNROLL=0"
run_variant "opt1_nounroll_K8192" "test_dbuf_8wave_pingpong_mxfp_gemm"     "2048,2048,8192" "256,256,256" "WAVE_MXFP4_VARIANT=opt1 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_nounroll_K8192" "test_dbuf_8wave_pingpong_mxfp_gemm"     "2048,2048,8192" "256,256,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_unroll_K8192"   "test_dbuf_8wave_pingpong_mxfp_gemm"     "2048,2048,8192" "256,256,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1"

log "=== ALL RUNS COMPLETE ==="
