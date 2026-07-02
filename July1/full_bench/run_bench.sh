#!/bin/bash
# Full benchmark: 8192x8192, K=1024 and K=8192, all variants, 128x128 and 256x256 tiles
# Run from inside docker at /workspace/wave

# Paths as seen from inside docker
OUTBASE="/workspace/wave/July1/full_bench"
LOG="$OUTBASE/bench.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

run_variant() {
    local name="$1"
    local test="$2"
    local shape="$3"
    local block="$4"
    # extra env vars as a single string like "VAR1=val1 VAR2=val2"
    local envvars="${5:-}"
    local outdir="$OUTBASE/$name"

    log "=== START: $name | shape=$shape block=$block ==="
    mkdir -p "$outdir"

    # Build command array
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

    # Run with env vars
    if [ -n "$envvars" ]; then
        env WAVE_CACHE_ON=0 $envvars "${cmd[@]}" >> "$LOG" 2>&1
    else
        env WAVE_CACHE_ON=0 "${cmd[@]}" >> "$LOG" 2>&1
    fi

    local rc=$?
    if [ $rc -eq 0 ]; then
        log "  OK: $name"
    else
        log "  FAILED: $name (exit $rc)"
    fi
}

# ─── 256×256 tile, K=1024 ───────────────────────────────────────────────────
run_variant "baseline_256_K1024"      "test_baseline_8wave_pingpong_mxfp_gemm"           "8192,8192,1024" "256,256,256" ""
run_variant "opt0_nounroll_256_K1024" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,1024" "256,256,256" "WAVE_MXFP4_VARIANT=opt0 WAVE_ENABLE_UNROLL=0"
run_variant "opt1_nounroll_256_K1024" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,1024" "256,256,256" "WAVE_MXFP4_VARIANT=opt1 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_nounroll_256_K1024" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,1024" "256,256,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_unroll_256_K1024"   "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,1024" "256,256,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1"
run_variant "vecxor_256_K1024"        "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,1024" "256,256,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1"
run_variant "permlane_ct_256_K1024"   "test_dbuf_8wave_pingpong_mxfp_gemm_ct_K1024"     "8192,8192,1024" "256,256,256" ""

# ─── 256×256 tile, K=8192 ───────────────────────────────────────────────────
run_variant "baseline_256_K8192"      "test_baseline_8wave_pingpong_mxfp_gemm"           "8192,8192,8192" "256,256,256" ""
run_variant "opt0_nounroll_256_K8192" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,8192" "256,256,256" "WAVE_MXFP4_VARIANT=opt0 WAVE_ENABLE_UNROLL=0"
run_variant "opt1_nounroll_256_K8192" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,8192" "256,256,256" "WAVE_MXFP4_VARIANT=opt1 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_nounroll_256_K8192" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,8192" "256,256,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_unroll_256_K8192"   "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,8192" "256,256,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1"
run_variant "vecxor_256_K8192"        "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,8192" "256,256,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1"
run_variant "permlane_ct_256_K8192"   "test_dbuf_8wave_pingpong_mxfp_gemm_ct"            "8192,8192,8192" "256,256,256" ""

# ─── 128×128 tile, K=1024 ───────────────────────────────────────────────────
run_variant "baseline_128_K1024"      "test_baseline_8wave_pingpong_mxfp_gemm"           "8192,8192,1024" "128,128,256" ""
run_variant "opt0_nounroll_128_K1024" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,1024" "128,128,256" "WAVE_MXFP4_VARIANT=opt0 WAVE_ENABLE_UNROLL=0"
run_variant "opt1_nounroll_128_K1024" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,1024" "128,128,256" "WAVE_MXFP4_VARIANT=opt1 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_nounroll_128_K1024" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,1024" "128,128,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_unroll_128_K1024"   "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,1024" "128,128,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1"
run_variant "vecxor_128_K1024"        "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,1024" "128,128,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1"
run_variant "permlane_ct_128_K1024"   "test_dbuf_8wave_pingpong_mxfp_gemm_ct_128x128"   "8192,8192,1024" "128,128,256" ""

# ─── 128×128 tile, K=8192 ───────────────────────────────────────────────────
run_variant "baseline_128_K8192"      "test_baseline_8wave_pingpong_mxfp_gemm"           "8192,8192,8192" "128,128,256" ""
run_variant "opt0_nounroll_128_K8192" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,8192" "128,128,256" "WAVE_MXFP4_VARIANT=opt0 WAVE_ENABLE_UNROLL=0"
run_variant "opt1_nounroll_128_K8192" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,8192" "128,128,256" "WAVE_MXFP4_VARIANT=opt1 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_nounroll_128_K8192" "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,8192" "128,128,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=0"
run_variant "opt2_unroll_128_K8192"   "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,8192" "128,128,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1"
run_variant "vecxor_128_K8192"        "test_dbuf_8wave_pingpong_mxfp_gemm"               "8192,8192,8192" "128,128,256" "WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1"
run_variant "permlane_ct_128_K8192"   "test_dbuf_8wave_pingpong_mxfp_gemm_ct_128x128"   "8192,8192,8192" "128,128,256" ""

log "=== ALL RUNS COMPLETE ==="
