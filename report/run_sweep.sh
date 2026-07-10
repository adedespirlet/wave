#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh — timing sweep for all MXFP4 GEMM kernel variants
#
# Runs all 8 kernel variants × selected shapes using rocprofv3 --stats,
# then prints a summary table with timing (ms) and TFLOPS per variant/shape.
#
# Usage (inside Docker container at /workspace/wave):
#   bash report/run_sweep.sh [--shapes 2048|16384|all] [--skip-existing] [--dry-run]
#
# Default: --shapes all (2048×2048×8192 and 16384×16384×8192)
#
# Variants covered:
#   baseline       — test_baseline_8wave_pingpong_mxfp_gemm
#   opt0           — WAVE_MXFP4_VARIANT=opt0 WAVE_ENABLE_UNROLL=0
#   opt1           — WAVE_MXFP4_VARIANT=opt1 WAVE_ENABLE_UNROLL=0
#   opt2           — WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=0
#   opt2_unroll    — WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1
#   vecxor         — WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1 WAVE_EPILOGUE_VARIANT=vecxor
#   dpp            — WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1 WAVE_EPILOGUE_VARIANT=dpp
#   permlane_ct    — test_dbuf_8wave_pingpong_mxfp_gemm_ct (no extra env)
#
# Output:
#   /workspace/wave/report/sweep_results/YYYY-MM-DD/<tag>/   (rocprofv3 CSVs)
#   /workspace/wave/report/sweep_results/YYYY-MM-DD/summary.txt
# =============================================================================

set -uo pipefail

WAVE_DIR="/workspace/wave"
DATE_TAG=$(date +%Y-%m-%d)
BASE_OUT="${WAVE_DIR}/report/sweep_results/${DATE_TAG}"
SCRIPT="${WAVE_DIR}/examples/python/7.1_schedule.py"
SUMMARY="${BASE_OUT}/summary.txt"

# ── Parse arguments ──────────────────────────────────────────────────────────
SHAPES_ARG="all"
SKIP_EXISTING=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shapes)       SHAPES_ARG="$2"; shift 2 ;;
        --shapes=*)     SHAPES_ARG="${1#--shapes=}"; shift ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      grep '^#' "$0" | head -25 | sed 's/^# \?//'; exit 0 ;;
        *)              echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$SHAPES_ARG" != "2048" && "$SHAPES_ARG" != "16384" && "$SHAPES_ARG" != "all" ]]; then
    echo "Error: --shapes must be 2048, 16384, or all" >&2
    exit 1
fi

mkdir -p "$BASE_OUT"
cd "$WAVE_DIR"

echo "================================================================"
echo " MXFP4 GEMM Variant Timing Sweep"
echo " Shapes: $SHAPES_ARG  |  Output: $BASE_OUT"
echo " Date: $DATE_TAG  |  dry-run=${DRY_RUN}  skip-existing=${SKIP_EXISTING}"
echo "================================================================"
echo ""

# ── run_variant ──────────────────────────────────────────────────────────────
# run_variant TAG TEST_FN SHAPE BLOCK [ENV_VAR=VAL ...]
run_variant() {
    local TAG="$1"
    local TEST_FN="$2"
    local SHAPE="$3"
    local BLOCK="$4"
    shift 4
    local ENV_VARS="${*}"          # remaining args are VAR=VAL pairs

    local OUT_DIR="${BASE_OUT}/${TAG}"

    echo "----------------------------------------------------------------"
    echo " Variant: $TAG"
    echo " Shape:   $SHAPE  Block: $BLOCK"
    echo " Test:    $TEST_FN"
    [[ -n "$ENV_VARS" ]] && echo " Env:     $ENV_VARS"
    echo "----------------------------------------------------------------"

    # Skip if kernel_stats.csv already exists
    if (( SKIP_EXISTING == 1 )); then
        if ls "$OUT_DIR"/*/[0-9]*_kernel_stats.csv 2>/dev/null | head -1 | grep -q .; then
            echo "  -> SKIP (kernel_stats.csv already exists)"
            echo ""
            return 0
        fi
    fi

    if (( DRY_RUN == 1 )); then
        echo "  -> DRY RUN — skipping execution"
        echo ""
        return 0
    fi

    mkdir -p "$OUT_DIR"

    # Clear stale variant-specific vars before each run
    unset WAVE_MXFP4_VARIANT WAVE_ENABLE_UNROLL WAVE_ENABLE_SWIZZLE \
          WAVE_VECTORIZED_STORE WAVE_EPILOGUE_VARIANT 2>/dev/null || true

    set +e
    # shellcheck disable=SC2086
    env WAVE_CACHE_ON=0 WAVE_ALWAYS_COMPILE=1 ${ENV_VARS} \
        rocprofv3 \
            --stats \
            --kernel-include-regex "gemm" \
            -d "${OUT_DIR}" \
            --output-format csv \
            -- python "${SCRIPT}" \
               --test "${TEST_FN}" \
               --shape "${SHAPE}" \
               --block "${BLOCK}" \
        2>&1 | tee "${OUT_DIR}/run.log"
    local RC=${PIPESTATUS[0]}
    set -e

    if (( RC == 0 )); then
        echo "  -> DONE: $TAG"
    else
        echo "  -> FAILED: $TAG (exit $RC)"
    fi
    echo ""
}

# ── Per-shape variant sweep ───────────────────────────────────────────────────
run_shape() {
    local M="$1" N="$2" K="$3"
    local SHAPE="${M},${N},${K}"
    local BLOCK="256,256,256"
    local SFX="${M}_K${K}"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo " SHAPE: ${M}×${N}×${K}"
    echo "════════════════════════════════════════════════════════════════"

    run_variant "baseline_${SFX}" \
        "test_baseline_8wave_pingpong_mxfp_gemm" \
        "$SHAPE" "$BLOCK" \
        ""

    run_variant "opt0_${SFX}" \
        "test_dbuf_8wave_pingpong_mxfp_gemm" \
        "$SHAPE" "$BLOCK" \
        "WAVE_MXFP4_VARIANT=opt0 WAVE_ENABLE_UNROLL=0"

    run_variant "opt1_${SFX}" \
        "test_dbuf_8wave_pingpong_mxfp_gemm" \
        "$SHAPE" "$BLOCK" \
        "WAVE_MXFP4_VARIANT=opt1 WAVE_ENABLE_UNROLL=0"

    run_variant "opt2_${SFX}" \
        "test_dbuf_8wave_pingpong_mxfp_gemm" \
        "$SHAPE" "$BLOCK" \
        "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=0"

    run_variant "opt2_unroll_${SFX}" \
        "test_dbuf_8wave_pingpong_mxfp_gemm" \
        "$SHAPE" "$BLOCK" \
        "WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1"

    run_variant "vecxor_${SFX}" \
        "test_dbuf_8wave_pingpong_mxfp_gemm" \
        "$SHAPE" "$BLOCK" \
        "WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1 WAVE_EPILOGUE_VARIANT=vecxor"

    run_variant "dpp_${SFX}" \
        "test_dbuf_8wave_pingpong_mxfp_gemm" \
        "$SHAPE" "$BLOCK" \
        "WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1 WAVE_EPILOGUE_VARIANT=dpp"

    run_variant "permlane_ct_${SFX}" \
        "test_dbuf_8wave_pingpong_mxfp_gemm_ct" \
        "$SHAPE" "$BLOCK" \
        ""
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
case "$SHAPES_ARG" in
    2048)  run_shape 2048  2048  8192 ;;
    16384) run_shape 16384 16384 8192 ;;
    all)   run_shape 2048  2048  8192
           run_shape 16384 16384 8192 ;;
esac

# ── Summary table (Python) ───────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " SUMMARY"
echo "════════════════════════════════════════════════════════════════"

python3 - "$BASE_OUT" "$SHAPES_ARG" << 'PYEOF'
import csv, os, glob, sys

base_out   = sys.argv[1]
shapes_arg = sys.argv[2]

VARIANTS = [
    "baseline",
    "opt0",
    "opt1",
    "opt2",
    "opt2_unroll",
    "vecxor",
    "dpp",
    "permlane_ct",
]

if shapes_arg == "2048":
    SHAPES = [(2048, 2048, 8192)]
elif shapes_arg == "16384":
    SHAPES = [(16384, 16384, 8192)]
else:
    SHAPES = [(2048, 2048, 8192), (16384, 16384, 8192)]


def get_gemm_avg_ns(tag_dir):
    """Return AverageNs for the 'gemm' kernel from *_kernel_stats.csv."""
    pattern = os.path.join(tag_dir, "**", "*_kernel_stats.csv")
    for csv_path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Name", "")
                    # strip surrounding quotes if present
                    name = name.strip('"')
                    if name == "gemm":
                        return float(row["AverageNs"])
        except Exception:
            continue
    return None


def tflops(M, N, K, avg_ns):
    """TFLOPS = 2*M*N*K / (avg_ns * 1e-9) / 1e12  =  2*M*N*K / avg_ns / 1e3"""
    return 2 * M * N * K / avg_ns / 1e3


# Collect results: results[(variant, M, K)] = (avg_ns, tflops, status)
results = {}
for M, N, K in SHAPES:
    sfx = f"{M}_K{K}"
    for var in VARIANTS:
        tag = f"{var}_{sfx}"
        tag_dir = os.path.join(base_out, tag)
        avg_ns = get_gemm_avg_ns(tag_dir)
        if avg_ns is not None:
            tf = tflops(M, N, K, avg_ns)
            log_path = os.path.join(tag_dir, "run.log")
            failed = False
            if os.path.exists(log_path):
                with open(log_path) as f:
                    failed = "FAILED" in f.read()
            status = "FAILED" if failed else "OK"
            results[(var, M, K)] = (avg_ns / 1e6, tf, status)  # store ms
        else:
            results[(var, M, K)] = (None, None, "NO_DATA")

# Print table
HDR = f"{'Variant':<22} | {'Shape (MxNxK)':<20} | {'Time (ms)':>10} | {'TFLOPS':>10} | Status"
SEP = "-" * len(HDR)
print(HDR)
print(SEP)

for M, N, K in SHAPES:
    for var in VARIANTS:
        ms, tf, status = results[(var, M, K)]
        ms_str = f"{ms:.3f}" if ms is not None else "N/A"
        tf_str = f"{tf:.1f}" if tf is not None else "N/A"
        print(f"{var:<22} | {M}x{N}x{K:<12} | {ms_str:>10} | {tf_str:>10} | {status}")
    print(SEP)

# Save to summary.txt
summary_path = os.path.join(base_out, "summary.txt")
with open(summary_path, "w") as f:
    f.write(HDR + "\n")
    f.write(SEP + "\n")
    for M, N, K in SHAPES:
        for var in VARIANTS:
            ms, tf, status = results[(var, M, K)]
            ms_str = f"{ms:.3f}" if ms is not None else "N/A"
            tf_str = f"{tf:.1f}" if tf is not None else "N/A"
            f.write(f"{var:<22} | {M}x{N}x{K:<12} | {ms_str:>10} | {tf_str:>10} | {status}\n")
        f.write(SEP + "\n")

print(f"\nSummary written to: {summary_path}")
PYEOF

echo ""
echo "Output directory: $BASE_OUT"
