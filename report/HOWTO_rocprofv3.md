# HOWTO: Profiling MXFP4 GEMM Kernels with rocprofv3

This guide consolidates how to run the MXFP4 GEMM kernel tests under `rocprofv3` for
timing, ATT (Asynchronous Thread Trace) collection, and hardware counter measurement.

---

## 1. Overview

`rocprofv3` is used here in three modes:

| Mode | Purpose |
|------|---------|
| `--stats` | Wall-time and kernel dispatch statistics |
| `--att` | Per-wavefront instruction traces (ATT), wave JSON files |
| `-i counters.json` | Hardware performance counter collection (e.g. LDS bank conflicts) |

**Test script:** `examples/python/7.1_schedule.py`  
All runs target the `test_dbuf_8wave_pingpong_mxfp_gemm` family of test functions
(or `_ct` / `_ct_K1024` variants for the `permlane_ct` epilogue).

### Docker container

All commands run inside a container with GPU access:

```bash
docker run -d --name wave_rocm720 \
  --device /dev/kfd --device /dev/dri \
  -v /home/adespirl/wave:/workspace/wave \
  -w /workspace/wave \
  rocm/sgl-dev:v0.5.13.post1-rocm720-mi35x-20260625 \
  sleep infinity
```

Container name (configurable via `$WAVE_CONTAINER`): **`wave_rocm720`**  
Working directory inside container: `/workspace/wave`

---

## 2. Environment Variables

These `WAVE_*` variables control which kernel variant is compiled and run.
Always set `WAVE_CACHE_ON=0 WAVE_ALWAYS_COMPILE=1` when profiling to ensure a fresh
compile and avoid cached bytecode from a different configuration.

| Variable | Values | Description |
|---|---|---|
| `WAVE_MXFP4_VARIANT` | `opt0`, `opt1`, `opt2` | Selects the MXFP4 scheduling variant. Use `opt2` for all current benchmark variants. |
| `WAVE_ENABLE_UNROLL` | `0` / `1` | Enables K-loop unrolling (`opt2_unroll` variant). Do **not** combine with `WAVE_VECTORIZED_STORE`. |
| `WAVE_ENABLE_SWIZZLE` | `0` / `1` | Enables LDS swizzling (default `1`). Disabling introduces ~4.7M LDS bank conflicts per dispatch. |
| `WAVE_VECTORIZED_STORE` | `0` / `1` | Activates the vectorized epilogue path. Required for `vecxor`, `dpp`, and `permlane_ct`. |
| `WAVE_EPILOGUE_VARIANT` | `vecxor` / `dpp` / `permlane_ct` | Selects the cross-lane reduction strategy when `WAVE_VECTORIZED_STORE=1`. |
| `WAVE_CACHE_ON` | `0` / `1` | Set to `0` before profiling to force recompilation. |
| `WAVE_ALWAYS_COMPILE` | `0` / `1` | Set to `1` to guarantee the kernel is recompiled, ignoring any cached artifact. |

### Correct env-var combinations per variant

| Variant | Required env vars | Test function |
|---|---|---|
| `opt2_unroll` | `WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1` | `test_dbuf_8wave_pingpong_mxfp_gemm` |
| `vecxor` | `WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1 WAVE_EPILOGUE_VARIANT=vecxor` | `test_dbuf_8wave_pingpong_mxfp_gemm` |
| `dpp` | `WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1 WAVE_EPILOGUE_VARIANT=dpp` | `test_dbuf_8wave_pingpong_mxfp_gemm` |
| `permlane_ct` (K=8192) | `WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1 WAVE_EPILOGUE_VARIANT=permlane_ct` | `test_dbuf_8wave_pingpong_mxfp_gemm_ct` |
| `permlane_ct` (K=1024) | same as above | `test_dbuf_8wave_pingpong_mxfp_gemm_ct_K1024` |

> **Warning:** Do **not** set `WAVE_VECTORIZED_STORE=1` together with `WAVE_ENABLE_UNROLL=1`.
> This will silently activate the `vecxor` MLIR path instead of the programmatic compile —
> the sanity checks in §7 will catch this.

---

## 3. Timing Runs

Use `rocprofv3 --att --kernel-trace --stats` to collect timing alongside ATT.
The `--stats` flag writes a `*_kernel_stats.csv` summarizing per-kernel dispatch
wall time and stall breakdowns.

### Example (inside container)

```bash
cd /workspace/wave

WAVE_CACHE_ON=0 WAVE_ALWAYS_COMPILE=1 \
WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1 WAVE_ENABLE_SWIZZLE=1 \
  rocprofv3 \
    --att \
    --kernel-trace \
    --stats \
    -d /workspace/wave/report/att_256x256/final_bench_$(date +%Y-%m-%d)/opt2_unroll_2048_K8192 \
    -- python examples/python/7.1_schedule.py \
      --test test_dbuf_8wave_pingpong_mxfp_gemm \
      --shape 2048,2048,8192 \
      --block 256,256,256
```

### Full benchmark script

`report/att_256x256/run_final_bench.sh` automates all 9 variants for both shapes
(2048×2048×8192 and 16384×16384×8192). Run it inside the container:

```bash
# All shapes
bash report/att_256x256/run_final_bench.sh

# One shape only
bash report/att_256x256/run_final_bench.sh --shape 2048
bash report/att_256x256/run_final_bench.sh --shape 16384
```

Output lands in `/workspace/wave/report/att_256x256/final_bench_<date>/<variant_tag>/`.
The script skips a variant if a `*_kernel_stats.csv` already exists in the output
directory (idempotent re-runs).

> **Note:** The 2048 shape typically produces 0 GEMM wave JSON files because the
> dispatch is too fast for ATT sampling. Matrix core utilization analysis (§6) is
> only meaningful for the 16384 shape.

---

## 4. ATT Trace Collection

### Command

```bash
cd /workspace/wave

WAVE_CACHE_ON=0 WAVE_ALWAYS_COMPILE=1 \
WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1 WAVE_EPILOGUE_VARIANT=dpp \
  rocprofv3 \
    --att \
    --att-target-cu 0 \
    --att-shader-engine-mask 0xFFFFFFF \
    --kernel-include-regex "gemm" \
    -d /workspace/wave/report/att_256x256/dpp_2048_K1024 \
    --output-format csv \
    -- python examples/python/7.1_schedule.py \
      --test test_dbuf_8wave_pingpong_mxfp_gemm \
      --shape 2048,2048,1024 \
      --block 256,256,256
```

Key flags:
- `--att-target-cu 0` — trace CU 0 only (sufficient for single-dispatch analysis)
- `--att-shader-engine-mask 0xFFFFFFF` — enable all shader engines
- `--kernel-include-regex "gemm"` — restrict tracing to kernels whose name contains `gemm`
- `--output-format csv` — emit CSV alongside the wave JSON files

### Output files

After a successful run, the output directory contains:

| File / pattern | Contents |
|---|---|
| `<hash>/wave_*.json` | Per-wavefront instruction traces (input to `get_loop_efficiency.py`) |
| `<hash>/code_*.json` | ISA disassembly with instruction indices (required alongside wave JSONs) |
| `<hash>/stats_ui_output_agent_*_dispatch_N.csv` | Per-instruction stall breakdown for dispatch N |

The `_dispatch_N.csv` files encode instruction-level stall data. ATT trace collection
for the canonical 16 runs (4 variants × 2 shapes × 2 K values) is fully automated by
`report/att_256x256/collect_traces.sh`:

```bash
# All 16 runs (from the host, outside Docker):
bash report/att_256x256/collect_traces.sh

# Single variant:
bash report/att_256x256/collect_traces.sh opt2_unroll

# Single run:
bash report/att_256x256/collect_traces.sh opt2_unroll 2048 1024
```

The script runs `docker exec wave_rocm720` internally; the container must already be
running. Override the container name with `WAVE_CONTAINER=<name>`.

---

## 5. Hardware Counter Runs

`rocprofv3 -i counters_XXX.json` collects specific hardware performance counters
across all kernel dispatches.

### Counter input file format

```json
{
    "jobs": [
        {
            "pmc": ["SQ_LDS_BANK_CONFLICT"],
            "kernel_include_regex": "gemm",
            "kernel_exclude_regex": ""
        }
    ]
}
```

`pmc` is a list of counter names. Multiple counters can be listed; rocprofv3 will
serialize dispatches as needed when the hardware cannot collect all simultaneously.

### Example — LDS bank conflict measurement

```bash
# counters file lives at report/counters_lds.json
docker exec wave_rocm720 bash -c "
  env WAVE_CACHE_ON=0 \
      WAVE_MXFP4_VARIANT=opt2 WAVE_ENABLE_UNROLL=1 WAVE_ENABLE_SWIZZLE=1 \
    rocprofv3 \
      -i /workspace/wave/report/counters_lds.json \
      -d /workspace/wave/report/lds_bank_conflicts/lds_K8192_swizzle_on \
      -- python3 /workspace/wave/examples/python/7.1_schedule.py \
           --test test_dbuf_8wave_pingpong_mxfp_gemm \
           --shape 2048,2048,8192 --block 256,256,256
"
```

See `report/lds_bank_conflicts/README.md` for the full swizzle-on vs swizzle-off
comparison. Key result: enabling swizzle (`WAVE_ENABLE_SWIZZLE=1`) reduces
`SQ_LDS_BANK_CONFLICT` from ~4.7M per dispatch to **0**.

### Other counter sets

- `report/hbm_counters/` — HBM bandwidth counters
- Add new counter sets by writing a new `counters_<name>.json` following the same
  schema, then passing it with `-i`.

---

## 6. Parsing Outputs

### Matrix core utilization — `get_loop_efficiency.py`

```bash
# From inside the container (or host with Python available):
python report/get_loop_efficiency.py <att_dir> [--inst mfma|wmma] [-v]

# Example:
python report/get_loop_efficiency.py \
    report/att_256x256/dpp_16384_K8192 \
    --inst mfma -v
```

The script:
1. Loads all `wave_*.json` files from `<att_dir>` (recursively)
2. Detects the mainloop boundaries by finding the first and last `v_mfma_*` /
   `v_wmma_*` instruction cluster
3. Counts MFMA instructions per loop iteration
4. Computes **per-wave efficiency** = `(n_mfma × mfma_cycles) / single_loop_cycles × 100`

For `gfx950` (MI350X), the relevant MFMA instruction is
`v_mfma_scale_f32_32x32x64_f8f6f4` (32 cycles) for the MXFP4 GEMM kernel.

Verbose output (`-v`) shows pre-loop, mainloop, and post-loop cycle counts per wave,
plus a per-wave breakdown table when multiple wavefronts are present.

### Stall breakdown — `kernel_stats.csv`

`rocprofv3 --stats` writes a `*_kernel_stats.csv` in the output directory.
Each row corresponds to one kernel dispatch and contains columns like:

- `Duration_ns` — kernel wall time
- `VALU_stall_cycles`, `SALU_stall_cycles`, `LDS_stall_cycles` — stall sources
- `Wavefronts` — number of wavefronts launched

Open the CSV directly or load it with pandas:

```python
import pandas as pd
df = pd.read_csv("path/to/kernel_stats.csv")
# Filter to GEMM dispatches:
gemm = df[df["KernelName"].str.contains("gemm")]
print(gemm[["KernelName", "Duration_ns", "LDS_stall_cycles"]].to_string())
```

The `stats_ui_output_agent_*_dispatch_N.csv` files from ATT runs contain a
per-instruction breakdown (one row per ISA instruction with stall-cycle counts),
which `collect_traces.sh` uses for the sanity checks described in §7.

---

## 7. Sanity Checks

Before trusting profiling results, verify that the correct kernel variant was actually
compiled by inspecting the dispatch CSV for characteristic instructions.

The `collect_traces.sh` script runs these checks automatically after each trace
collection; here is a summary of what to look for in
`stats_ui_output_agent_*_dispatch_53.csv`:

| Variant | Expected | Failure sign |
|---|---|---|
| `opt2_unroll` | `ds_swizzle` count **= 0**; `v_xor_b32`/`v_add_u32` count **> 0** | `ds_swizzle > 0` → vecxor MLIR was loaded instead |
| `vecxor` | `ds_swizzle` count **> 0** | `ds_swizzle = 0` → wrong variant compiled |
| `dpp` | `v_mov_b32_dpp` count **> 0**; `ds_swizzle` count **= 0** | `v_mov_b32_dpp = 0` → DPP MLIR not loaded; `ds_swizzle > 0` → vecxor loaded |
| `permlane_ct` | `v_permlane`/`ds_permlane` count **> 0** | count = 0 → wrong test function used (fell through to vecxor) |

Quick one-liner to check from the host after a trace run:

```bash
# Count ds_swizzle instructions in the dispatch CSV (should be 0 for opt2_unroll):
grep -c "ds_swizzle" report/att_256x256/opt2_unroll_2048_K1024/stats_ui_output_agent_*_dispatch_53.csv

# Count v_mov_b32_dpp (should be > 0 for dpp):
grep -c "v_mov_b32_dpp" report/att_256x256/dpp_2048_K1024/stats_ui_output_agent_*_dispatch_53.csv
```

The `[WAVE] MLIR override loaded:` and `[WAVE] kernel config:` lines in the
`rocprofv3` stdout (captured by `collect_traces.sh` in `/tmp/wave_rocprofv3_last.log`)
also show which MLIR file was loaded and what compile-time options were active.
