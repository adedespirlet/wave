"""
MXFP4 Scaled GEMM Scheduling for GFX950 (MI350)

Double-buffered MXFP4 GEMM with 4-wave and 8-wave configurations.
Uses get_tagged_mxfp4_gemm (templates) + get_mxfp4_dbuf_schedule (schedules).

The --splitk N flag enables split-K with N splits for supported tests.

Usage:
    python 7.1_schedule.py --test test_dbuf_4wave_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_4wave_mxfp_gemm --splitk 2
    python 7.1_schedule.py --test test_dbuf_8wave_pingpong_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_8wave_pingpong_mxfp_gemm --splitk 2
    python 7.1_schedule.py --test test_splitk_preshuffle_scales_gemm_cpp
    python 7.1_schedule.py --list_tests
"""

import os
import sys
import torch
from utils import list_tests, parse_args, run_test
import pathlib
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.schedules import (
    get_mxfp4_asymmetric_schedule,
    get_mxfp4_dbuf_mixed_pingpong_schedule,
    get_mxfp4_dbuf_mixed_pingpong_shuffle_schedule,
    get_mxfp4_dbuf_pingpong_schedule,
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled,
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds,
    get_mxfp4_dbuf_schedule,
)
from wave_lang.kernel.wave.schedules.gemm_mxfp4_double_buffer import (
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt_opt00,
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt0,
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt1,
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt2,
)
from wave_lang.kernel.wave.templates import (
    get_tagged_mxfp4_gemm,
    get_tagged_mxfp4_gemm_preshuffle_b,
    get_tagged_mxfp4_gemm_preshuffle_b_wide_store,
    get_tagged_mxfp4_gemm_preshuffle_scales,
    get_tagged_mxfp4_gemm_preshuffle_scales_and_B,
    get_tagged_splitk_mxfp4_gemm,
    get_tagged_splitk_mxfp4_gemm_preshuffle_b,
    get_tagged_splitk_mxfp4_gemm_preshuffle_scales,
)
from wave_lang.kernel.wave.utils.mxfp_utils import (
    b_preshuffle,
    e8m0_shuffle,
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
)
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config


def _run_mxfp_gemm(gemm, shape):
    """Run compiled GEMM kernel and verify against reference."""
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32).cuda()

    gemm(x, x_scales, w.T.contiguous(), w_scales, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def _run_mxfp_gemm_preshuffle(
    gemm,
    shape,
    all=False,
    only_scale=False,
    only_b=False,
    output_dtype=torch.float32,
    atol=None,
    rtol=None,
):
    """Run compiled GEMM kernel with preshuffled B and B_scale, verify against reference.

    Shuffling is applied based on the flags:
      all        - shuffle a_scale (x_scales), b_scale (w_scales), and b (w_t)
      only_scale - shuffle a_scale (x_scales) and b_scale (w_scales) only
      only_b     - shuffle b_scale (w_scales) only
    """
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    w_t = w.T.contiguous()

    # Apply b (w_t) preshuffle only when all=True
    w_t_ps = b_preshuffle(w_t) if all else w_t

    # Apply a_scale shuffle when all=True or only_scale=True
    x_scales_ps = e8m0_shuffle(x_scales) if (all or only_scale) else x_scales

    # Apply b_scale shuffle when all=True, only_scale=True, or only_b=True
    w_scales_ps = e8m0_shuffle(w_scales) if (all or only_scale or only_b) else w_scales

    x, w_t_ps = x.cuda(), w_t_ps.cuda()
    x_scales_ps, w_scales_ps = x_scales_ps.cuda(), w_scales_ps.cuda()
    out = torch.zeros(x.shape[0], w_t_ps.shape[0], dtype=output_dtype).cuda()

    for _ in range(200):
        gemm(x, x_scales_ps, w_t_ps, w_scales_ps, out)

    tol_kwargs = {}
    if atol is not None:
        tol_kwargs["atol"] = atol
    if rtol is not None:
        tol_kwargs["rtol"] = rtol
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False, **tol_kwargs
    )


def _run_mxfp_gemm_preshuffle_ct(gemm, shape, output_dtype=torch.bfloat16):
    """Run a transposed GEMM kernel that computes C^T = B * A^T (scales-only preshuffle).

    The kernel receives (w, w_scales, x, x_scales) in swapped roles so that it
    computes C^T of shape (N, M).  B (activation x) is NOT preshuffled — it is
    read from global memory to LDS in-kernel, just like the base test.
    The output is compared against torch_out.T.
    """
    M_orig, N_orig, K = shape
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)  # (M, N)
    w_scales_ps = e8m0_shuffle(w_scales)
    x_scales_ps = e8m0_shuffle(x_scales)
    w_t = w.T.contiguous()
    w_t, x_c = w_t.cuda(), x.cuda()
    w_scales_ps, x_scales_ps = w_scales_ps.cuda(), x_scales_ps.cuda()
    out = torch.zeros(N_orig, M_orig, dtype=output_dtype).cuda()  # C^T shape
    gemm(w_t, w_scales_ps, x_c, x_scales_ps, out)
    torch.testing.assert_close(
        torch_out.T.contiguous(), out.cpu(), check_dtype=False, check_device=False
    )


def _run_mxfp_gemm_preshuffle_transposed(
    gemm, shape, output_dtype=torch.float32, b_preshuffled=False
):
    """Run transposed GEMM that outputs C[M, N] in row-major layout.
    Internally computes C^T = B * A^T, then the permlane_swap epilogue
    writes the result as C[M, N] to global memory.
    Output is allocated as (M, N) and compared directly against the reference.

    b_preshuffled=False (default): B (activation x) is passed as-is; the
        non-B-shuffled kernel handles the layout via global→LDS internally.
    b_preshuffled=True: x is preshuffled via b_preshuffle() before launch,
        for use with get_tagged_mxfp4_gemm_preshuffle_scales_and_B kernels.
    """
    M_orig, N_orig, K = shape
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)
    w_t = w.T.contiguous()
    x_b = b_preshuffle(x) if b_preshuffled else x
    w_scales_ps = e8m0_shuffle(w_scales)
    x_scales_ps = e8m0_shuffle(x_scales)
    w_t, x_b = w_t.cuda(), x_b.cuda()
    w_scales_ps, x_scales_ps = w_scales_ps.cuda(), x_scales_ps.cuda()

    out = torch.zeros(M_orig, N_orig, dtype=output_dtype).cuda()

    
    for _ in range(200):    
        gemm(w_t, w_scales_ps, x_b, x_scales_ps, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def _get_8wave_shape_from_block(block):
    """Choose an 8-wave shape (4x2 or 2x4) from block M/N dims.

    If either block M or N is 32, force that corresponding wave dimension to 2.
    """
    m_blk, n_blk = block[0], block[1]
    if m_blk == 32 and n_blk == 32:
        raise ValueError(
            "Cannot satisfy both M and N=32 with an 8-wave shape constrained to (4, 2) or (2, 4)."
        )
    if m_blk == 32:
        return (2, 4)
    if n_blk == 32:
        return (4, 2)
    return (4, 2)


def test_dbuf_4wave_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256), splitk=None
):
    """Double-buffered MXFP4 GEMM, 4 waves, no stagger."""
    if splitk and block == (256, 256, 256):
        block = (128, 128, 256)
    if splitk:
        gemm, options = get_tagged_splitk_mxfp4_gemm(
            shape, num_splits=splitk, block_shape=block, wave_shape=(2, 2)
        )
    else:
        gemm, options = get_tagged_mxfp4_gemm(shape, block, wave_shape=(2, 2))
    schedule = get_mxfp4_dbuf_schedule(use_stagger=False)

    options.print_ir_after = "all" if is_debug else []
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave.mlir"
    options.print_mlir = True
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    sk = f" split-K({splitk})" if splitk else ""
    print(f"MXFP GEMM double-buffer 4-wave{sk} test passed!")



def test_baseline_8wave_pingpong_mxfp_gemm(
    is_debug=False,
    shape=(2048, 2048, 8192),
    block=(256, 256, 256),
    dynamic=False,
    splitk=None,
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    A and B are read from global memory directly to LDS.

    Note: for dynamic mode, keep block MxN at or below 128x256 or 256x128
    to avoid exceeding shared-memory limits.
    """
    if splitk and block == (256, 256, 256):
        block = (128, 256, 256)
    wave_shape = _get_8wave_shape_from_block(block)
    if splitk:
        gemm, options = get_tagged_splitk_mxfp4_gemm_preshuffle_scales(
            shape,
            num_splits=splitk,
            block_shape=block,
            wave_shape=wave_shape,
            output_type=tkl.bf16,
        )
    else:
        gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
            shape,
            block,
            wave_shape=wave_shape,
            b_address_space=SHARED_ADDRESS_SPACE,
            output_dtype=tkl.bf16,
        )
    options.specialize = False
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    options.enable_swizzle = os.environ.get("WAVE_ENABLE_SWIZZLE", "1") not in ("0", "false", "False")

    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt_opt00(
        use_stagger=False, shape=shape, block=block
    )

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)

    _run_mxfp_gemm_preshuffle(
        gemm, shape, only_scale=True, output_dtype=torch.bfloat16
    )
    mode = "dynamic" if dynamic else "static"
    sk = f", split-K({splitk})" if splitk else ""
    print(
        f"MXFP GEMM basline 8-wave with scale shuffling ({mode}{sk}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm(
    is_debug=False,
    shape=(2048, 2048, 1024),
    block=(256, 256, 256),
    dynamic=False,
    splitk=None,
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    A and B are read from global memory directly to LDS.

    Note: for dynamic mode, keep block MxN at or below 128x256 or 256x128
    to avoid exceeding shared-memory limits.
    """
    if splitk and block == (256, 256, 256):
        block = (128, 256, 256)
    wave_shape = _get_8wave_shape_from_block(block)
    if splitk:
        gemm, options = get_tagged_splitk_mxfp4_gemm_preshuffle_scales(
            shape,
            num_splits=splitk,
            block_shape=block,
            wave_shape=wave_shape,
            output_type=tkl.bf16,
        )
    else:
        gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
            shape,
            block,
            wave_shape=wave_shape,
            b_address_space=SHARED_ADDRESS_SPACE,
            output_dtype=tkl.bf16,
        )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    options.enable_swizzle = os.environ.get("WAVE_ENABLE_SWIZZLE", "1") not in ("0", "false", "False")
    if os.environ.get("WAVE_VECTORIZED_STORE", "0") not in ("0", "false", "False"):
        _K = shape[2]
        _block_m = block[0]
        if _block_m == 128:
            _M, _N = shape[0], shape[1]
            if _M == 16384 and _N == 16384:
                _mlir_map = {
                    1024: "mxfp4_epilogue_opt_128x128_16384x16384_K1024.mlir",
                    8192: "mxfp4_epilogue_opt_128x128_16384x16384_K8192.mlir",
                }
            elif _M == 8192 and _N == 8192:
                _mlir_map = {
                    1024: "mxfp4_epilogue_opt_128x128_8192x8192_K1024.mlir",
                    2048: "mxfp4_epilogue_opt_128x128_8192x8192_K2048.mlir",
                    4096: "mxfp4_epilogue_opt_128x128_8192x8192_K4096.mlir",
                    8192: "mxfp4_epilogue_opt_128x128_8192x8192_K8192.mlir",
                }
            else:
                _mlir_map = {
                    1024: "mxfp4_epilogue_opt_128x128_K1024.mlir",
                    8192: "mxfp4_epilogue_opt_128x128_K8192.mlir",
                }
        else:
            _M, _N = shape[0], shape[1]
            _epilogue = os.environ.get("WAVE_EPILOGUE_VARIANT", "vecxor")
            if _epilogue == "dpp":
                if _M == 16384 and _N == 16384:
                    _mlir_map = {
                        1024: "mxfp4_epilogue_opt_dpp_256x256_16384x16384_K1024.mlir",
                        8192: "mxfp4_epilogue_opt_dpp_256x256_16384x16384_K8192.mlir",
                    }
                elif _M == 8192 and _N == 8192:
                    _mlir_map = {
                        1024: "mxfp4_epilogue_opt_dpp_256x256_8192x8192_K1024.mlir",
                        8192: "mxfp4_epilogue_opt_dpp_256x256_8192x8192_K8192.mlir",
                    }
                elif _M == 2048 and _N == 2048:
                    _mlir_map = {
                        1024: "mxfp4_epilogue_opt_dpp_256x256_2048x2048_K1024.mlir",
                        8192: "mxfp4_epilogue_opt_dpp_256x256_2048x2048_K8192.mlir",
                    }
                else:
                    raise ValueError(
                        f"WAVE_EPILOGUE_VARIANT=dpp: no DPP MLIR for 256x256 shape ({_M}×{_N}). "
                        f"Supported: 2048×2048, 8192×8192, 16384×16384."
                    )
            elif _M == 16384 and _N == 16384:
                _mlir_map = {
                    1024: "mxfp4_epilogue_opt_256x256_16384x16384_K1024.mlir",
                    8192: "mxfp4_epilogue_opt_256x256_16384x16384_K8192.mlir",
                }
            elif _M == 8192 and _N == 8192:
                _mlir_map = {
                    1024: "mxfp4_epilogue_opt_256x256_8192x8192_K1024.mlir",
                    8192: "mxfp4_epilogue_opt_256x256_8192x8192_K8192.mlir",
                }
            elif _M == 2048 and _N == 2048:
                _mlir_map = {
                    1024: "mxfp4_epilogue_opt_256x256_2048x2048_K1024.mlir",
                    8192: "mxfp4_epilogue_opt_256x256_2048x2048_K8192.mlir",
                }
            else:
                _mlir_map = {
                    1024: "mxfp4_epilogue_opt_256x256_K1024.mlir",
                    2048: "mxfp4_epilogue_opt_256x256_K2048.mlir",
                    4096: "mxfp4_epilogue_opt_256x256_K4096.mlir",
                    8192: "mxfp4_epilogue_opt_256x256x256.mlir",
                }
        _mlir_file = _mlir_map.get(_K)
        if _mlir_file is None:
            raise ValueError(
                f"WAVE_VECTORIZED_STORE: no hand-optimised MLIR for K={_K} "
                f"(block_m={_block_m}). "
                f"Available K values: {list(_mlir_map.keys())}"
            )
        options.override_mlir = (
            pathlib.Path(__file__).parent / "mlir" / _mlir_file
        ).read_text()
        print(
            f"[WAVE] MLIR override loaded: {_mlir_file}"
            f"  epilogue={os.environ.get('WAVE_EPILOGUE_VARIANT', 'vecxor')}"
            f"  block_m={_block_m}",
            file=sys.stderr,
        )
    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]

    # Variant is controlled via env var to make A/B perf sweeps reproducible:
    #   WAVE_MXFP4_VARIANT=opt00|opt0|opt1|opt2
    # Default keeps current behavior.
    variant = os.environ.get("WAVE_MXFP4_VARIANT", "opt00")
    if variant == "opt0":
        schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt0(
            use_stagger=False, shape=shape, block=block
        )
    elif variant == "opt1":
        schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt1(
            use_stagger=True, shape=shape, block=block
        )
    elif variant == "opt2":
        schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt2(
            use_stagger=True, shape=shape, block=block
        )
    else:
        schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt_opt00(
            use_stagger=False, shape=shape, block=block
        )

    # Keep original unroll behavior as default; allow explicit disable for ablation.
    enable_unroll = os.environ.get("WAVE_ENABLE_UNROLL", "1") not in (
        "0",
        "false",
        "False",
    )
    print(
        f"[WAVE] kernel config:"
        f"  WAVE_MXFP4_VARIANT={os.environ.get('WAVE_MXFP4_VARIANT', 'opt00')}"
        f"  WAVE_VECTORIZED_STORE={os.environ.get('WAVE_VECTORIZED_STORE', '0')}"
        f"  WAVE_ENABLE_UNROLL={enable_unroll}"
        f"  WAVE_EPILOGUE_VARIANT={os.environ.get('WAVE_EPILOGUE_VARIANT', 'vecxor')}",
        file=sys.stderr,
    )
    if enable_unroll:
        options.postprocess = """
        module attributes {transform.with_named_sequence} {
            transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
                %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
                transform.loop.unroll %0 { factor = 2 } : !transform.any_op
                transform.yield
            }
        }
        """

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)

    _run_mxfp_gemm_preshuffle(
        gemm, shape, only_scale=True, output_dtype=torch.bfloat16
    )
    mode = "dynamic" if dynamic else "static"
    sk = f", split-K({splitk})" if splitk else ""
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scale shuffling ({mode}{sk}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_Bshuffle(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(256, 192, 256),
    dynamic=False,
    splitk=None,
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    Same for B data. However, prefetching shuffled B directly to VGPR consumes too many VGPRs and causes spilling.
    A is read from global memory directly to LDS.
    """
    if splitk:
        raise NotImplementedError(
            "split-K is not yet supported with B-shuffled ping-pong schedule"
        )
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
        shape, block, wave_shape=wave_shape
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled(use_stagger=True, shape=shape)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scale and B shuffling and B->VGPR ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_Bshuffle_lds(
    is_debug=False, shape=(1024, 1024, 1024), block=(256, 256, 256), dynamic=False
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    B data is preshuffled and loaded to LDS (shared memory), not directly to VGPRs.
    A data is read from global memory directly to LDS.
    """

    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
        shape,
        block,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    
    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds(
        use_stagger=True, shape=shape, block=block
    )
    # options.postprocess = """
    # module attributes {transform.with_named_sequence} {
    #     transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    #         %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    #         transform.loop.unroll %0 { factor = 2 } : !transform.any_op
    #         transform.yield
    #     }
    # }
    # """
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True, output_dtype=torch.bfloat16)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scales and B shuffling and B->LDS ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_Bshuffle_lds_optimized_epilogue(
    is_debug=False, shape=(1024, 1024, 1920), block=(256, 192, 256), dynamic=False
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    B data is preshuffled and loaded to LDS (shared memory), not directly to VGPRs.
    A data is read from global memory directly to LDS.
    A handwritten dynamic MLIR kernel is used which uses an optimized epilogue that uses swizzle and dword stores to global memory (instead of u shorts).
    """

    # This test is wired for the 256x256x256 tile-specific optimized epilogue.
    # if block != (256, 256, 256):
    #     raise ValueError(
    #         "optimized_epilogue test currently supports only block=(256, 256, 256)"
    #     )

    # For quick experiments you can paste an inline override here:
    # xx = """<full IR module>"""
    # options.override_mlir = xx
    # xx = (
    #     pathlib.Path(__file__).parent
    #     / "mlir"
    #     / "mxfp4_epilogue_opt2_256x256x256_xor_candidate.mlir"
    # ).read_text()
    xx = (
        pathlib.Path(__file__).parent
        / "mlir"
        / "mxfp4_epilogue_opt_256x192x256.mlir"
    ).read_text()
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
        shape,
        block,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    options.override_mlir = xx
    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds(
        use_stagger=True, shape=shape, block=block
    )
    options.postprocess = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
            %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
            transform.loop.unroll %0 { factor = 2 } : !transform.any_op
            transform.yield
        }
    }
    """
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True, output_dtype=torch.bfloat16)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scales and B shuffling and B->LDS ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_Bshuffle_lds_transposed(
    is_debug=False, shape=(1024, 1920, 4096), block=(256, 192, 256), dynamic=True
):
    """MXFP4 Dynamic GEMM with transposed computation for wide stores.

    The output is C[M, N] in row-major layout, same as a standard GEMM.
    Internally, the kernel computes C^T = B * A^T by swapping the operand
    roles: the weight matrix B becomes the MFMA left operand and the
    activation matrix A becomes the MFMA right operand. This means C^T
    lives in registers after the MFMAs.

    The MLIR used below is handwritten and contains a custom epilogue that uses
    amdgpu.permlane_swap to exchange data between lanes, packs the f32
    MFMA outputs to bf16 (vector<4xbf16>), and reassembles the C^T
    register data into contiguous C[M, N] rows. This enables wide
    vector<8xbf16> stores (buffer_store_dwordx4) instead of the
    96 scalar stores the compiler would otherwise emit.

    Activation A is preshuffled (since it is now in the MFMA "B" role).
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    """

    mlir_epilogue_opt_256x192x256 = (
        pathlib.Path(__file__).parent
        / "mlir"
        / "mxfp4_transposed_epilogue_opt_256x192x256.mlir"
    ).read_text()

    M_orig, N_orig, K = shape
    shape_t = (N_orig, M_orig, K)
    block_t = (block[1], block[0], block[2])

    wave_shape = _get_8wave_shape_from_block(block_t)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
        shape_t,
        block_t,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    options.override_mlir = mlir_epilogue_opt_256x192x256
    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds(
        use_stagger=True, shape=shape_t, block=block_t
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    _run_mxfp_gemm_preshuffle_transposed(gemm, shape, output_dtype=torch.bfloat16, b_preshuffled=True)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM transposed (C^T=B*A^T) 8-wave ping pong B->LDS ({mode}) test passed!"
    )


def test_dbuf_8wave_mixed_pingpong_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256), splitk=None
):
    """Double-buffered MXFP4 GEMM, 8 waves, with stagger.

    A variant of the ping-pong schedule that hides the latency of the extra
    WorkgroupBarrier required for large shapes. With staggering, the two
    clusters of waves write to LDS at different times.
    When the bus becomes congested, memory operations loaded by the later cluster may not arrive
    in LDS before the other cluster attempts to read from it. In this case,
    we add a second workgroup barrier to fix the timing and prevent incorrect output results.

    This schedule overlaps that barrier with useful work by splitting LDS loads:
      - "Safe" loads: rows this wave wrote itself — readable immediately after
        memory_counter_wait, before the global WorkgroupBarrier.
      - "Dependent" loads: rows written by other waves — deferred until after
        the global WorkgroupBarrier.

    This lets the MFMAs on the safe operands start firing as soon as the
    barrier releases, effectively hiding the second barrier's latency behind
    the early loads and compute.
    """
    if splitk:
        raise NotImplementedError(
            "split-K is not yet supported with the mixed ping-pong schedule"
        )
    gemm, options = get_tagged_mxfp4_gemm(shape, block, wave_shape=(4, 2))
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    schedule = get_mxfp4_dbuf_mixed_pingpong_schedule(use_stagger=True)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 8-wave mixed ping pong test passed!")


def test_dbuf_8wave_mixed_pingpong_shuffle_mxfp_gemm(
    is_debug=False, shape=(16384, 16384, 16384), block=(256, 256, 256), splitk=None
):
    """Like :func:`test_dbuf_8wave_mixed_pingpong_mxfp_gemm` but with A_scale & B_scale
    preshuffled and prefetched to VGPRs.

    Note: preshuffling B and loading it directly to VGPRs combined with prefetching
    consumes too many VGPRs and causes spilling.
    """
    if splitk:
        raise NotImplementedError(
            "split-K is not yet supported with the mixed ping-pong shuffle schedule"
        )
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape, block, wave_shape=(4, 2)
    )

    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    schedule = get_mxfp4_dbuf_mixed_pingpong_shuffle_schedule(use_stagger=True)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, only_scale=True)
    print("MXFP GEMM double-buffer 8-wave mixed ping pong with shuffling test passed!")


def test_dbuf_4wave_mxfp_asymmetric_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256), splitk=None
):
    """Asymmetric-prefetch MXFP4 GEMM: A through LDS (2x prefetch), B direct from global."""
    if splitk:
        raise NotImplementedError(
            "split-K is not yet supported with the asymmetric schedule"
        )
    gemm, options = get_tagged_mxfp4_gemm(
        shape, block, wave_shape=(1, 4), b_address_space=GLOBAL_ADDRESS_SPACE
    )
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric.mlir"
    options.print_mlir = True
    options.dump_binaries = "build/binaries"
    options.dump_intermediates = "build/intermediates"
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.use_water_backend = True
    schedule = get_mxfp4_asymmetric_schedule()

    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM asymmetric-prefetch 4-wave test passed!")


def test_dbuf_4wave_mxfp_preshuffle_b_gemm(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(128, 256, 256),
    eliminate_epilogue=True,
    splitk=None,
):
    """Asymmetric MXFP4 GEMM with preshuffled B data and B scales."""
    if splitk:
        gemm, options = get_tagged_splitk_mxfp4_gemm_preshuffle_b(
            shape, num_splits=splitk, block_shape=block, wave_shape=(1, 4)
        )
    else:
        gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
            shape, block, wave_shape=(1, 4)
        )
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/"
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )

    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    sk = f" split-K({splitk})" if splitk else ""
    print(f"MXFP GEMM preshuffle-B 4-wave{sk} test passed!")


def test_dbuf_4wave_mxfp_asymmetric_gemm_cpp(
    is_debug=False, shape=(1024, 1024, 8192), block=(128, 256, 256), splitk=None
):
    """Asymmetric MXFP4 GEMM using C++ WaveASM backend (no preshuffle)."""
    if splitk:
        raise NotImplementedError(
            "split-K is not yet supported with the asymmetric C++ backend"
        )
    gemm, options = get_tagged_mxfp4_gemm(
        shape, block, wave_shape=(1, 4), b_address_space=GLOBAL_ADDRESS_SPACE
    )
    options.backend = "asm"
    options.wave_runtime = True
    options.dump_intermediates = "build/intermediates"
    schedule = get_mxfp4_asymmetric_schedule()
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM asymmetric 4-wave (WaveASM backend) test passed!")


def test_dbuf_4wave_mxfp_preshuffle_b_gemm_cpp(
    is_debug=False,
    shape=(512, 1024, 8192),  # 4*T0, 4*T1, 8192
    block=(128, 256, 256),
    eliminate_epilogue=True,
    splitk=None,
):
    """Preshuffle-B MXFP4 GEMM using C++ WaveASM backend."""
    if splitk:
        raise NotImplementedError(
            "split-K with WaveASM backend hits register alignment errors in "
            "the assembler; use test_dbuf_4wave_mxfp_preshuffle_b_gemm instead"
        )
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(shape, block, wave_shape=(1, 4))
    options.backend = "asm"
    options.use_buffer_ops = True
    options.wave_runtime = True
    options.use_wave_asm_backend = True
    options.dump_intermediates = "build/intermediates"
    options.eliminate_epilogue = eliminate_epilogue
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print(
        f"MXFP GEMM preshuffle-B 4-wave (WaveASM) epilogue elimination={eliminate_epilogue} PASSED"
    )


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(128, 256, 256),
    eliminate_epilogue=True,
    splitk=None,
):
    """Preshuffle-B MXFP4 GEMM with dynamic M, N, K."""
    if splitk:
        raise NotImplementedError(
            "split-K is not yet supported with the dynamic preshuffle-B schedule"
        )
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(shape, block, wave_shape=(1, 4))
    # Make M, N, K dynamic so the compiler does not specialize on problem size.
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "llvm"
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/"
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print("MXFP GEMM preshuffle-B 4-wave dynamic M, N, K (LLVM backend) test passed!")


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_wide_stores(
    is_debug=False,
    shape=(1024, 3072, 8192),
    block=(256, 192, 256),
    eliminate_epilogue=False,
):
    """Preshuffle-B MXFP4 GEMM with dynamic M, N, K and wide epilogue stores.

    Uses the wide_store variant to swap MFMA operands (B as LHS, A as RHS),
    aligning the accumulator's contiguous values with the output's stride-1
    dimension. The coalesce_wide_stores pass emits v_permlane16_swap_b32
    + buffer_store_dwordx4 (8 bf16 per store) instead of buffer_store_short.
    """
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b_wide_store(
        shape,
        block,
        wave_shape=(2, 2),
        reorder_workgroups=True,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "llvm"
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True, output_dtype=torch.bfloat16)
    print("MXFP GEMM preshuffle-B 4-wave dynamic M, N, K (wide stores) test passed!")


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(128, 256, 256),
    eliminate_epilogue=False,
):
    """Preshuffle-B MXFP4 GEMM with dynamic M, N, K."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, block, wave_shape=(1, 4), reorder_workgroups=False
    )
    # Make M, N, K dynamic so the compiler does not specialize on problem size.
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/"
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print(
        "MXFP GEMM preshuffle-B 4-wave dynamic M, N, K (WaveASM backend) test passed!"
    )


def test_splitk_preshuffle_scales_gemm_cpp(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(128, 128, 256),
    splitk=None,
):
    """Split-K MXFP4 GEMM using C++ WaveASM backend (preshuffled scales)."""
    num_splits = splitk if splitk else 2
    splitk_fn, options = get_tagged_splitk_mxfp4_gemm_preshuffle_scales(
        shape,
        num_splits=num_splits,
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
        block_shape=block,
        wave_shape=(2, 2),
        output_type=tkl.f32,
    )
    options.backend = "asm"
    options.wave_runtime = True
    options.use_wave_asm_backend = True
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, splitk_fn)

    _run_mxfp_gemm_preshuffle(gemm, shape, only_scale=True)
    print("Split-K MXFP4 GEMM (preshuffled scales, WaveASM backend) test passed!")

def test_dbuf_8wave_pingpong_mxfp_gemm_ct_naive(
    is_debug=False,
    shape=(8192, 8192, 1024),
    block=(128, 128, 256),
    dynamic=False,
):
    """Double-buffered MXFP4 GEMM computing C^T = B * A^T (256x256, scales-only preshuffle).

    The kernel receives (w, w_scales, x, x_scales) with swapped roles so that
    it computes C^T of shape (N, M). B (activation x) is NOT preshuffled —
    it goes global → LDS in-kernel, matching the base test_dbuf_8wave_pingpong_mxfp_gemm.
    Only scales are preshuffled.
    """
    M_orig, N_orig, K = shape
    shape_t = (N_orig, M_orig, K)
    block_t = (block[1], block[0], block[2])

    wave_shape = _get_8wave_shape_from_block(block_t)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape_t,
        block_t,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    options.override_mlir = (
        pathlib.Path(__file__).parent
        / "mlir"
        / "mxfp4_transposed_epilogue_opt_128x128_noperm.mlir"
    ).read_text()
    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt2(
        use_stagger=True, shape=shape_t, block=block_t
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)
    _run_mxfp_gemm_preshuffle_transposed(gemm, shape, output_dtype=torch.bfloat16)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM C^T=B*A^T optimized epilogue 8-wave ping pong 128x128 ({mode}) test passed!"
    )

def test_dbuf_8wave_pingpong_mxfp_gemm_ct_naive256(
    is_debug=False,
    shape=(1024, 1024, 1024),
    block=(256, 256, 256),
    dynamic=False,
):
    """Naive (no permlane) 256x256 CT kernel - used for IR generation only."""
    M_orig, N_orig, K = shape
    shape_t = (N_orig, M_orig, K)
    block_t = (block[1], block[0], block[2])
    wave_shape = _get_8wave_shape_from_block(block_t)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape_t, block_t, wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE, output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    # NO override_mlir – emit naive scalar-store epilogue
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt2(
        use_stagger=True, shape=shape_t, block=block_t
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)
    _run_mxfp_gemm_preshuffle_transposed(gemm, shape, output_dtype=torch.bfloat16)
    print("Naive 256x256 CT test passed!")


def test_dbuf_8wave_pingpong_mxfp_gemm_ct(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(256, 256, 256),
    dynamic=False,
):
    """Double-buffered MXFP4 GEMM computing C^T = B * A^T (256x256, scales-only preshuffle).

    The kernel receives (w, w_scales, x, x_scales) with swapped roles so that
    it computes C^T of shape (N, M). B (activation x) is NOT preshuffled —
    it goes global → LDS in-kernel, matching the base test_dbuf_8wave_pingpong_mxfp_gemm.
    Only scales are preshuffled.
    """
    M_orig, N_orig, K = shape
    shape_t = (N_orig, M_orig, K)
    block_t = (block[1], block[0], block[2])

    wave_shape = _get_8wave_shape_from_block(block_t)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape_t,
        block_t,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    _mlir_files = {
        (1024, 1024): "mxfp4_transposed_epilogue_opt_256x256_K8192.mlir",
        (2048, 2048): "mxfp4_transposed_epilogue_opt_256x256_2048x2048_K8192.mlir",
        (8192, 8192): "mxfp4_transposed_epilogue_opt_256x256_8192x8192_K8192.mlir",
        (16384, 16384): "mxfp4_transposed_epilogue_opt_256x256_16384x16384_K8192.mlir",
    }
    _mlir_file = _mlir_files.get((M_orig, N_orig))
    if _mlir_file:
        options.override_mlir = (
            pathlib.Path(__file__).parent / "mlir" / _mlir_file
        ).read_text()
    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt2(
        use_stagger=True, shape=shape_t, block=block_t
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)
    _run_mxfp_gemm_preshuffle_transposed(gemm, shape, output_dtype=torch.bfloat16)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM C^T=B*A^T optimized epilogue 8-wave ping pong 256x256 ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_ct_K1024(
    is_debug=False,
    shape=(1024, 1024, 1024),
    block=(256, 256, 256),
    dynamic=False,
):
    """Same as test_dbuf_8wave_pingpong_mxfp_gemm_ct but with K=1024."""
    M_orig, N_orig, K = shape
    shape_t = (N_orig, M_orig, K)
    block_t = (block[1], block[0], block[2])

    wave_shape = _get_8wave_shape_from_block(block_t)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape_t,
        block_t,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    _mlir_files = {
        (1024, 1024): "mxfp4_transposed_epilogue_opt_256x256_K1024.mlir",
        (2048, 2048): "mxfp4_transposed_epilogue_opt_256x256_2048x2048_K1024.mlir",
        (8192, 8192): "mxfp4_transposed_epilogue_opt_256x256_8192x8192_K1024.mlir",
        (16384, 16384): "mxfp4_transposed_epilogue_opt_256x256_16384x16384_K1024.mlir",
    }
    _mlir_file = _mlir_files.get((M_orig, N_orig))
    if _mlir_file:
        options.override_mlir = (
            pathlib.Path(__file__).parent / "mlir" / _mlir_file
        ).read_text()
    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt2(
        use_stagger=True, shape=shape_t, block=block_t
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)
    _run_mxfp_gemm_preshuffle_transposed(gemm, shape, output_dtype=torch.bfloat16)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM C^T=B*A^T optimized epilogue 8-wave ping pong 256x256 K=1024 ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_ct_K2048(
    is_debug=False,
    shape=(1024, 1024, 2048),
    block=(256, 256, 256),
    dynamic=False,
):
    """Same as test_dbuf_8wave_pingpong_mxfp_gemm_ct but with K=2048."""
    M_orig, N_orig, K = shape
    shape_t = (N_orig, M_orig, K)
    block_t = (block[1], block[0], block[2])

    wave_shape = _get_8wave_shape_from_block(block_t)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape_t,
        block_t,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    options.override_mlir = (
        pathlib.Path(__file__).parent
        / "mlir"
        / "mxfp4_transposed_epilogue_opt_256x256_K2048.mlir"
    ).read_text()
    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt2(
        use_stagger=True, shape=shape_t, block=block_t
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)
    _run_mxfp_gemm_preshuffle_transposed(gemm, shape, output_dtype=torch.bfloat16)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM C^T=B*A^T optimized epilogue 8-wave ping pong 256x256 K=2048 ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_ct_K4096(
    is_debug=False,
    shape=(1024, 1024, 4096),
    block=(256, 256, 256),
    dynamic=False,
):
    """Same as test_dbuf_8wave_pingpong_mxfp_gemm_ct but with K=4096."""
    M_orig, N_orig, K = shape
    shape_t = (N_orig, M_orig, K)
    block_t = (block[1], block[0], block[2])

    wave_shape = _get_8wave_shape_from_block(block_t)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape_t,
        block_t,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    options.override_mlir = (
        pathlib.Path(__file__).parent
        / "mlir"
        / "mxfp4_transposed_epilogue_opt_256x256_K4096.mlir"
    ).read_text()
    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt2(
        use_stagger=True, shape=shape_t, block=block_t
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)
    _run_mxfp_gemm_preshuffle_transposed(gemm, shape, output_dtype=torch.bfloat16)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM C^T=B*A^T optimized epilogue 8-wave ping pong 256x256 K=4096 ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_128x128(
    is_debug=False,
    shape=(2048, 2048, 1024),
    block=(128, 128, 256),
    dynamic=False,
):
    """Double-buffered MXFP4 GEMM with 128×128 tile block (8 waves, ping-pong).

    Uses the XOR-vectorized optimized epilogue from a hand-written MLIR override.
    Set WAVE_VECTORIZED_STORE=1 to activate the MLIR override; without it the
    kernel falls back to the naive scalar-store epilogue.

    Supported shapes (set via --shape M,N,K):
      2048,2048,1024  (default)
      2048,2048,8192

    Example:
        WAVE_MXFP4_VARIANT=opt2 WAVE_VECTORIZED_STORE=1 \\
            python 7.1_schedule.py --test test_dbuf_8wave_pingpong_mxfp_gemm_128x128 \\
            --shape 2048,2048,1024 --block 128,128,256
    """
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape,
        block,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    options.enable_swizzle = True

    if os.environ.get("WAVE_VECTORIZED_STORE", "0") not in ("0", "false", "False"):
        _K = shape[2]
        _M, _N = shape[0], shape[1]
        _epilogue = os.environ.get("WAVE_EPILOGUE_VARIANT", "vecxor")
        if _epilogue == "dpp":
            if _M == 16384 and _N == 16384:
                _mlir_map = {
                    1024: "mxfp4_epilogue_opt_dpp_128x128_16384x16384_K1024.mlir",
                    8192: "mxfp4_epilogue_opt_dpp_128x128_16384x16384_K8192.mlir",
                }
            else:
                _mlir_map = {
                    1024: "mxfp4_epilogue_opt_dpp_128x128_K1024.mlir",
                    8192: "mxfp4_epilogue_opt_dpp_128x128_K8192.mlir",
                }
        else:
            _mlir_map = {
                1024: "mxfp4_epilogue_opt_128x128_K1024.mlir",
                8192: "mxfp4_epilogue_opt_128x128_K8192.mlir",
            }
        _mlir_file = _mlir_map.get(_K)
        if _mlir_file is None:
            raise ValueError(
                f"WAVE_VECTORIZED_STORE: no hand-optimised 128x128 MLIR for K={_K} "
                f"(epilogue={_epilogue}). "
                f"Available K values: {list(_mlir_map.keys())}"
            )
        options.override_mlir = (
            pathlib.Path(__file__).parent / "mlir" / _mlir_file
        ).read_text()

    variant = os.environ.get("WAVE_MXFP4_VARIANT", "opt2")
    if variant == "opt0":
        schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt0(
            use_stagger=False, shape=shape, block=block
        )
    elif variant == "opt1":
        schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt1(
            use_stagger=True, shape=shape, block=block
        )
    elif variant == "opt2":
        schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt2(
            use_stagger=True, shape=shape, block=block
        )
    else:
        schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt_opt00(
            use_stagger=False, shape=shape, block=block
        )

    enable_unroll = os.environ.get("WAVE_ENABLE_UNROLL", "1") not in (
        "0",
        "false",
        "False",
    )
    if enable_unroll:
        options.postprocess = """
        module attributes {transform.with_named_sequence} {
            transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
                %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
                transform.loop.unroll %0 { factor = 2 } : !transform.any_op
                transform.yield
            }
        }
        """

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)

    _run_mxfp_gemm_preshuffle(
        gemm, shape, only_scale=True, output_dtype=torch.bfloat16
    )
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM 128x128 XOR-epilogue 8-wave ping-pong ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_ct_128x128(
    is_debug=False,
    shape=(2048, 2048, 1024),
    block=(128, 128, 256),
    dynamic=False,
):
    """Transposed MXFP4 GEMM (C^T = B·A^T) with 128×128 tile, permlane + bf16-shuffle epilogue.

    Uses the hand-optimized transposed epilogue with bf16-shuffle optimization:
    accumulators are truncated f32→bf16, packed into i32, then 2×i32 permlane_swaps
    are performed (halving shuffle count vs. naive 4×f32 permlane approach).

    Block coordinates are swapped so the kernel computes (N_orig × M_orig) which
    is the transposed output. The test validates the result against a reference
    computed from the non-transposed path.

    Supported shapes (set via --shape M,N,K):
      2048,2048,1024  (default)
      2048,2048,8192

    Example:
        WAVE_MXFP4_VARIANT=opt2 \\
            python 7.1_schedule.py --test test_dbuf_8wave_pingpong_mxfp_gemm_ct_128x128 \\
            --shape 2048,2048,1024 --block 128,128,256
    """
    M_orig, N_orig, K = shape
    shape_t = (N_orig, M_orig, K)
    block_t = (block[1], block[0], block[2])

    wave_shape = _get_8wave_shape_from_block(block_t)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape_t,
        block_t,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    options.enable_swizzle = True

    _K = shape[2]
    _M, _N = shape[0], shape[1]
    _mlir_map_small = {
        1024: "mxfp4_transposed_epilogue_opt_128x128_K1024.mlir",
        8192: "mxfp4_transposed_epilogue_opt_128x128_K8192.mlir",
    }
    _mlir_map_8192 = {
        1024: "mxfp4_transposed_epilogue_opt_128x128_8192x8192_K1024.mlir",
        8192: "mxfp4_transposed_epilogue_opt_128x128_8192x8192_K8192.mlir",
    }
    _mlir_map_16384 = {
        1024: "mxfp4_transposed_epilogue_opt_128x128_16384x16384_K1024.mlir",
        8192: "mxfp4_transposed_epilogue_opt_128x128_16384x16384_K8192.mlir",
    }
    if _M == 16384 and _N == 16384:
        _mlir_map = _mlir_map_16384
    elif _M == 8192 and _N == 8192:
        _mlir_map = _mlir_map_8192
    else:
        _mlir_map = _mlir_map_small
    _mlir_file = _mlir_map.get(_K)
    if _mlir_file is None:
        raise ValueError(
            f"No hand-optimised transposed 128x128 MLIR for K={_K}. "
            f"Available K values: {list(_mlir_map.keys())}"
        )
    options.override_mlir = (
        pathlib.Path(__file__).parent / "mlir" / _mlir_file
    ).read_text()

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]

    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt2(
        use_stagger=True, shape=shape_t, block=block_t
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)

    _run_mxfp_gemm_preshuffle_transposed(gemm, shape, output_dtype=torch.bfloat16)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM 128x128 transposed permlane+bf16-shuffle epilogue 8-wave ping-pong ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_ct_128x128_v2(
    is_debug=False,
    shape=(2048, 2048, 1024),
    block=(128, 128, 256),
    dynamic=False,
):
    """Like test_dbuf_8wave_pingpong_mxfp_gemm_ct_128x128 but uses v2 MLIR files
    that call amdgpu.permlane_swap directly on vector<4xbf16> instead of the
    scalar i32 bitcast approach.
    """
    M_orig, N_orig, K = shape
    shape_t = (N_orig, M_orig, K)
    block_t = (block[1], block[0], block[2])

    wave_shape = _get_8wave_shape_from_block(block_t)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape_t,
        block_t,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.wave_runtime = True
    options.enable_swizzle = True

    _K = shape[2]
    _M, _N = shape[0], shape[1]
    _mlir_map_small = {
        1024: "mxfp4_transposed_epilogue_opt_128x128_K1024_v2.mlir",
        8192: "mxfp4_transposed_epilogue_opt_128x128_K8192_v2.mlir",
    }
    _mlir_map_8192 = {
        1024: "mxfp4_transposed_epilogue_opt_128x128_8192x8192_K1024.mlir",
        8192: "mxfp4_transposed_epilogue_opt_128x128_8192x8192_K8192.mlir",
    }
    _mlir_map_16384 = {
        1024: "mxfp4_transposed_epilogue_opt_128x128_16384x16384_K1024_v2.mlir",
        8192: "mxfp4_transposed_epilogue_opt_128x128_16384x16384_K8192_v2.mlir",
    }
    if _M == 16384 and _N == 16384:
        _mlir_map = _mlir_map_16384
    elif _M == 8192 and _N == 8192:
        _mlir_map = _mlir_map_8192
    else:
        _mlir_map = _mlir_map_small
    _mlir_file = _mlir_map.get(_K)
    if _mlir_file is None:
        raise ValueError(
            f"No v2 transposed 128x128 MLIR for K={_K}. "
            f"Available K values: {list(_mlir_map.keys())}"
        )
    options.override_mlir = (
        pathlib.Path(__file__).parent / "mlir" / _mlir_file
    ).read_text()

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]

    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds_opt2(
        use_stagger=True, shape=shape_t, block=block_t
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    print(gemm.asm)

    _run_mxfp_gemm_preshuffle_transposed(gemm, shape, output_dtype=torch.bfloat16)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM 128x128 transposed permlane_vec v2 epilogue 8-wave ping-pong ({mode}) test passed!"
    )


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(
        args.test,
        globals(),
        args.debug,
        args.repeat,
        args.shape,
        args.block,
        args.eliminate_epilogue,
        args.splitk,
    )
    exit(0 if success else 1)
