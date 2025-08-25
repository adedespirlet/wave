import triton
import triton.language as tl
import torch
import itertools

import triton.compiler as tc


   # Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from torch.nn import functional as F
import wave_lang.kernel as tk
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)

from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile

from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.lang import DataType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
    enable_scheduling_barriers,
    dump_generated_mlir,
    check_individual_kernels,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType

@triton.jit
def matmul_abt_kernel(
    C, A, B,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)         # (BM,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)         # (BN,)
    offs_k = tl.arange(0, BLOCK_K)                           # (BK,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K                          

        # A tile pointers: (BM, BK)
        A_ptrs = (A
                  + offs_m[:, None] * stride_am              
                  + (k0 + offs_k)[None, :] * stride_ak)     
        A_tile = tl.load(A_ptrs,
                         mask=(offs_m[:, None] < M) & k_mask[None, :],
                         other=0.0)

        # B tile pointers: (BN, BK)  <-- FIXED SHAPES
        B_ptrs = (B
                  + offs_n[:, None] * stride_bn            
                  + (k0 + offs_k)[None, :] * stride_bk)    
        B_tile_NK = tl.load(B_ptrs,
                            mask=(offs_n[:, None] < N) & k_mask[None, :],
                            other=0.0)                        
        B_tile = tl.trans(B_tile_NK)                          

        acc += tl.dot(A_tile, B_tile)

    C_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul_abt(A: torch.Tensor, B: torch.Tensor, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3):
    """
    C = A @ B.T
    A: (M,K)  B: (N,K)  -> C: (M,N)
    """
    assert A.ndim == 2 and B.ndim == 2
    M, K = A.shape
    N, Kb = B.shape
    assert K == Kb, "K mismatch"
    # result in fp32 to match your Wave kernel’s accum type
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_abt_kernel[grid](
        C, A, B,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C


def get_wave_gemm(shape: tuple[int, int, int],dtype: torch.dtype, dynamic_dims: bool | tuple[bool, bool, bool],mfma_variant: MMAType):
    
    m= shape[0]
    n= shape[1]
    k= shape[2]
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    wave_dtype = torch_dtype_to_wave(dtype)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    # With dynamic dimensions, we need to add an assumption on how big
    # the iterate dimension is to determine whether we can schedule or not.
    if dynamic_dims[2]:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, wave_dtype],
        b: tkl.Memory[N, K, ADDRESS_SPACE, wave_dtype],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)


        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc
        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
    }
    hyperparams.update(get_default_scheduling_params())


    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
    )
    
    options = set_default_run_config(options)
    return wave_compile(options, gemm)
    

def torch_compile_matmul(a, b):
    # Simple wrapper around matmul
    def fn(x, y):
        return torch.mm(x, y.t())   # same convention as your GEMM test

    compiled_fn = torch.compile(fn)     # JIT-compile it with TorchDynamo+Inductor
    return compiled_fn(a, b)



def calculate_diff_gemm(M, N, K, dtype=torch.bfloat16):
    # Random test matrices
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(N, K, dtype=dtype, device="cuda")  # careful: ABᵀ → shape (M,N)
    C = torch.empty((M, N), dtype=torch.float32, device="cuda")

    # ---- WAVE ----
    wave_kernel = get_wave_gemm((M,N,K), dtype, [False,False,False],MMAType.F32_32x32x16_K8_F16)  # <- your Wave GEMM builder
    wave_kernel(A.clone(), B.clone(),C) 

    # ---- TRITON ----
    output_triton = triton_matmul_abt(A.clone(), B.clone())  

    # ---- TORCH (reference, uses rocBLAS) ----
    # GEMM ABᵀ → (M,K) * (N,K)ᵀ = (M,N)
    output_torch = torch.matmul(A, B.t())

    # ---- Compare ----
    print(f"Wave output shape:   {C.shape}")
    print(f"Triton output shape: {output_triton.shape}")
    print(f"Torch output shape:  {output_torch.shape}")

    if torch.allclose(C, output_torch.to(torch.float32), atol=1e-2, rtol=1e-2) and \
       torch.allclose(output_triton.to(torch.float32), output_torch.to(torch.float32), atol=1e-2, rtol=1e-2):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")
        max_diff_wave = (C - output_torch).abs().max().item()
        max_diff_triton = (output_triton - output_torch).abs().max().item()
        print(f"Max diff Wave vs Torch:   {max_diff_wave}")
        print(f"Max diff Triton vs Torch: {max_diff_triton}")


# Pick a grid to match what you want to compare with Wave
M_vals = [64,128,256]
N_vals = [128 ,128,256]
K_vals = [511,511,511 ]
configs = list(itertools.product(M_vals, N_vals, K_vals))

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["wave","triton", "torch_compile","torch"],
        line_names=["wave","Triton", "torch_compile","Torch.mm"],
        styles=[("blue","-"), ("red","-"),("green","-"),("orange","-")],
        ylabel="ms",
        plot_name="gemm-abt-performance",
        args={},
    )
)
def bench(M, N, K, provider):
    dtype = torch.bfloat16
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(N, K, dtype=dtype, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        # warmup
        triton_matmul_abt(A, B)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_matmul_abt(A, B),
            quantiles=quantiles,
        )
    elif provider == "wave":
        # plug your compiled wave GEMM here; it should compute C in fp32
        wave_gemm = get_wave_gemm( (M,N,K), dtype, [False,False,False],MMAType.F32_32x32x16_K8_F16)
        C = torch.empty((M, N), dtype=torch.float32, device="cuda")
        _ = wave_gemm(A, B, C)   # warmup; expect A(M,K), B(N,K), C(M,N)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: wave_gemm(A, B, C),
            quantiles=quantiles,
        )
    elif provider == "torch_compile":
        ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: torch_compile_matmul(A, B),
        quantiles=quantiles,
    )
    elif provider == "torch":
        ref = lambda: torch.mm(A, B.t()).float()
        _ = ref()
        ms, min_ms, max_ms = triton.testing.do_bench(ref, quantiles=quantiles)

    else:
        raise ValueError(provider)

    return ms, min_ms, max_ms

if __name__ == "__main__":

    # perf sweep
    bench.run(print_data=True, show_plots=True)
    
    calculate_diff_gemm(64, 128, 511)
    


