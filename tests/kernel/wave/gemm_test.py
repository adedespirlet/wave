   # Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest
import torch
from torch.nn import functional as F
import wave_lang.kernel as tk
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from .common.utils import (
    require_e2e,
)
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

@pytest.mark.parametrize(
    "shape,dtype,dynamic_dims,mfma_variant",
    [
        #((1, 5120), torch.bfloat16, False, MMAType.DEFAULT),
        ((64, 128, 511),torch.bfloat16, [False,False,False], MMAType.F32_16x16x16_F16 ), 
        # ((64, 1024, 511), torch.float16, (False, True), MMAType.MFMA_32x32), 
        # ((64, 128, 511), torch.float16, (False, True), MMAType.MFMA_32x32), 
    ],
)
@require_e2e
def test_gemm(shape: tuple[int, int, int],dtype: torch.dtype,
    dynamic_dims: bool | tuple[bool, bool, bool],
    mfma_variant: MMAType):
    
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

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, wave_dtype],
        b: tkl.Memory[N, K, ADDRESS_SPACE, wave_dtype],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, dtype]
            a_reg = tkw.read(a)
            # b_reg: tkw.Register[N, K, dtype]
            b_reg = tkw.read(b)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
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
        print_mlir=True,
    )
    
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)


    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(n, k, dtype=dtype, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    gemm(a,b,c)
    # Verify the result using PyTorch's matmul
    expected = torch.matmul(a, b.t()).to(c.dtype)

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2), \
        f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")

