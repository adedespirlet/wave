# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


## works BUT performs out of bounds memory accesses! Input tensor [256;256] is loaded with vector.load[block_ID, map]
##with #map = affine_map<()[s0] -> (s0 * 4)> and with 2 waves== 128 threads
## so at some point vector.load input[blockID; >257] when threads > 64 are executed ==> OOB , should not exceed 256!

import pytest
import torch
import math
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_empty,
    device_arange,
    device_randint,
    device_ones,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
import os
from torch.nn import functional as F
from ..common.utils import (
    require_e2e,
    require_cdna3,
    dump_generated_mlir,
    perf_test,
    param_bool,
    enable_scheduling_barriers,
)

from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
from ..common.shapes import get_test_shapes


@require_e2e
# @pytest.mark.parametrize("shape", get_test_shapes("test_block_reduce"))
@pytest.mark.parametrize("shape", [(256, 256, 16)])
def test_reduce_sum(shape, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size
    ELEMS_PER_THREAD = BLOCK_N // wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 1)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        # rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
        # res = lhs * rhs
        flattened_reg = tkw.reshape(lhs, target_vector_shape={N: N * K})

        res = tkw.sum(flattened_reg, dim=N)

        tkw.write(res, c, elements_per_thread=1)

    print(shape)
    torch.manual_seed(1)
    # a = device_randn(shape, dtype=torch.float16)
    # b = device_randn(shape, dtype=torch.float16)
    c = device_zeros(shape[0], dtype=torch.float16)

    # a = (
    #     device_arange(shape[0] * shape[1], dtype=torch.float16)
    #     .reshape(shape[0], shape[1])
    #     .contiguous()
    # )
    a = device_ones(shape, dtype=torch.float16)
    b = device_ones(shape, dtype=torch.float16)

    ref = torch.sum((a * b), dim=-1)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        print_mlir=True,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b, c)
    print("input", a.cpu())
    print("output", c.cpu())
    print("ref", ref)
    torch.testing.assert_close(ref, c, atol=0.1, rtol=1e-05)
