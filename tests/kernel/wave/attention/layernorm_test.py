# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


## fails-- affine map does not cover whole input tensor here during sum reduction

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


@require_e2e
@pytest.mark.parametrize("shape", [(16, 16)])
def test_scatter_add(shape, request):
    run_bench = request.config.getoption("--runperf")

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    B = tkl.sym.B
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: 64},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 1)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i, N: j},
    )

    @tkw.wave(constraints)
    def layernorm(
        input: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        gamma: tkl.Memory[N, ADDRESS_SPACE, tkl.f32],
        beta: tkl.Memory[N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, tkl.f32](64.0)

        ## Compute mean for each datapoint
        input_reg = tkw.read(input, elements_per_thread=ELEMS_PER_THREAD)
        sum = tkw.sum(input_reg, dim=N)
        # mean= sum / c_reg
        tkw.write(sum, output, elements_per_thread=ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: shape[0],
            BLOCK_N: shape[1],
            ELEMS_PER_THREAD: 1,
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        print_mlir=True,
    )
    options = set_default_run_config(options)
    test_fn = wave_compile(options, layernorm)

    input = (
        device_arange(shape[0] * shape[1], dtype=torch.float32)
        .reshape(shape[0], shape[1])
        .contiguous()
    )

    gamma = device_ones(shape[0], dtype=torch.float32).contiguous()

    beta = device_ones(shape[0], dtype=torch.float32).contiguous()

    output = device_zeros(shape[0], dtype=torch.float32).contiguous()

    test_fn(input, gamma, beta, output)
    print("input", input.cpu())
    print("output", output.cpu())

    # torch.testing.assert_close(output, torch_output)
