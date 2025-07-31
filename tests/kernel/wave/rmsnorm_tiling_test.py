# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from torch.nn import functional as F
import math
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
    device_ones,
    device_arange,
    device_randint,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
import os
from torch.testing import assert_close
from .common.utils import (
    require_e2e,
)
from wave_lang.kernel.wave.constraints import MMAType, MMAOperand, GenericDot
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions


K1 = tkl.sym.K1
M = tkl.sym.M
N = tkl.sym.N

BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K1 = tkl.sym.BLOCK_K1
EMB_SIZE = tkl.sym.EMB_SIZE
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD


@require_e2e
def test():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    num_waves = 4
    wave_size = 64
    BLOCK_N = N // 1
    tiling_factor = 4
    ELEMS_PER_WAVE = N // num_waves
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: ELEMS_PER_THREAD * wave_size},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / num_waves)]
    constraints += [tkw.TilingConstraint(N, (BLOCK_N) / tiling_factor)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        gamma: tkl.Memory[N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        length_embedding = tkl.Register[M, tkl.f16](EMB_SIZE)

        init_red = tkl.Register[M, tkl.f16](0.0)

        @tkw.iterate(N, init_args=[init_red])
        def repeat(
            partial_red: tkl.Register[M, tkl.f16],
        ) -> tkl.Register[M, tkl.f16]:
            lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
            lhs_sq = lhs * lhs
            partial_red = tkw.sum(lhs_sq, partial_red, dim=N)
            return partial_red

        result = repeat / length_embedding
        rms = tkw.sqrt(result)
        rms_broad = tkw.broadcast(rms, [M, N])

        lhs2 = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        a_scaled = lhs2 / rms_broad
        # gamma_reg = tkw.read(gamma)
        # gamma_broad = tkw.broadcast(gamma_reg, [M, N])
        # output = a_scaled * gamma_broad

        tkw.write(a_scaled, c)

    shape = (1, 5120)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            EMB_SIZE: shape[1],
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        print_ir_after=["set_thread_dependent_index_from_reduce", "expand_graph"],
        print_ir_before=["set_thread_dependent_index_from_reduce", "expand graph"],
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)
    print(test.asm)

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float16)
    gamma = device_randn(shape[1], dtype=torch.float16)
    c = device_zeros(shape, dtype=torch.float16)
    test(a, gamma, c)
    print(test.asm)
    print(a.cpu())
    print(c.cpu())

    # rms = torch.sqrt(torch.mean(a * a, dim=-1, keepdim=True))
    # ref = (a / rms) * gamma
    # torch.testing.assert_close(ref, c, atol=1e-02, rtol=1e-05)
