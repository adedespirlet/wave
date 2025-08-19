import itertools
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from torch import nn
from vllm import _custom_ops as vllm_ops


import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from sympy import Integer, log, ceiling, Max, Min

@triton.jit
def fused_rmsnorm_kernel(
    output_ptr,
    activ_ptr,
    weight_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    w1_ = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a_rms = a / rms * w1

    tl.store(
        output_ptr + input_start + offsets,
        a_rms,  # implicitly casts to output dtype here
        mask=mask,
    )


def fused_rmsnorm(x, weight, eps: float = 1e-6, autotune=False, inplace=False):
    assert len(x.shape) == 2
    if inplace:
        output = x
    else:
        output = torch.empty_like(x)
    bs, hidden_dim = x.shape
    max_warps = 16
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }
 
    compiled_kernel = fused_rmsnorm_kernel[(bs,)](
        output, x, weight, eps=eps, hidden_dim=hidden_dim, **config
    )
    # print(compiled_kernel.asm['amdgcn'])
    return output


def get_rmsnorm_wave_rsqrt(shape, eps: float = 1e-6):
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = sympy.Min(sympy.Max(64 * 4, N / 16), N)
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: BLOCK_N },
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def rmsnorm(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
        weight: tkl.Memory[N, ADDRESS_SPACE, tkl.bf16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
    ):
        length_embedding = tkl.Register[M, tkl.f32](N)
        eps_reg = tkl.Register[M, tkl.f32](eps)
        a_reg = tkw.read(a)
        a_reg = tkw.cast(a_reg, tkl.f32)
        mean = tkw.sum(a_reg * a_reg, dim=N, block=True) / length_embedding + eps_reg
        rms = tkw.rsqrt(mean)
        rms_broad = tkw.broadcast(rms, [M, N])
        a_scaled = a_reg * rms_broad
        w_reg = tkw.read(weight)
        w_reg = tkw.cast(w_reg, tkl.f32)
        w_broad = tkw.broadcast(w_reg, [M, N])
        output = a_scaled * w_broad
        output = tkw.cast(output, tkl.bf16)
        tkw.write(output, c)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    return wave_compile(options, rmsnorm)

    
    
def get_rmsnorm_wave_block_false(shape, eps: float = 1e-6):
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = sympy.Min(sympy.Max(64 * 4, N / 16), N)
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    

    def next_power_of_2(expr):
        expr = Integer(expr) if isinstance(expr, int) else expr
        expr = Max(Integer(1), expr)
        exponent = ceiling(log(expr, 2))    
        return Integer(2)**exponent

    threads_per_wave = 64
    min_num_waves = 4
    max_num_waves = 16

    #BLOCK_N = Min(Max(threads_per_wave * min_num_waves, next_power_of_2(ceiling(N / max_num_waves))), next_power_of_2(N))
        
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: BLOCK_N },
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def rmsnorm(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
        weight: tkl.Memory[N, ADDRESS_SPACE, tkl.bf16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
    ):
        length_embedding = tkl.Register[M, tkl.f32](N)
        eps_reg = tkl.Register[M, tkl.f32](eps)
        a_reg = tkw.read(a)
        a_reg = tkw.cast(a_reg, tkl.f32)
        mean = tkw.sum(a_reg * a_reg, dim=N, block=False) / length_embedding + eps_reg
        rms = tkw.rsqrt(mean)
        rms_broad = tkw.broadcast(rms, [M, N])
        a_scaled = a_reg * rms_broad
        w_reg = tkw.read(weight)
        w_reg = tkw.cast(w_reg, tkl.f32)
        w_broad = tkw.broadcast(w_reg, [M, N])
        output = a_scaled * w_broad
        output = tkw.cast(output, tkl.bf16)
        tkw.write(output, c)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        use_buffer_load_ops=False,
        use_buffer_store_ops=False,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    return wave_compile(options, rmsnorm)


def get_rmsnorm_wave_initial(shape, eps: float = 1e-6):
    
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    EMB_SIZE = tkl.sym.EMB_SIZE
    TOKENS_PER_WK = tkl.sym.TOKENS_PER_WK

    num_waves = 4
    wave_size = 64
    BLOCK_N = N
    BLOCK_M = TOKENS_PER_WK

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

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
        weight: tkl.Memory[N, ADDRESS_SPACE, tkl.bf16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
    ):
        length_embedding = tkl.Register[M, tkl.f32](N)
        eps_reg = tkl.Register[M, tkl.f32](eps)
        a_reg = tkw.read(a)
        a_reg = tkw.cast(a_reg, tkl.f32)
        mean = tkw.sum(a_reg * a_reg, dim=N, block=True) / length_embedding + eps_reg
        rms = tkw.sqrt(mean)
        rms_broad = tkw.broadcast(rms, [M, N])
        a_scaled = a_reg / rms_broad
        w_reg = tkw.read(weight)
        w_reg = tkw.cast(w_reg, tkl.f32)
        w_broad = tkw.broadcast(w_reg, [M, N])
        output = a_scaled * w_broad
        output = tkw.cast(output, tkl.bf16)
        tkw.write(output, c)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            TOKENS_PER_WK: 1,
            EMB_SIZE: shape[1],
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    return wave_compile(options, test)

def get_rmsnorm_wave_initial_dpp(shape, eps: float = 1e-6):
    
    override_mlir_str128_initialdpp= """
    #map = affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 1024)>
    #map1 = affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 1024 + 256)>
    #map2 = affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 1024 + 512)>
    #map3 = affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 1024 + 768)>
    #map4 = affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 1024 + 1024)>
    #map5 = affine_map<()[s0] -> (s0 mod 64)>
    #map6 = affine_map<()[s0] -> ((s0 floordiv 64) mod 4)>
    #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 64>
    module attributes {transform.with_named_sequence} {
    stream.executable private @test {
        stream.executable.export public @test workgroups() -> (index, index, index) {
        %c1 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        stream.return %c1, %c128, %c1 : index, index, index
        }
        builtin.module {
        func.func @test(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
            %cst = arith.constant 9.99999997E-7 : f32
            %cst_0 = arith.constant 5.120000e+03 : f32
            %c0_i32 = arith.constant 0 : i32
            %c32_i32 = arith.constant 32 : i32
            %c16_i32 = arith.constant 16 : i32
            %c8_i32 = arith.constant 8 : i32
            %c4_i32 = arith.constant 4 : i32
            %c2_i32 = arith.constant 2 : i32
            %c64_i32 = arith.constant 64 : i32
            %c1_i32 = arith.constant 1 : i32
            %c2147483645_i32 = arith.constant 2147483645 : i32
            %c1073741822 = arith.constant 1073741822 : index
            %c0 = arith.constant 0 : index
            %c5120 = arith.constant 5120 : index
            %block_id_y = gpu.block_id  y upper_bound 128
            %thread_id_x = gpu.thread_id  x upper_bound 256
            %alloc = memref.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
            %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<128x5120xbf16, strided<[5120, 1], offset: ?>>
            %1 = affine.apply #map()[%thread_id_x]
            %2 = arith.muli %block_id_y, %c5120 overflow<nsw> : index
            %3 = arith.addi %2, %1 overflow<nsw> : index
            %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [%c1073741822], strides: [1] : memref<128x5120xbf16, strided<[5120, 1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
            %4 = amdgpu.fat_raw_buffer_cast %reinterpret_cast validBytes(%c2147483645_i32) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
            %5 = vector.load %4[%3] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %6 = affine.apply #map1()[%thread_id_x]
            %7 = arith.addi %2, %6 overflow<nsw> : index
            %8 = vector.load %4[%7] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %9 = affine.apply #map2()[%thread_id_x]
            %10 = arith.addi %2, %9 overflow<nsw> : index
            %11 = vector.load %4[%10] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %12 = affine.apply #map3()[%thread_id_x]
            %13 = arith.addi %2, %12 overflow<nsw> : index
            %14 = vector.load %4[%13] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %15 = affine.apply #map4()[%thread_id_x]
            %16 = arith.addi %2, %15 overflow<nsw> : index
            %17 = vector.load %4[%16] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %18 = arith.extf %5 : vector<4xbf16> to vector<4xf32>
            %19 = arith.extf %8 : vector<4xbf16> to vector<4xf32>
            %20 = arith.extf %11 : vector<4xbf16> to vector<4xf32>
            %21 = arith.extf %14 : vector<4xbf16> to vector<4xf32>
            %22 = arith.extf %17 : vector<4xbf16> to vector<4xf32>
            %23 = arith.mulf %18, %18 : vector<4xf32>
            %24 = arith.mulf %19, %19 : vector<4xf32>
            %25 = arith.mulf %20, %20 : vector<4xf32>
            %26 = arith.mulf %21, %21 : vector<4xf32>
            %27 = arith.mulf %22, %22 : vector<4xf32>
            %28 = arith.addf %23, %24 : vector<4xf32>
            %29 = arith.addf %28, %25 : vector<4xf32>
            %30 = arith.addf %29, %26 : vector<4xf32>
            %31 = arith.addf %30, %27 : vector<4xf32>
            %32 = vector.extract %31[0] : f32 from vector<4xf32>
            %33 = vector.extract %31[1] : f32 from vector<4xf32>
            %34 = arith.addf %32, %33 : f32
            %35 = vector.extract %31[2] : f32 from vector<4xf32>
            %36 = arith.addf %34, %35 : f32
            %37 = vector.extract %31[3] : f32 from vector<4xf32>
            %38 = arith.addf %36, %37 : f32
            %r = gpu.subgroup_reduce add %38 : (f32) -> f32
            %r_vec = vector.broadcast %r : f32 to vector<1xf32>

            %view = memref.view %alloc[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<4xf32, #gpu.address_space<workgroup>>
            %46 = affine.apply #map5()[%thread_id_x]
            %47 = arith.index_cast %46 : index to i32
            %48 = arith.cmpi eq, %47, %c0_i32 : i32
            scf.if %48 {
            %90 = affine.apply #map6()[%thread_id_x]
            vector.store %r_vec, %view[%90] : memref<4xf32, #gpu.address_space<workgroup>>, vector<1xf32>
            }
            amdgpu.lds_barrier
            %49 = vector.load %view[%c0] : memref<4xf32, #gpu.address_space<workgroup>>, vector<4xf32>
            %50 = vector.extract %49[0] : f32 from vector<4xf32>
            %51 = vector.extract %49[1] : f32 from vector<4xf32>
            %52 = arith.addf %50, %51 : f32
            %53 = vector.extract %49[2] : f32 from vector<4xf32>
            %54 = arith.addf %52, %53 : f32
            %55 = vector.extract %49[3] : f32 from vector<4xf32>
            %56 = arith.addf %54, %55 : f32
            %57 = arith.divf %56, %cst_0 : f32
            %58 = arith.addf %57, %cst : f32
            %59 = math.sqrt %58 : f32
            %60 = vector.broadcast %59 : f32 to vector<4xf32>
            %61 = arith.divf %18, %60 : vector<4xf32>
            %62 = arith.divf %19, %60 : vector<4xf32>
            %63 = arith.divf %20, %60 : vector<4xf32>
            %64 = arith.divf %21, %60 : vector<4xf32>
            %65 = arith.divf %22, %60 : vector<4xf32>
            %66 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<5120xbf16, strided<[1], offset: ?>>
            %reinterpret_cast_11 = memref.reinterpret_cast %66 to offset: [%c0], sizes: [%c1073741822], strides: [1] : memref<5120xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
            %67 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_11 validBytes(%c2147483645_i32) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
            %68 = vector.load %67[%1] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %69 = vector.load %67[%6] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %70 = vector.load %67[%9] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %71 = vector.load %67[%12] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %72 = vector.load %67[%15] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %73 = arith.extf %68 : vector<4xbf16> to vector<4xf32>
            %74 = arith.extf %69 : vector<4xbf16> to vector<4xf32>
            %75 = arith.extf %70 : vector<4xbf16> to vector<4xf32>
            %76 = arith.extf %71 : vector<4xbf16> to vector<4xf32>
            %77 = arith.extf %72 : vector<4xbf16> to vector<4xf32>
            %78 = arith.mulf %61, %73 : vector<4xf32>
            %79 = arith.mulf %62, %74 : vector<4xf32>
            %80 = arith.mulf %63, %75 : vector<4xf32>
            %81 = arith.mulf %64, %76 : vector<4xf32>
            %82 = arith.mulf %65, %77 : vector<4xf32>
            %83 = arith.truncf %78 : vector<4xf32> to vector<4xbf16>
            %84 = arith.truncf %79 : vector<4xf32> to vector<4xbf16>
            %85 = arith.truncf %80 : vector<4xf32> to vector<4xbf16>
            %86 = arith.truncf %81 : vector<4xf32> to vector<4xbf16>
            %87 = arith.truncf %82 : vector<4xf32> to vector<4xbf16>
            %88 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<128x5120xbf16, strided<[5120, 1], offset: ?>>
            %reinterpret_cast_12 = memref.reinterpret_cast %88 to offset: [%c0], sizes: [%c1073741822], strides: [1] : memref<128x5120xbf16, strided<[5120, 1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
            %89 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_12 validBytes(%c2147483645_i32) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
            vector.store %83, %89[%3] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            vector.store %84, %89[%7] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            vector.store %85, %89[%10] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            vector.store %86, %89[%13] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            vector.store %87, %89[%16] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            return
        }
        }
    }
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<128x5120xbf16>
        %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<5120xbf16>
        %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<128x5120xbf16>
        %3 = flow.dispatch @test::@test(%0, %1, %2) : (tensor<128x5120xbf16>, tensor<5120xbf16>, tensor<128x5120xbf16>) -> %2
        %4 = hal.tensor.barrier join(%3 : tensor<128x5120xbf16>) => %arg4 : !hal.fence
        %5 = hal.tensor.export %4 : tensor<128x5120xbf16> -> !hal.buffer_view
        return %5 : !hal.buffer_view
    }
    }
"""
    
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    EMB_SIZE = tkl.sym.EMB_SIZE
    TOKENS_PER_WK = tkl.sym.TOKENS_PER_WK

    num_waves = 4
    wave_size = 64
    BLOCK_N = N
    BLOCK_M = TOKENS_PER_WK

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

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
        weight: tkl.Memory[N, ADDRESS_SPACE, tkl.bf16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
    ):
        length_embedding = tkl.Register[M, tkl.f32](N)
        eps_reg = tkl.Register[M, tkl.f32](eps)
        a_reg = tkw.read(a)
        a_reg = tkw.cast(a_reg, tkl.f32)
        mean = tkw.sum(a_reg * a_reg, dim=N, block=True) / length_embedding + eps_reg
        rms = tkw.sqrt(mean)
        rms_broad = tkw.broadcast(rms, [M, N])
        a_scaled = a_reg / rms_broad
        w_reg = tkw.read(weight)
        w_reg = tkw.cast(w_reg, tkl.f32)
        w_broad = tkw.broadcast(w_reg, [M, N])
        output = a_scaled * w_broad
        output = tkw.cast(output, tkl.bf16)
        tkw.write(output, c)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            TOKENS_PER_WK: 1,
            EMB_SIZE: shape[1],
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        override_mlir=override_mlir_str128_initialdpp,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    return wave_compile(options, test)


def get_rmsnorm_wave_initial_dpp_reduce(shape, eps: float = 1e-6):
    
    override_mlir_str128_initialdpp= """
    #map = affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 1024)>
    #map1 = affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 1024 + 256)>
    #map2 = affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 1024 + 512)>
    #map3 = affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 1024 + 768)>
    #map4 = affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 1024 + 1024)>
    #map5 = affine_map<()[s0] -> (s0 mod 64)>
    #map6 = affine_map<()[s0] -> ((s0 floordiv 64) mod 4)>
    #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 64>
    module attributes {transform.with_named_sequence} {
    stream.executable private @test {
        stream.executable.export public @test workgroups() -> (index, index, index) {
        %c1 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        stream.return %c1, %c128, %c1 : index, index, index
        }
        builtin.module {
        func.func @test(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
            %cst = arith.constant 9.99999997E-7 : f32
            %cst_0 = arith.constant 5.120000e+03 : f32
            %c0_i32 = arith.constant 0 : i32
            %c32_i32 = arith.constant 32 : i32
            %c16_i32 = arith.constant 16 : i32
            %c8_i32 = arith.constant 8 : i32
            %c4_i32 = arith.constant 4 : i32
            %c2_i32 = arith.constant 2 : i32
            %c64_i32 = arith.constant 64 : i32
            %c1_i32 = arith.constant 1 : i32
            %c2147483645_i32 = arith.constant 2147483645 : i32
            %c1073741822 = arith.constant 1073741822 : index
            %c0 = arith.constant 0 : index
            %c5120 = arith.constant 5120 : index
            %block_id_y = gpu.block_id  y upper_bound 128
            %thread_id_x = gpu.thread_id  x upper_bound 256
            %alloc = memref.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
            %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<128x5120xbf16, strided<[5120, 1], offset: ?>>
            %1 = affine.apply #map()[%thread_id_x]
            %2 = arith.muli %block_id_y, %c5120 overflow<nsw> : index
            %3 = arith.addi %2, %1 overflow<nsw> : index
            %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [%c1073741822], strides: [1] : memref<128x5120xbf16, strided<[5120, 1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
            %4 = amdgpu.fat_raw_buffer_cast %reinterpret_cast validBytes(%c2147483645_i32) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
            %5 = vector.load %4[%3] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %6 = affine.apply #map1()[%thread_id_x]
            %7 = arith.addi %2, %6 overflow<nsw> : index
            %8 = vector.load %4[%7] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %9 = affine.apply #map2()[%thread_id_x]
            %10 = arith.addi %2, %9 overflow<nsw> : index
            %11 = vector.load %4[%10] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %12 = affine.apply #map3()[%thread_id_x]
            %13 = arith.addi %2, %12 overflow<nsw> : index
            %14 = vector.load %4[%13] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %15 = affine.apply #map4()[%thread_id_x]
            %16 = arith.addi %2, %15 overflow<nsw> : index
            %17 = vector.load %4[%16] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %18 = arith.extf %5 : vector<4xbf16> to vector<4xf32>
            %19 = arith.extf %8 : vector<4xbf16> to vector<4xf32>
            %20 = arith.extf %11 : vector<4xbf16> to vector<4xf32>
            %21 = arith.extf %14 : vector<4xbf16> to vector<4xf32>
            %22 = arith.extf %17 : vector<4xbf16> to vector<4xf32>
            %23 = arith.mulf %18, %18 : vector<4xf32>
            %24 = arith.mulf %19, %19 : vector<4xf32>
            %25 = arith.mulf %20, %20 : vector<4xf32>
            %26 = arith.mulf %21, %21 : vector<4xf32>
            %27 = arith.mulf %22, %22 : vector<4xf32>
            %28 = arith.addf %23, %24 : vector<4xf32>
            %29 = arith.addf %28, %25 : vector<4xf32>
            %30 = arith.addf %29, %26 : vector<4xf32>
            %31 = arith.addf %30, %27 : vector<4xf32>
            %reduc   = vector.reduction <add>, %31 : vector<4xf32> into f32
            %r = gpu.subgroup_reduce add %reduc : (f32) -> f32
            %r_vec = vector.broadcast %r : f32 to vector<1xf32>
            %view = memref.view %alloc[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<4xf32, #gpu.address_space<workgroup>>
            %46 = affine.apply #map5()[%thread_id_x]
            %47 = arith.index_cast %46 : index to i32
            %48 = arith.cmpi eq, %47, %c0_i32 : i32
            scf.if %48 {
            %90 = affine.apply #map6()[%thread_id_x]
            vector.store %r_vec, %view[%90] : memref<4xf32, #gpu.address_space<workgroup>>, vector<1xf32>
            }
            amdgpu.lds_barrier
            %49 = vector.load %view[%c0] : memref<4xf32, #gpu.address_space<workgroup>>, vector<4xf32>  
            %reduc3 = vector.reduction <add>, %49 : vector<4xf32> into f32
            %57 = arith.divf %reduc3, %cst_0 : f32
            %58 = arith.addf %57, %cst : f32
            %59 = math.sqrt %58 : f32
            %60 = vector.broadcast %59 : f32 to vector<4xf32>
            %61 = arith.divf %18, %60 : vector<4xf32>
            %62 = arith.divf %19, %60 : vector<4xf32>
            %63 = arith.divf %20, %60 : vector<4xf32>
            %64 = arith.divf %21, %60 : vector<4xf32>
            %65 = arith.divf %22, %60 : vector<4xf32>
            %66 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<5120xbf16, strided<[1], offset: ?>>
            %reinterpret_cast_11 = memref.reinterpret_cast %66 to offset: [%c0], sizes: [%c1073741822], strides: [1] : memref<5120xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
            %67 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_11 validBytes(%c2147483645_i32) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
            %68 = vector.load %67[%1] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %69 = vector.load %67[%6] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %70 = vector.load %67[%9] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %71 = vector.load %67[%12] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %72 = vector.load %67[%15] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            %73 = arith.extf %68 : vector<4xbf16> to vector<4xf32>
            %74 = arith.extf %69 : vector<4xbf16> to vector<4xf32>
            %75 = arith.extf %70 : vector<4xbf16> to vector<4xf32>
            %76 = arith.extf %71 : vector<4xbf16> to vector<4xf32>
            %77 = arith.extf %72 : vector<4xbf16> to vector<4xf32>
            %78 = arith.mulf %61, %73 : vector<4xf32>
            %79 = arith.mulf %62, %74 : vector<4xf32>
            %80 = arith.mulf %63, %75 : vector<4xf32>
            %81 = arith.mulf %64, %76 : vector<4xf32>
            %82 = arith.mulf %65, %77 : vector<4xf32>
            %83 = arith.truncf %78 : vector<4xf32> to vector<4xbf16>
            %84 = arith.truncf %79 : vector<4xf32> to vector<4xbf16>
            %85 = arith.truncf %80 : vector<4xf32> to vector<4xbf16>
            %86 = arith.truncf %81 : vector<4xf32> to vector<4xbf16>
            %87 = arith.truncf %82 : vector<4xf32> to vector<4xbf16>
            %88 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<128x5120xbf16, strided<[5120, 1], offset: ?>>
            %reinterpret_cast_12 = memref.reinterpret_cast %88 to offset: [%c0], sizes: [%c1073741822], strides: [1] : memref<128x5120xbf16, strided<[5120, 1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
            %89 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_12 validBytes(%c2147483645_i32) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
            vector.store %83, %89[%3] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            vector.store %84, %89[%7] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            vector.store %85, %89[%10] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            vector.store %86, %89[%13] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            vector.store %87, %89[%16] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<4xbf16>
            return
        }
        }
    }
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<128x5120xbf16>
        %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<5120xbf16>
        %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<128x5120xbf16>
        %3 = flow.dispatch @test::@test(%0, %1, %2) : (tensor<128x5120xbf16>, tensor<5120xbf16>, tensor<128x5120xbf16>) -> %2
        %4 = hal.tensor.barrier join(%3 : tensor<128x5120xbf16>) => %arg4 : !hal.fence
        %5 = hal.tensor.export %4 : tensor<128x5120xbf16> -> !hal.buffer_view
        return %5 : !hal.buffer_view
    }
    }
"""
    
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    EMB_SIZE = tkl.sym.EMB_SIZE
    TOKENS_PER_WK = tkl.sym.TOKENS_PER_WK

    num_waves = 4
    wave_size = 64
    BLOCK_N = N
    BLOCK_M = TOKENS_PER_WK

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

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
        weight: tkl.Memory[N, ADDRESS_SPACE, tkl.bf16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
    ):
        length_embedding = tkl.Register[M, tkl.f32](N)
        eps_reg = tkl.Register[M, tkl.f32](eps)
        a_reg = tkw.read(a)
        a_reg = tkw.cast(a_reg, tkl.f32)
        mean = tkw.sum(a_reg * a_reg, dim=N, block=True) / length_embedding + eps_reg
        rms = tkw.sqrt(mean)
        rms_broad = tkw.broadcast(rms, [M, N])
        a_scaled = a_reg / rms_broad
        w_reg = tkw.read(weight)
        w_reg = tkw.cast(w_reg, tkl.f32)
        w_broad = tkw.broadcast(w_reg, [M, N])
        output = a_scaled * w_broad
        output = tkw.cast(output, tkl.bf16)
        tkw.write(output, c)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            TOKENS_PER_WK: 1,
            EMB_SIZE: shape[1],
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        override_mlir=override_mlir_str128_initialdpp,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    return wave_compile(options, test)


def get_rmsnorm_wave_self_buffer_dpp(shape, eps: float = 1e-6):
    override_mlir_str128_buffer2 = """
    #map  = affine_map<()[s0, s1] -> ((s0 mod 64) * 8 + s1 * 512)>
    #map1 = affine_map<()[s0, s1] -> (s0 floordiv 64 + s1 * 4 )>
    #map2 = affine_map<()[s0] -> (s0 floordiv 64 )>
    #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 64>
    module attributes {transform.with_named_sequence} {
    stream.executable private @test {
        stream.executable.export public @test workgroups() -> (index, index, index) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 32 : index
        stream.return %c1, %c2, %c1 : index, index, index
        }
        builtin.module {
        func.func @test(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
            %cst               = arith.constant dense<5.120000e+03> : vector<1xf32>        
            %c32_i32           = arith.constant 32 : i32
            %c0_i32            = arith.constant 0 : i32
            %c64_i32           = arith.constant 64 : i32
            %c1_i32            = arith.constant 1 : i32
            %c5120             = arith.constant 5120 : index
            %c1                = arith.constant 1 : index
            %c0                = arith.constant 0 : index
            %c10               = arith.constant 10 : index
            %c2147483645_i32   = arith.constant 2147483645 : i32
            %c1073741822       = arith.constant 1073741822 : index
            %cst_0      = arith.constant dense<0.0> : vector<1xf32>
            %block_id_y  = gpu.block_id  y upper_bound 128
            %thread_id_x = gpu.thread_id x upper_bound 256
            %alloc  = memref.alloc() : memref<40960xi8, #gpu.address_space<workgroup>>
            %buffer = memref.view %alloc[%c0][] : memref<40960xi8, #gpu.address_space<workgroup>> to memref<4x5120xbf16, #gpu.address_space<workgroup>>
            %input = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<128x5120xbf16, strided<[5120, 1], offset: ?>>
            %reinterpret_cast = memref.reinterpret_cast %input to offset: [%c0], sizes: [%c1073741822], strides: [1]: memref<128x5120xbf16, strided<[5120, 1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
            %in_buf = amdgpu.fat_raw_buffer_cast %reinterpret_cast validBytes(%c2147483645_i32) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
            %sum_vec = scf.for %tile = %c0 to %c10 step %c1 iter_args(%acc = %cst_0) -> (vector<1xf32>) {
            %row        = affine.apply #map1()[%thread_id_x, %block_id_y]
            %rowbuffer  = affine.apply #map2()[%thread_id_x]
            %col        = affine.apply #map()[%thread_id_x, %tile]
            %rowoffset  = arith.muli %row, %c5120 : index
            %id_offset  = arith.addi %rowoffset, %col : index
            %val_bf16   = vector.load %in_buf[%id_offset] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
            vector.store %val_bf16, %buffer[%rowbuffer, %col] : memref<4x5120xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
            %val_f32    = arith.extf %val_bf16 : vector<8xbf16> to vector<8xf32>
            %sq_f32     = arith.mulf %val_f32, %val_f32 : vector<8xf32>
            %tile_sum   = vector.reduction <add>, %sq_f32 : vector<8xf32> into f32
            %r          = gpu.subgroup_reduce add %tile_sum : (f32) -> f32
            %r_vec      = vector.broadcast %r : f32 to vector<1xf32>
            %new_acc    = arith.addf %r_vec, %acc : vector<1xf32>
            scf.yield %new_acc : vector<1xf32>
            }
            %mean_vec = arith.divf %sum_vec, %cst : vector<1xf32>
            %rms_vec  = math.sqrt %mean_vec : vector<1xf32>
            %broadcasted = vector.broadcast %rms_vec : vector<1xf32> to vector<8xf32>
            %w_bind = stream.binding.subspan %arg1[%c0]: !stream.binding -> memref<5120xbf16, strided<[1], offset: ?>>
            %w_rc   = memref.reinterpret_cast %w_bind to offset: [%c0], sizes: [%c1073741822], strides: [1]: memref<5120xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
            %w_buf  = amdgpu.fat_raw_buffer_cast %w_rc validBytes(%c2147483645_i32) resetOffset: memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
            %out_bind = stream.binding.subspan %arg2[%c0]: !stream.binding -> memref<128x5120xbf16, strided<[5120, 1], offset: ?>>
            %out_rc   = memref.reinterpret_cast %out_bind to offset: [%c0], sizes: [%c1073741822], strides: [1] : memref<128x5120xbf16, strided<[5120, 1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
            %out_buf  = amdgpu.fat_raw_buffer_cast %out_rc validBytes(%c2147483645_i32) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
            scf.for %tile = %c0 to %c10 step %c1 {
            %rowbuffer  = affine.apply #map2()[%thread_id_x]
            %row        = affine.apply #map1()[%thread_id_x, %block_id_y]
            %col        = affine.apply #map()[%thread_id_x, %tile]
            %rowoffset  = arith.muli %row, %c5120 : index
            %id_offset  = arith.addi %rowoffset, %col : index
            %val_bf16   = vector.load %buffer[%rowbuffer, %col]: memref<4x5120xbf16, #gpu.address_space<workgroup>>,vector<8xbf16>
            %val_f32    = arith.extf %val_bf16 : vector<8xbf16> to vector<8xf32>
            %w_bf16     = vector.load %w_buf[%col]: memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>,vector<8xbf16>
            %w_f32      = arith.extf %w_bf16 : vector<8xbf16> to vector<8xf32>
            %normed     = arith.divf %val_f32, %broadcasted : vector<8xf32>
            %scaled     = arith.mulf %normed, %w_f32 : vector<8xf32>
            %out_bf16   = arith.truncf %scaled : vector<8xf32> to vector<8xbf16>
            vector.store %out_bf16, %out_buf[%id_offset]: memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
            }
            return
        }
        }
    }

    // async wrapper uses bf16 tensors too
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %t0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<128x5120xbf16>
        %t1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<5120xbf16>
        %t2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<128x5120xbf16>
        %t3 = flow.dispatch @test::@test(%t0, %t1, %t2) : (tensor<128x5120xbf16>, tensor<5120xbf16>, tensor<128x5120xbf16>) -> %t2
        %b  = hal.tensor.barrier join(%t3 : tensor<128x5120xbf16>) => %arg4 : !hal.fence
        %o  = hal.tensor.export %b : tensor<128x5120xbf16> -> !hal.buffer_view
        return %o : !hal.buffer_view
    }
    }
    """
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = sympy.Min(sympy.Max(64 * 4, N / 16), N)
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: BLOCK_N },
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
        weight: tkl.Memory[N, ADDRESS_SPACE, tkl.bf16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
    ):
        length_embedding = tkl.Register[M, tkl.f32](N)
        eps_reg = tkl.Register[M, tkl.f32](eps)
        a_reg = tkw.read(a)
        a_reg = tkw.cast(a_reg, tkl.f32)
        mean = tkw.sum(a_reg * a_reg, dim=N, block=True) / length_embedding + eps_reg
        rms = tkw.rsqrt(mean)
        rms_broad = tkw.broadcast(rms, [M, N])
        a_scaled = a_reg * rms_broad
        w_reg = tkw.read(weight)
        w_reg = tkw.cast(w_reg, tkl.f32)
        w_broad = tkw.broadcast(w_reg, [M, N])
        output = a_scaled * w_broad
        output = tkw.cast(output, tkl.bf16)
        tkw.write(output, c)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        # use_buffer_load_ops=True,
        # use_buffer_store_ops=True,
        override_mlir=override_mlir_str128_buffer2,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    return wave_compile(options, test)


def rmsnorm_wave(
    kernel,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    c = torch.empty_like(x)
    kernel(x, weight, c)

    return c


class HuggingFaceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    naive_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    naive_norm.weight = nn.Parameter(weight)
    naive_norm = naive_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = naive_norm(x, residual)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, weight, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def calculate_diff(seq_len, hidden_size, use_residual=True):
    dtype = torch.bfloat16
    x = torch.randn(seq_len, hidden_size, dtype=dtype, device="cuda")
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x) if use_residual else None
    #wave_kernel = get_rmsnorm_wave_initial(x.shape)
    

    wave_kernel = get_rmsnorm_wave_initial_dpp_reduce(x.shape)

    #wave_kernel = get_rmsnorm_wave_self_buffer_dpp(x.shape)

    output_naive = rmsnorm_naive(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_wave = rmsnorm_wave(
        wave_kernel,
        x.clone(),
        weight,
        # residual.clone() if residual is not None else None,
    )
    output_vllm = rmsnorm_vllm(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_triton = fused_rmsnorm(
        x.clone(), weight
    )

    if use_residual:
        output_naive = output_naive[0]
        output_wave = output_wave[0]
        output_vllm = output_vllm[0]
        output_triton = output_triton[0]

    print(f"Naive output={output_naive}")
    print(f"Wave output={output_wave}")
    print(f"VLLM output={output_vllm}")
    print(f"Triton output={output_triton}")

    if torch.allclose(
        output_naive, output_wave, atol=1e-2, rtol=1e-2
    ) and torch.allclose(output_naive, output_vllm, atol=1e-2, rtol=1e-2) and torch.allclose(output_naive, output_triton, atol=1e-2, rtol=1e-2):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


# seq_length_range = [1, 16] + [2**i for i in range(6, 11, 1)]
# hidden_size_range = [i * 128 for i in [1, 32, 48]] + [5120]
seq_length_range = [128]
hidden_size_range = [5120]
configs = list(itertools.product(seq_length_range, hidden_size_range))


def get_benchmark(use_residual):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_len", "hidden_size",],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["wave_initial", "wave_initial_dpp", "wave_initial_dpp_reduce","wave_self", "triton"],
            line_names=["wave_initial", "wave_initial_dpp","wave_initial_dpp_reduce", "Wave_self", "Triton"],
            line_vals=["wave_initial", "triton"],
            line_names=["wave_initial", "Triton"],
            styles=[("blue", "-"), ("red", "-"), ("purple", "-"), ("yellow", "-"), ("teal", "-")],
            ylabel="us",
            plot_name=f"rmsnorm-performance-{'with' if use_residual else 'without'}-residual",
            args={},
        )
    )
    def benchmark(seq_len, hidden_size, provider):
        dtype = torch.bfloat16

        x = torch.randn(seq_len, hidden_size, dtype=dtype, device="cuda")
        weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
        residual = torch.randn_like(x) if use_residual else None

        quantiles = [0.5, 0.2, 0.8]

        if provider == "huggingface":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_naive(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "wave_rsqrt":
            wave_kernel = get_rmsnorm_wave_rsqrt(x.shape)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_wave(
                    wave_kernel,
                    x.clone(),
                    weight
                ),
                quantiles=quantiles,
            )
        elif provider == "wave_initial":
            wave_kernel = get_rmsnorm_wave_initial(x.shape)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_wave(
                    wave_kernel,
                    x.clone(),
                    weight
                ),
                quantiles=quantiles,
            )
        elif provider == "wave_initial_dpp":
            wave_kernel = get_rmsnorm_wave_initial_dpp(x.shape)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_wave(
                    wave_kernel,
                    x.clone(),
                    weight
                ),
                quantiles=quantiles,
            )
        elif provider == "wave_initial_dpp_reduce":
            wave_kernel = get_rmsnorm_wave_initial_dpp_reduce(x.shape)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_wave(
                    wave_kernel,
                    x.clone(),
                    weight
                ),
                quantiles=quantiles,
            )
        elif provider == "wave_self":
            wave_kernel = get_rmsnorm_wave_self_buffer_dpp(x.shape)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_wave(
                    wave_kernel,
                    x.clone(),
                    weight
                ),
                quantiles=quantiles,
            )
        
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_rmsnorm(
                    x.clone(),
                    weight,
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_vllm(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_residual", action="store_true", help="Whether to use residual connection"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="Path to save rmsnorm benchmark results",
    )
    args = parser.parse_args()

    # Run correctness test
    calculate_diff(
        seq_len=128, hidden_size=5120, use_residual=args.use_residual
    )
    

    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark(args.use_residual)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)