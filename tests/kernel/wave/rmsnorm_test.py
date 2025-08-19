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


@pytest.mark.parametrize(
    "shape",
    [
        (128, 5120),
    ],
)
@require_e2e
def test_rmsnorm(shape, eps: float = 1e-6):
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

    
    override_mlir_str128 = """
    #map = affine_map<()[s0, s1] -> ((s0 mod 64) * 8 + s1 * 512)>
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
            %cst = arith.constant dense<5.120000e+03> : vector<1xf16>
            %c32_i32 = arith.constant 32 : i32
            %c0_i32 = arith.constant 0 : i32
            %c16_i32 = arith.constant 16 : i32
            %c8_i32 = arith.constant 8 : i32
            %c4_i32 = arith.constant 4 : i32
            %c2_i32 = arith.constant 2 : i32
            %c64_i32 = arith.constant 64 : i32
            %c1_i32 = arith.constant 1 : i32
            %c1 = arith.constant 1 : index
            %c0 = arith.constant 0 : index
            %c10 = arith.constant 10 : index
            %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf16>
            %block_id_y = gpu.block_id  y upper_bound 32
            %thread_id_x = gpu.thread_id  x upper_bound 256
            %alloc = memref.alloc() : memref<40960xi8, #gpu.address_space<workgroup>>
            %buffer = memref.view %alloc[%c0][] : memref<40960xi8, #gpu.address_space<workgroup>> to memref<4x5120xf16, #gpu.address_space<workgroup>>
            %input = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<128x5120xf16, strided<[5120, 1], offset: ?>>
            %1 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %cst_0) -> (vector<1xf16>) {
            %row = affine.apply #map1()[%thread_id_x,%block_id_y]
            %rowbuffer = affine.apply #map2()[%thread_id_x]
            %col = affine.apply #map()[%thread_id_x, %arg3]
            %val = vector.load %input[%row, %col] : memref<128x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
            vector.store %val, %buffer[%rowbuffer, %col] : memref<4x5120xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %15 = arith.mulf %val, %val : vector<8xf16>
            %30 = vector.reduction <add>, %15 : vector<8xf16> into f16
            %r = gpu.subgroup_reduce add %30 : (f16) -> f16
            %31 = vector.broadcast %r : f16 to vector<1xf16>
            %38 = arith.addf %31, %arg4 : vector<1xf16>
            scf.yield %38 : vector<1xf16>
            }
            %2 = arith.divf %1, %cst : vector<1xf16>
            %3 = math.sqrt %2 : vector<1xf16>
            %broadcasted = vector.broadcast %3 : vector<1xf16> to vector<8xf16>
            %185 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<5120xf16, strided<[1], offset: ?>>
            %70 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<128x5120xf16, strided<[5120, 1], offset: ?>>
            scf.for %arg3 = %c0 to %c10 step %c1 {
            %rowbuffer = affine.apply #map2()[%thread_id_x]
            %row = affine.apply #map1()[%thread_id_x,%block_id_y]
            %col = affine.apply #map()[%thread_id_x, %arg3]
            %val = vector.load %buffer[%rowbuffer, %col] : memref<4x5120xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %53 = vector.load %185[%col] : memref<5120xf16, strided<[1], offset: ?>>, vector<8xf16>
            %54 = arith.divf %val, %broadcasted : vector<8xf16>
            %55 = arith.mulf %54, %53 : vector<8xf16>
            vector.store %55, %70[%row, %col] : memref<128x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
            }
            return
        }
        }
    }
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<128x5120xf16>
        %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<5120xf16>
        %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<128x5120xf16>
        %3 = flow.dispatch @test::@test(%0, %1, %2) : (tensor<128x5120xf16>, tensor<5120xf16>, tensor<128x5120xf16>) -> %2
        %4 = hal.tensor.barrier join(%3 : tensor<128x5120xf16>) => %arg4 : !hal.fence
        %5 = hal.tensor.export %4 : tensor<128x5120xf16> -> !hal.buffer_view
        return %5 : !hal.buffer_view
    }
    }
    """
    
    
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
            ELEMS_PER_THREAD: 32,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        # print_ir_after=["set_thread_dependent_index_from_reduce", "expand_graph"],
        # print_ir_before=["set_thread_dependent_index_from_reduce", "expand graph"],
        #override_mlir=override_mlir_str128_initialdpp,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.bfloat16)
    gamma = device_randn(shape[1], dtype=torch.bfloat16)
    c = device_zeros(shape, dtype=torch.bfloat16)
    test(a, gamma, c)
    print(test.asm)
    print(c.cpu())
  
    eps = 1e-6
    orig_dtype = torch.bfloat16  # or a.dtype if you want to keep original
    mean = (a.float() * a.float()).mean(dim=-1, keepdim=True)
    rms  = torch.sqrt(mean + torch.tensor(eps, dtype=mean.dtype, device=a.device))
    ref  = ((a.float() / rms) * gamma.float()).to(torch.bfloat16)  # cast back here
    print("ref",ref)
    torch.testing.assert_close(ref, c, atol=1e-01, rtol=1e-05)
