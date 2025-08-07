import torch
from torch.profiler import profile, record_function, ProfilerActivity

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
    device_ones,
    device_arange,
    device_randint,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from .common.utils import (
    require_e2e,
)
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
import time
import torch.nn as nn


@pytest.mark.parametrize(
    "shape",
    [
        (1, 5120),
    ],
)
@require_e2e
def test(shape):
    override_mlir_str1_tiled_shared = """
    #map = affine_map<()[s0, s1] -> (s0 * 4 + (s0 floordiv 64) * 1024 + s1 * 256)>
    #map1 = affine_map<()[s0] -> (s0 mod 64)>
    #map2 = affine_map<()[s0] -> ((s0 floordiv 64) mod 4)>
    #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 64>
    module attributes {transform.with_named_sequence} {
    stream.executable private @test {
        stream.executable.export public @test workgroups() -> (index, index, index) {
        %c1 = arith.constant 1 : index
        stream.return %c1, %c1, %c1 : index, index, index
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
            %c5 = arith.constant 5 : index
            %c0 = arith.constant 0 : index
            %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf16>
            %thread_id_x = gpu.thread_id  x upper_bound 256
            %alloc_red = memref.alloc() : memref<8xi8, #gpu.address_space<workgroup>>
            %alloc_input = memref.alloc() : memref<10240xi8, #gpu.address_space<workgroup>>
            %buffer = memref.view %alloc_input[%c0][] : memref<10240xi8, #gpu.address_space<workgroup>> to memref<1x5120xf16, #gpu.address_space<workgroup>>
            %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<1x5120xf16, strided<[5120, 1], offset: ?>>
            %1 = scf.for %arg3 = %c0 to %c5 step %c1 iter_args(%arg4 = %cst_0) -> (vector<1xf16>) {
            %5 = affine.apply #map()[%thread_id_x, %arg3]
            %6 = vector.load %0[%c0, %5] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            vector.store %6, %buffer[%c0, %5] : memref<1x5120xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %15 = arith.mulf %6, %6 : vector<4xf16>
            %30 = vector.reduction <add>, %15 : vector<4xf16> into f16
            %31 = vector.broadcast %30 : f16 to vector<1xf16>
            %shuffleResult, %valid = gpu.shuffle  xor %31, %c1_i32, %c64_i32 : vector<1xf16>
            %32 = arith.addf %31, %shuffleResult : vector<1xf16>
            %shuffleResult_1, %valid_2 = gpu.shuffle  xor %32, %c2_i32, %c64_i32 : vector<1xf16>
            %33 = arith.addf %32, %shuffleResult_1 : vector<1xf16>
            %shuffleResult_3, %valid_4 = gpu.shuffle  xor %33, %c4_i32, %c64_i32 : vector<1xf16>
            %34 = arith.addf %33, %shuffleResult_3 : vector<1xf16>
            %shuffleResult_5, %valid_6 = gpu.shuffle  xor %34, %c8_i32, %c64_i32 : vector<1xf16>
            %35 = arith.addf %34, %shuffleResult_5 : vector<1xf16>
            %shuffleResult_7, %valid_8 = gpu.shuffle  xor %35, %c16_i32, %c64_i32 : vector<1xf16>
            %36 = arith.addf %35, %shuffleResult_7 : vector<1xf16>
            %shuffleResult_9, %valid_10 = gpu.shuffle  xor %36, %c32_i32, %c64_i32 : vector<1xf16>
            %37 = arith.addf %36, %shuffleResult_9 : vector<1xf16>
            %38 = arith.addf %37, %arg4 : vector<1xf16>
            scf.yield %38 : vector<1xf16>
            }
            %view = memref.view %alloc_red[%c0][] : memref<8xi8, #gpu.address_space<workgroup>> to memref<4xf16, #gpu.address_space<workgroup>>
            %39 = affine.apply #map1()[%thread_id_x]
            %41 = arith.cmpi eq, %39, %c0 : index
            scf.if %41 {
            %54 = affine.apply #map2()[%thread_id_x]
            vector.store %1, %view[%54] : memref<4xf16, #gpu.address_space<workgroup>>, vector<1xf16>
            }
            amdgpu.lds_barrier
            %42 = vector.load %view[%c0] : memref<4xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %49 = vector.reduction <add>, %42 : vector<4xf16> into f16
            %result = vector.broadcast %49 : f16 to vector<1xf16>
            %2 = arith.divf %result, %cst : vector<1xf16>
            %3 = math.sqrt %2 : vector<1xf16>
            %broadcasted = vector.broadcast %3 : vector<1xf16> to vector<4xf16>
            %185 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<5120xf16, strided<[1], offset: ?>>
            %70 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<1x5120xf16, strided<[5120, 1], offset: ?>>
            scf.for %arg3 = %c0 to %c5 step %c1 {
            %51 = affine.apply #map()[%thread_id_x, %arg3]
            %52 = vector.load %buffer[%c0, %51] : memref<1x5120xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %53 = vector.load %185[%51] : memref<5120xf16, strided<[1], offset: ?>>, vector<4xf16>
            %54 = arith.divf %52, %broadcasted : vector<4xf16>
            %55 = arith.mulf %54, %53 : vector<4xf16>
            vector.store %55, %70[%c0, %51] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            }
            return
        }
        }
    }
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<1x5120xf16>
        %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<5120xf16>
        %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<1x5120xf16>
        %3 = flow.dispatch @test::@test(%0, %1, %2) : (tensor<1x5120xf16>, tensor<5120xf16>, tensor<1x5120xf16>) -> %2
        %4 = hal.tensor.barrier join(%3 : tensor<1x5120xf16>) => %arg4 : !hal.fence
        %5 = hal.tensor.export %4 : tensor<1x5120xf16> -> !hal.buffer_view
        return %5 : !hal.buffer_view
    }
    }
    """
    override_mlir_str1_dpp = """
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
        stream.return %c1, %c1, %c1 : index, index, index
        }
        builtin.module {
        func.func @test(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
            %cst = arith.constant 5.120000e+03 : f16
            %c0_i32 = arith.constant 0 : i32
            %c32_i32 = arith.constant 32 : i32
            %c16_i32 = arith.constant 16 : i32
            %c8_i32 = arith.constant 8 : i32
            %c4_i32 = arith.constant 4 : i32
            %c2_i32 = arith.constant 2 : i32
            %c64_i32 = arith.constant 64 : i32
            %c1_i32 = arith.constant 1 : i32
            %c0 = arith.constant 0 : index
            %thread_id_x = gpu.thread_id  x upper_bound 256
            %alloc = memref.alloc() : memref<8xi8, #gpu.address_space<workgroup>>
            %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<1x5120xf16, strided<[5120, 1], offset: ?>>
            %1 = affine.apply #map()[%thread_id_x]
            %2 = vector.load %0[%c0, %1] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            %3 = affine.apply #map1()[%thread_id_x]
            %4 = vector.load %0[%c0, %3] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            %5 = affine.apply #map2()[%thread_id_x]
            %6 = vector.load %0[%c0, %5] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            %7 = affine.apply #map3()[%thread_id_x]
            %8 = vector.load %0[%c0, %7] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            %9 = affine.apply #map4()[%thread_id_x]
            %10 = vector.load %0[%c0, %9] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            %11 = arith.mulf %2, %2 : vector<4xf16>
            %12 = arith.mulf %4, %4 : vector<4xf16>
            %13 = arith.mulf %6, %6 : vector<4xf16>
            %14 = arith.mulf %8, %8 : vector<4xf16>
            %15 = arith.mulf %10, %10 : vector<4xf16>
            %16 = arith.addf %11, %12 : vector<4xf16>
            %17 = arith.addf %16, %13 : vector<4xf16>
            %18 = arith.addf %17, %14 : vector<4xf16>
            %19 = arith.addf %18, %15 : vector<4xf16>
            %30 = vector.reduction <add>, %19 : vector<4xf16> into f16
            %r = gpu.subgroup_reduce add %30 : (f16) -> f16
            %33 = vector.broadcast %r : f16 to vector<1xf16>
            %view = memref.view %alloc[%c0][] : memref<8xi8, #gpu.address_space<workgroup>> to memref<4xf16, #gpu.address_space<workgroup>>
            %34 = affine.apply #map5()[%thread_id_x]
            %35 = arith.index_cast %34 : index to i32
            %36 = arith.cmpi eq, %35, %c0_i32 : i32
            scf.if %36 {
            %65 = affine.apply #map6()[%thread_id_x]
            vector.store %33, %view[%65] : memref<4xf16, #gpu.address_space<workgroup>>, vector<1xf16>
            }
            amdgpu.lds_barrier
            %37 = vector.load %view[%c0] : memref<4xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %38 = vector.extract %37[0] : f16 from vector<4xf16>
            %39 = vector.extract %37[1] : f16 from vector<4xf16>
            %40 = arith.addf %38, %39 : f16
            %41 = vector.extract %37[2] : f16 from vector<4xf16>
            %42 = arith.addf %40, %41 : f16
            %43 = vector.extract %37[3] : f16 from vector<4xf16>
            %44 = arith.addf %42, %43 : f16
            %45 = arith.divf %44, %cst : f16
            %46 = math.sqrt %45 : f16
            %47 = vector.broadcast %46 : f16 to vector<4xf16>
            %48 = arith.divf %2, %47 : vector<4xf16>
            %49 = arith.divf %4, %47 : vector<4xf16>
            %50 = arith.divf %6, %47 : vector<4xf16>
            %51 = arith.divf %8, %47 : vector<4xf16>
            %52 = arith.divf %10, %47 : vector<4xf16>
            %53 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<5120xf16, strided<[1], offset: ?>>
            %54 = vector.load %53[%1] : memref<5120xf16, strided<[1], offset: ?>>, vector<4xf16>
            %55 = vector.load %53[%3] : memref<5120xf16, strided<[1], offset: ?>>, vector<4xf16>
            %56 = vector.load %53[%5] : memref<5120xf16, strided<[1], offset: ?>>, vector<4xf16>
            %57 = vector.load %53[%7] : memref<5120xf16, strided<[1], offset: ?>>, vector<4xf16>
            %58 = vector.load %53[%9] : memref<5120xf16, strided<[1], offset: ?>>, vector<4xf16>
            %59 = arith.mulf %48, %54 : vector<4xf16>
            %60 = arith.mulf %49, %55 : vector<4xf16>
            %61 = arith.mulf %50, %56 : vector<4xf16>
            %62 = arith.mulf %51, %57 : vector<4xf16>
            %63 = arith.mulf %52, %58 : vector<4xf16>
            %64 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<1x5120xf16, strided<[5120, 1], offset: ?>>
            vector.store %59, %64[%c0, %1] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            vector.store %60, %64[%c0, %3] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            vector.store %61, %64[%c0, %5] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            vector.store %62, %64[%c0, %7] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            vector.store %63, %64[%c0, %9] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            return
        }
        }
    }
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<1x5120xf16>
        %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<5120xf16>
        %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<1x5120xf16>
        %3 = flow.dispatch @test::@test(%0, %1, %2) : (tensor<1x5120xf16>, tensor<5120xf16>, tensor<1x5120xf16>) -> %2
        %4 = hal.tensor.barrier join(%3 : tensor<1x5120xf16>) => %arg4 : !hal.fence
        %5 = hal.tensor.export %4 : tensor<1x5120xf16> -> !hal.buffer_view
        return %5 : !hal.buffer_view
    }
    }
    """
    override_mlir_str1_tiled = """
    #map = affine_map<()[s0, s1] -> (s0 * 4 + (s0 floordiv 64) * 1024 + s1 * 256)>
    #map1 = affine_map<()[s0] -> (s0 mod 64)>
    #map2 = affine_map<()[s0] -> ((s0 floordiv 64) mod 4)>
    #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 64>
    module attributes {transform.with_named_sequence} {
    stream.executable private @test {
        stream.executable.export public @test workgroups() -> (index, index, index) {
        %c1 = arith.constant 1 : index
        stream.return %c1, %c1, %c1 : index, index, index
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
            %c5 = arith.constant 5 : index
            %c0 = arith.constant 0 : index
            %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf16>
            %thread_id_x = gpu.thread_id  x upper_bound 256
            %alloc = memref.alloc() : memref<8xi8, #gpu.address_space<workgroup>>
            %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<1x5120xf16, strided<[5120, 1], offset: ?>>
            %1 = scf.for %arg3 = %c0 to %c5 step %c1 iter_args(%arg4 = %cst_0) -> (vector<1xf16>) {
                %5 = affine.apply #map()[%thread_id_x, %arg3]
                %6 = vector.load %0[%c0, %5] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
                %15 = arith.mulf %6, %6 : vector<4xf16>
                %30 = vector.reduction <add>, %15 : vector<4xf16> into f16
                %31 = vector.broadcast %30 : f16 to vector<1xf16>
                %shuffleResult, %valid = gpu.shuffle  xor %31, %c1_i32, %c64_i32 : vector<1xf16>
                %32 = arith.addf %31, %shuffleResult : vector<1xf16>
                %shuffleResult_1, %valid_2 = gpu.shuffle  xor %32, %c2_i32, %c64_i32 : vector<1xf16>
                %33 = arith.addf %32, %shuffleResult_1 : vector<1xf16>
                %shuffleResult_3, %valid_4 = gpu.shuffle  xor %33, %c4_i32, %c64_i32 : vector<1xf16>
                %34 = arith.addf %33, %shuffleResult_3 : vector<1xf16>
                %shuffleResult_5, %valid_6 = gpu.shuffle  xor %34, %c8_i32, %c64_i32 : vector<1xf16>
                %35 = arith.addf %34, %shuffleResult_5 : vector<1xf16>
                %shuffleResult_7, %valid_8 = gpu.shuffle  xor %35, %c16_i32, %c64_i32 : vector<1xf16>
                %36 = arith.addf %35, %shuffleResult_7 : vector<1xf16>
                %shuffleResult_9, %valid_10 = gpu.shuffle  xor %36, %c32_i32, %c64_i32 : vector<1xf16>
                %37 = arith.addf %36, %shuffleResult_9 : vector<1xf16>
                %38 = arith.addf %37, %arg4 : vector<1xf16>
                scf.yield %38 : vector<1xf16>
            }
            %view = memref.view %alloc[%c0][] : memref<8xi8, #gpu.address_space<workgroup>> to memref<4xf16, #gpu.address_space<workgroup>>
            %39 = affine.apply #map1()[%thread_id_x]
            %41 = arith.cmpi eq, %39, %c0 : index
            scf.if %41 {
            %54 = affine.apply #map2()[%thread_id_x]
            vector.store %1, %view[%54] : memref<4xf16, #gpu.address_space<workgroup>>, vector<1xf16>
            }
            amdgpu.lds_barrier
            %42 = vector.load %view[%c0] : memref<4xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %49 = vector.reduction <add>, %42 : vector<4xf16> into f16
            %result = vector.broadcast %49 : f16 to vector<1xf16>
            %2 = arith.divf %result, %cst : vector<1xf16>
            %3 = math.sqrt %2 : vector<1xf16>
            %broadcasted = vector.broadcast %3 : vector<1xf16> to vector<4xf16>
            %185 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<5120xf16, strided<[1], offset: ?>>
            %70 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<1x5120xf16, strided<[5120, 1], offset: ?>>
            scf.for %arg3 = %c0 to %c5 step %c1 {
                %51 = affine.apply #map()[%thread_id_x, %arg3]
                %52 = vector.load %0[%c0, %51] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
                %53 = vector.load %185[%51] : memref<5120xf16, strided<[1], offset: ?>>, vector<4xf16>
                %54 = arith.divf %52, %broadcasted : vector<4xf16>
                %55 = arith.mulf %54, %53 : vector<4xf16>
                vector.store %55, %70[%c0, %51] : memref<1x5120xf16, strided<[5120, 1], offset: ?>>, vector<4xf16>
            }
            return
        }
        }
    }
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<1x5120xf16>
        %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<5120xf16>
        %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<1x5120xf16>
        %3 = flow.dispatch @test::@test(%0, %1, %2) : (tensor<1x5120xf16>, tensor<5120xf16>, tensor<1x5120xf16>) -> %2
        %4 = hal.tensor.barrier join(%3 : tensor<1x5120xf16>) => %arg4 : !hal.fence
        %5 = hal.tensor.export %4 : tensor<1x5120xf16> -> !hal.buffer_view
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

    override_mlir_str128_nosharedmem = """
    #map = affine_map<()[s0, s1] -> ((s0 mod 64) * 8 + s1 * 512)>
    #map1 = affine_map<()[s0, s1] -> (s0 floordiv 64 + s1 * 4 )>
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
            %input = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<128x5120xf16, strided<[5120, 1], offset: ?>>
            %1 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %cst_0) -> (vector<1xf16>) {
            %row = affine.apply #map1()[%thread_id_x,%block_id_y]
            %col = affine.apply #map()[%thread_id_x, %arg3]
            %val = vector.load %input[%row, %col] : memref<128x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
            %15 = arith.mulf %val, %val : vector<8xf16>
            %30 = vector.reduction <add>, %15 : vector<8xf16> into f16
            %31 = vector.broadcast %30 : f16 to vector<1xf16>
            %shuffleResult, %valid = gpu.shuffle  xor %31, %c1_i32, %c64_i32 : vector<1xf16>
            %32 = arith.addf %31, %shuffleResult : vector<1xf16>
            %shuffleResult_1, %valid_2 = gpu.shuffle  xor %32, %c2_i32, %c64_i32 : vector<1xf16>
            %33 = arith.addf %32, %shuffleResult_1 : vector<1xf16>
            %shuffleResult_3, %valid_4 = gpu.shuffle  xor %33, %c4_i32, %c64_i32 : vector<1xf16>
            %34 = arith.addf %33, %shuffleResult_3 : vector<1xf16>
            %shuffleResult_5, %valid_6 = gpu.shuffle  xor %34, %c8_i32, %c64_i32 : vector<1xf16>
            %35 = arith.addf %34, %shuffleResult_5 : vector<1xf16>
            %shuffleResult_7, %valid_8 = gpu.shuffle  xor %35, %c16_i32, %c64_i32 : vector<1xf16>
            %36 = arith.addf %35, %shuffleResult_7 : vector<1xf16>
            %shuffleResult_9, %valid_10 = gpu.shuffle  xor %36, %c32_i32, %c64_i32 : vector<1xf16>
            %37 = arith.addf %36, %shuffleResult_9 : vector<1xf16>
            %38 = arith.addf %37, %arg4 : vector<1xf16>
            scf.yield %38 : vector<1xf16>
            }
            %2 = arith.divf %1, %cst : vector<1xf16>
            %3 = math.sqrt %2 : vector<1xf16>
            %broadcasted = vector.broadcast %3 : vector<1xf16> to vector<8xf16>
            %185 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<5120xf16, strided<[1], offset: ?>>
            %70 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<128x5120xf16, strided<[5120, 1], offset: ?>>
            scf.for %arg3 = %c0 to %c10 step %c1 {
            %row = affine.apply #map1()[%thread_id_x,%block_id_y]
            %col = affine.apply #map()[%thread_id_x, %arg3]
            %val = vector.load %input[%row, %col] : memref<128x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
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
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        gamma: tkl.Memory[N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        length_embedding = tkl.Register[M, tkl.f16](EMB_SIZE)
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        lhs_pow = lhs * lhs
        red = tkw.sum(lhs_pow, dim=N, block=True)
        result = red / length_embedding
        rms = tkw.sqrt(result)
        rms_broad = tkw.broadcast(rms, [M, N])
        a_scaled = lhs / rms_broad
        gamma_reg = tkw.read(gamma, elements_per_thread=ELEMS_PER_THREAD)
        gamma_broad = tkw.broadcast(gamma_reg, [M, N])
        output = a_scaled * gamma_broad
        tkw.write(output, c, elements_per_thread=ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            TOKENS_PER_WK: 1,
            EMB_SIZE: shape[1],
            ELEMS_PER_THREAD: 8,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        override_mlir=override_mlir_str1_dpp,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)
    print(test.asm)
    c = device_zeros(shape, dtype=torch.float16)
    a, gamma = generate_inputs(shape)

    # for _ in range(10):
    #     test(a, gamma, c)

    # torch.cuda.synchronize()
    # start = time.time()
    # test(a, gamma, c)
    # torch.cuda.synchronize()
    # end = time.time()

    # latency_ms = (end - start) * 10000
    print("Wave RMSnorm")
    # print(f"[Wave] Latency per token: {latency_ms:.3f} μs")
    # print(f"[Wave] Tokens/sec: {1 / (end - start):.1f}")

    for _ in range(10):
        test(a, gamma, c)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("wave_rmsnorm"):
            test(a, gamma, c)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    rms = torch.sqrt(torch.mean(a * a, dim=-1, keepdim=True))
    ref = (a / rms) * gamma
    torch.testing.assert_close(ref, c, atol=0.1, rtol=1e-05)


def test_rmsnorm_pytorch():
    shape = (1, 5120)
    a, gamma = generate_inputs(shape)
    print("Pytorch RMSnorm", flush=True)
    eps = 1e-5

    def rmsnorm(x, weight, eps):
        rms = (x**2).mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + eps) * weight

    for _ in range(10):
        rmsnorm(a, gamma, eps)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("rmsnorm_pytorch"):
            out = rmsnorm(a, gamma, eps)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    rms = torch.sqrt(torch.mean(a * a, dim=-1, keepdim=True))
    ref = (a / rms) * gamma
    print(out)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-2)


def test_rmsnorm_torch_compile():
    shape = (1, 5120)
    a, gamma = generate_inputs(shape)

    class RMSNormModule(nn.Module):
        def __init__(self, eps=1e-5):
            super().__init__()
            self.eps = eps

        def forward(self, x, weight):
            rms = (x**2).mean(dim=-1, keepdim=True).sqrt()
            return x / (rms + self.eps) * weight

    compiled_rms = torch.compile(RMSNormModule())
    for _ in range(10):
        compiled_rms(a, gamma)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("rmsnorm_compiled"):
            out = compiled_rms(a, gamma)
    print("\n[Torch.compile RMSNorm]")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(out)
    rms = torch.sqrt(torch.mean(a * a, dim=-1, keepdim=True))
    ref = (a / rms) * gamma
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-2)


def generate_inputs(shape, seed=2):
    torch.manual_seed(seed)
    a = device_randint(4, shape, dtype=torch.float16)
    gamma = device_randint(10, (shape[1],), dtype=torch.float16)
    return a, gamma


def test_torch_layernorm():
    print("\n[Torch LayerNorm]", flush=True)
    shape = (1, 5120)
    a, _ = generate_inputs(shape)
    B, H = shape
    layernorm = nn.LayerNorm(H, eps=1e-5).to("cuda", dtype=torch.float16)

    for _ in range(10):
        layernorm(a)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("torch_layernorm"):
            out = layernorm(a)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(out)
    return None


if __name__ == "__main__":
    print("Running benchmarks with shared inputs")

    out_pt = test_rmsnorm_pytorch()
    out_compile = test_rmsnorm_torch_compile()
    out_layer = test_torch_layernorm()

    print(out_pt)
    print(out_compile)
    print(out_layer)
