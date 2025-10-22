import triton
import triton.language as tl
import torch
import itertools
import math

import triton.compiler as tc
from .torch_kernels import moe_align_block_size_pytorch
from pathlib import Path
import datetime as dt
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



def build_wave_moe_gemm_fixed_fp16_33x2_E8_K128_N256():
    """
    Returns a compiled Wave kernel specialized to:
      M=33, TOPK=2, E=8, K=128, N=256, BLOCK=64, BLOCK_K=32, EM=633, MAX_M_BLOCKS=10, dtype=fp16.

    You *must* call it with tensors of the exact shapes/dtypes shown below.
    """
    
    
    asm_dtype0_256_128_8_64_2_33_mfma = (
            """
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

        #map_load_row = affine_map<()[s0] -> (s0 mod 16)>
        #map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

        #map_store_col = affine_map<()[s0] -> (s0 mod 16)>
        #map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

        module attributes {transform.with_named_sequence} {
        stream.executable private @fused_moe_kernel {
            stream.executable.export public @fused_moe_kernel workgroups() -> (index, index, index) {
            %c40 = arith.constant 40 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            stream.return %c40, %c1, %c1 : index, index, index
            }
            builtin.module {
            func.func @fused_moe_kernel(
                // Input memrefs
            // %a_ptr: memref<33x128xf16>,
            // %b_ptr: memref<8x256x128xf16>,
            // %sorted_token_ids_ptr: memref<633xi32>,
            // %expert_ids_ptr: memref<10xi32>,
            // %num_tokens_post_padded_ptr: memref<1xi32>,
            // %c_ptr: memref<33x2x256xf16>
                %arg0: !stream.binding,
                %arg1: !stream.binding,
                %arg2: !stream.binding,
                %arg3: !stream.binding,
                %arg4: !stream.binding,
                %arg5: !stream.binding
            ) attributes {translation_info = #translation} {
                // N = 256
                // K = 128
                // EM = 633
                // top_k = 2
                // num_valid_tokens = 66
                // GROUP_SIZE_M = 8
                // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
                // BLOCK_SIZE_K = 32
                %N = arith.constant 256 : index
                %K = arith.constant 128 : index
                %EM = arith.constant 633 : index
                %top_k = arith.constant 2 : index
                %num_valid_tokens = arith.constant 66 : index
                %GROUP_SIZE_M = arith.constant 8 : index
                %BLOCK_SIZE_M = arith.constant 64 : index
                %BLOCK_SIZE_N = arith.constant 64 : index
                %BLOCK_SIZE_K = arith.constant 32 : index

                %c16384 = arith.constant 16384 : index
                %c32768 = arith.constant 32768 : index
                %c0 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                %c2 = arith.constant 2 : index
                %c3 = arith.constant 3 : index
                %c16 = arith.constant 16 : index
                %c32 = arith.constant 32 : index
                %c48 = arith.constant 48 : index
                %c63 = arith.constant 63 : index
                %c127 = arith.constant 127 : index
                %f0 = arith.constant 0.0 : f32
                %f0_f16 = arith.constant 0.0 : f16
                %cst_mfma = arith.constant dense<0.000000e+00> : vector<4xf32>

                %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<33x128xf16>
                %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x256x128xf16>
                %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<33x2x256xf16>
                %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<633xi32>
                %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<10xi32>
                %num_tokens_post_padded_ptr = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1xi32>

                // Program ID mapping
                %pid = gpu.block_id x
                %num_pid_m = arith.ceildivui %EM, %BLOCK_SIZE_M : index
                %num_pid_n = arith.ceildivui %N, %BLOCK_SIZE_N : index
                %num_pid_in_group = arith.muli %GROUP_SIZE_M, %num_pid_n : index
                %group_id = arith.divui %pid, %num_pid_in_group : index
                %first_pid_m = arith.muli %group_id, %GROUP_SIZE_M : index
                %min_group_size_m = arith.subi %num_pid_m, %first_pid_m : index
                %group_size_m = arith.minui %GROUP_SIZE_M, %min_group_size_m : index
                %0 = arith.remsi %pid, %num_pid_in_group : index
                %1 = arith.remsi %0, %group_size_m : index
                %pid_m = arith.addi %first_pid_m, %1 : index
                %pid_n = arith.divui %0, %group_size_m : index

                %thread_id = gpu.thread_id x upper_bound 64

                // Early exit check
                %2 = memref.load %num_tokens_post_padded_ptr[%c0] : memref<1xi32>
                %num_tokens_post_padded = arith.index_cast %2 : i32 to index
                %pid_m_offset = arith.muli %pid_m, %BLOCK_SIZE_M : index
                %should_exit = arith.cmpi sge, %pid_m_offset, %num_tokens_post_padded : index
                scf.if %should_exit {
                scf.yield
                } else {
                // Compute token mask
                %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index
                %thread_token_id = arith.addi %offs_token_id_base, %thread_id : index

                // Load token ID for this row
                %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<633xi32>
                %token_id = arith.index_cast %token_id_val : i32 to index

                %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
                %token_mask = vector.broadcast %token_valid : i1 to vector<128xi1>

                // Compute A row index: token_id // top_k
                %a_row = arith.divui %token_id, %top_k : index
                
                //add safe guard check for OOB prevention
                %last_row = arith.constant 32 : index
                %a_row_clamped = arith.minsi %a_row, %last_row : index


                // Load expert ID
                %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<10xi32>
                %expert_id = arith.index_cast %expert_id_val : i32 to index

                // Compute B row offset for this thread
                %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
                %b_row = arith.addi %offs_bn_base, %thread_id : index

                // Allocate shared memory: 64×128 for A, 64×128 for B
                %alloc = memref.alloc() : memref<32768xi8, #gpu.address_space<workgroup>>
                %shared_a = memref.view %alloc[%c0][] : memref<32768xi8, #gpu.address_space<workgroup>>
                    to memref<64x128xf16, #gpu.address_space<workgroup>>
                %shared_b = memref.view %alloc[%c16384][] : memref<32768xi8, #gpu.address_space<workgroup>>
                    to memref<64x128xf16, #gpu.address_space<workgroup>>
        //%alloc_c = memref.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
        //%shared_c = memref.view %alloc_c[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>>
        //  to memref<64x64xf16, #gpu.address_space<workgroup>>

                // Each thread loads its full row from A (128 f16)
                %a_row_vec = vector.transfer_read %a_ptr[%a_row_clamped, %c0], %f0_f16, %token_mask :
                    memref<33x128xf16>, vector<128xf16>
                // Store to shared memory
                vector.store %a_row_vec, %shared_a[%thread_id, %c0] :
                    memref<64x128xf16, #gpu.address_space<workgroup>>, vector<128xf16>

                // Each thread loads its row from B (128 f16)
                // B is [8, 256, 128], we need [expert_id, b_row, :]
                // Note: b_row is always < 256 since pid_n * 64 + thread_id_x < 256
                %b_row_vec = vector.transfer_read %b_ptr[%expert_id, %b_row, %c0], %f0_f16 :
                    memref<8x256x128xf16>, vector<128xf16>
                // Store to shared memory
                vector.store %b_row_vec, %shared_b[%thread_id, %c0] :
                    memref<64x128xf16, #gpu.address_space<workgroup>>, vector<128xf16>

                amdgpu.lds_barrier

        //amdgpu.lds_barrier
                // Thread-level indices for MFMA loading
                %load_col = affine.apply #map_load_col()[%thread_id]
                %load_row = affine.apply #map_load_row()[%thread_id]
                %load_row_1 = arith.addi %load_row, %c16 : index
                %load_row_2 = arith.addi %load_row, %c32 : index
                %load_row_3 = arith.addi %load_row, %c48 : index
        //gpu.printf "T%d load_col %d load_row %d\\n", %thread_id, %load_col, %load_row : index, index, index

                // =========================================================================
                // MFMA COMPUTATION
                // =========================================================================
                %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index

                %result:16 = scf.for %k_block = %c0 to %num_blocks step %c1
                    iter_args(%a00=%cst_mfma, %a01=%cst_mfma, %a02=%cst_mfma, %a03=%cst_mfma,
                                %a10=%cst_mfma, %a11=%cst_mfma, %a12=%cst_mfma, %a13=%cst_mfma,
                                %a20=%cst_mfma, %a21=%cst_mfma, %a22=%cst_mfma, %a23=%cst_mfma,
                                %a30=%cst_mfma, %a31=%cst_mfma, %a32=%cst_mfma, %a33=%cst_mfma)
                    -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {

                    // Compute K offset for this iteration
                    %k_start = arith.muli %k_block, %BLOCK_SIZE_K : index
                    %k_col = arith.addi %k_start, %load_col : index
                    %k_col_k = arith.addi %k_col, %c16 : index

                    // Load A vectors: 4 M tiles × 2 K slices (columns k_col and k_col+16)
                    %a0 = vector.load %shared_a[%load_row, %k_col] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a1 = vector.load %shared_a[%load_row_1, %k_col] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a2 = vector.load %shared_a[%load_row_2, %k_col] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a3 = vector.load %shared_a[%load_row_3, %k_col] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    %a0k = vector.load %shared_a[%load_row, %k_col_k] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a1k = vector.load %shared_a[%load_row_1, %k_col_k] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a2k = vector.load %shared_a[%load_row_2, %k_col_k] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a3k = vector.load %shared_a[%load_row_3, %k_col_k] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // Load B vectors: 4 N tiles × 2 K slices
                    // Note: B is stored as [64, 128] where rows are output features
                    // For MFMA, we need B[n, k], which maps to shared_b[row, k_col]
                    %b0 = vector.load %shared_b[%load_row, %k_col] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b1 = vector.load %shared_b[%load_row_1, %k_col] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b2 = vector.load %shared_b[%load_row_2, %k_col] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b3 = vector.load %shared_b[%load_row_3, %k_col] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    %b0k = vector.load %shared_b[%load_row, %k_col_k] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b1k = vector.load %shared_b[%load_row_1, %k_col_k] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b2k = vector.load %shared_b[%load_row_2, %k_col_k] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b3k = vector.load %shared_b[%load_row_3, %k_col_k] :
                        memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // MFMA operations: 4×4 tile grid
                    // Tile (0,0)
                    %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (0,1)
                    %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (0,2)
                    %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (0,3)
                    %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,0)
                    %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,1)
                    %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,2)
                    %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,3)
                    %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,0)
                    %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,1)
                    %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,2)
                    %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,3)
                    %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,0)
                    %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,1)
                    %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,2)
                    %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,3)
                    %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                            %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
                }

                // =========================================================================
                // STORE RESULTS
                // =========================================================================

                // Truncate to f16
                %r00_f16 = arith.truncf %result#0 : vector<4xf32> to vector<4xf16>
                %r01_f16 = arith.truncf %result#1 : vector<4xf32> to vector<4xf16>
                %r02_f16 = arith.truncf %result#2 : vector<4xf32> to vector<4xf16>
                %r03_f16 = arith.truncf %result#3 : vector<4xf32> to vector<4xf16>
                %r10_f16 = arith.truncf %result#4 : vector<4xf32> to vector<4xf16>
                %r11_f16 = arith.truncf %result#5 : vector<4xf32> to vector<4xf16>
                %r12_f16 = arith.truncf %result#6 : vector<4xf32> to vector<4xf16>
                %r13_f16 = arith.truncf %result#7 : vector<4xf32> to vector<4xf16>
                %r20_f16 = arith.truncf %result#8 : vector<4xf32> to vector<4xf16>
                %r21_f16 = arith.truncf %result#9 : vector<4xf32> to vector<4xf16>
                %r22_f16 = arith.truncf %result#10 : vector<4xf32> to vector<4xf16>
                %r23_f16 = arith.truncf %result#11 : vector<4xf32> to vector<4xf16>
                %r30_f16 = arith.truncf %result#12 : vector<4xf32> to vector<4xf16>
                %r31_f16 = arith.truncf %result#13 : vector<4xf32> to vector<4xf16>
                %r32_f16 = arith.truncf %result#14 : vector<4xf32> to vector<4xf16>
                %r33_f16 = arith.truncf %result#15 : vector<4xf32> to vector<4xf16>

                %store_col_0 = affine.apply #map_store_col()[%thread_id]
                %store_col_1 = arith.addi %store_col_0, %c16 : index
                %store_col_2 = arith.addi %store_col_0, %c32 : index
                %store_col_3 = arith.addi %store_col_0, %c48 : index
                %store_row_0_0 = affine.apply #map_store_row()[%thread_id]
                %store_row_0_1 = arith.addi %store_row_0_0, %c1 : index
                %store_row_0_2 = arith.addi %store_row_0_0, %c2 : index
                %store_row_0_3 = arith.addi %store_row_0_0, %c3 : index
                %store_row_16_0 = arith.addi %store_row_0_0, %c16 : index
                %store_row_16_1 = arith.addi %store_row_16_0, %c1 : index
                %store_row_16_2 = arith.addi %store_row_16_0, %c2 : index
                %store_row_16_3 = arith.addi %store_row_16_0, %c3 : index
                %store_row_32_0 = arith.addi %store_row_0_0, %c32 : index
                %store_row_32_1 = arith.addi %store_row_32_0, %c1 : index
                %store_row_32_2 = arith.addi %store_row_32_0, %c2 : index
                %store_row_32_3 = arith.addi %store_row_32_0, %c3 : index
                %store_row_48_0 = arith.addi %store_row_0_0, %c48 : index
                %store_row_48_1 = arith.addi %store_row_48_0, %c1 : index
                %store_row_48_2 = arith.addi %store_row_48_0, %c2 : index
                %store_row_48_3 = arith.addi %store_row_48_0, %c3 : index

                %r00_0_f16 = vector.extract_strided_slice %r00_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_1_f16 = vector.extract_strided_slice %r00_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_2_f16 = vector.extract_strided_slice %r00_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_3_f16 = vector.extract_strided_slice %r00_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_0_f16 = vector.extract_strided_slice %r01_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_1_f16 = vector.extract_strided_slice %r01_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_2_f16 = vector.extract_strided_slice %r01_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_3_f16 = vector.extract_strided_slice %r01_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_0_f16 = vector.extract_strided_slice %r02_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_1_f16 = vector.extract_strided_slice %r02_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_2_f16 = vector.extract_strided_slice %r02_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_3_f16 = vector.extract_strided_slice %r02_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_0_f16 = vector.extract_strided_slice %r03_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_1_f16 = vector.extract_strided_slice %r03_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_2_f16 = vector.extract_strided_slice %r03_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_3_f16 = vector.extract_strided_slice %r03_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r10_0_f16 = vector.extract_strided_slice %r10_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_1_f16 = vector.extract_strided_slice %r10_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_2_f16 = vector.extract_strided_slice %r10_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_3_f16 = vector.extract_strided_slice %r10_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_0_f16 = vector.extract_strided_slice %r11_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_1_f16 = vector.extract_strided_slice %r11_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_2_f16 = vector.extract_strided_slice %r11_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_3_f16 = vector.extract_strided_slice %r11_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_0_f16 = vector.extract_strided_slice %r12_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_1_f16 = vector.extract_strided_slice %r12_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_2_f16 = vector.extract_strided_slice %r12_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_3_f16 = vector.extract_strided_slice %r12_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_0_f16 = vector.extract_strided_slice %r13_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_1_f16 = vector.extract_strided_slice %r13_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_2_f16 = vector.extract_strided_slice %r13_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_3_f16 = vector.extract_strided_slice %r13_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r20_0_f16 = vector.extract_strided_slice %r20_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_1_f16 = vector.extract_strided_slice %r20_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_2_f16 = vector.extract_strided_slice %r20_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_3_f16 = vector.extract_strided_slice %r20_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_0_f16 = vector.extract_strided_slice %r21_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_1_f16 = vector.extract_strided_slice %r21_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_2_f16 = vector.extract_strided_slice %r21_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_3_f16 = vector.extract_strided_slice %r21_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_0_f16 = vector.extract_strided_slice %r22_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_1_f16 = vector.extract_strided_slice %r22_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_2_f16 = vector.extract_strided_slice %r22_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_3_f16 = vector.extract_strided_slice %r22_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_0_f16 = vector.extract_strided_slice %r23_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_1_f16 = vector.extract_strided_slice %r23_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_2_f16 = vector.extract_strided_slice %r23_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_3_f16 = vector.extract_strided_slice %r23_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r30_0_f16 = vector.extract_strided_slice %r30_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_1_f16 = vector.extract_strided_slice %r30_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_2_f16 = vector.extract_strided_slice %r30_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_3_f16 = vector.extract_strided_slice %r30_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_0_f16 = vector.extract_strided_slice %r31_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_1_f16 = vector.extract_strided_slice %r31_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_2_f16 = vector.extract_strided_slice %r31_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_3_f16 = vector.extract_strided_slice %r31_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_0_f16 = vector.extract_strided_slice %r32_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_1_f16 = vector.extract_strided_slice %r32_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_2_f16 = vector.extract_strided_slice %r32_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_3_f16 = vector.extract_strided_slice %r32_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_0_f16 = vector.extract_strided_slice %r33_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_1_f16 = vector.extract_strided_slice %r33_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_2_f16 = vector.extract_strided_slice %r33_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_3_f16 = vector.extract_strided_slice %r33_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>


                // Flatten c_ptr for easier indexing
                %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<33x2x256xf16> into memref<16896xf16>

                // Each thread writes to 4 different rows (from load_row, load_row+16, load_row+32, load_row+48)
                // across 4 column groups (base, base+16, base+32, base+48)

                // Get token indices for output rows
                %out_token_0_0 = arith.addi %offs_token_id_base, %store_row_0_0 : index
                %out_token_0_1 = arith.addi %offs_token_id_base, %store_row_0_1 : index
                %out_token_0_2 = arith.addi %offs_token_id_base, %store_row_0_2 : index
                %out_token_0_3 = arith.addi %offs_token_id_base, %store_row_0_3 : index
                %out_token_16_0 = arith.addi %offs_token_id_base, %store_row_16_0 : index
                %out_token_16_1 = arith.addi %offs_token_id_base, %store_row_16_1 : index
                %out_token_16_2 = arith.addi %offs_token_id_base, %store_row_16_2 : index
                %out_token_16_3 = arith.addi %offs_token_id_base, %store_row_16_3 : index
                %out_token_32_0 = arith.addi %offs_token_id_base, %store_row_32_0 : index
                %out_token_32_1 = arith.addi %offs_token_id_base, %store_row_32_1 : index
                %out_token_32_2 = arith.addi %offs_token_id_base, %store_row_32_2 : index
                %out_token_32_3 = arith.addi %offs_token_id_base, %store_row_32_3 : index
                %out_token_48_0 = arith.addi %offs_token_id_base, %store_row_48_0 : index
                %out_token_48_1 = arith.addi %offs_token_id_base, %store_row_48_1 : index
                %out_token_48_2 = arith.addi %offs_token_id_base, %store_row_48_2 : index
                %out_token_48_3 = arith.addi %offs_token_id_base, %store_row_48_3 : index

                %tok_id_0_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_0] : memref<633xi32>
                %tok_id_0_0 = arith.index_cast %tok_id_0_0_i32 : i32 to index
                %out_base_0_0 = arith.muli %tok_id_0_0, %N : index
                %tok_id_0_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_1] : memref<633xi32>
                %tok_id_0_1 = arith.index_cast %tok_id_0_1_i32 : i32 to index
                %out_base_0_1 = arith.muli %tok_id_0_1, %N : index
                %tok_id_0_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_2] : memref<633xi32>
                %tok_id_0_2 = arith.index_cast %tok_id_0_2_i32 : i32 to index
                %out_base_0_2 = arith.muli %tok_id_0_2, %N : index
                %tok_id_0_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_3] : memref<633xi32>
                %tok_id_0_3 = arith.index_cast %tok_id_0_3_i32 : i32 to index
                %out_base_0_3 = arith.muli %tok_id_0_3, %N : index

                %tok_id_16_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_0] : memref<633xi32>
                %tok_id_16_0 = arith.index_cast %tok_id_16_0_i32 : i32 to index
                %out_base_16_0 = arith.muli %tok_id_16_0, %N : index
                %tok_id_16_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_1] : memref<633xi32>
                %tok_id_16_1 = arith.index_cast %tok_id_16_1_i32 : i32 to index
                %out_base_16_1 = arith.muli %tok_id_16_1, %N : index
                %tok_id_16_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_2] : memref<633xi32>
                %tok_id_16_2 = arith.index_cast %tok_id_16_2_i32 : i32 to index
                %out_base_16_2 = arith.muli %tok_id_16_2, %N : index
                %tok_id_16_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_3] : memref<633xi32>
                %tok_id_16_3 = arith.index_cast %tok_id_16_3_i32 : i32 to index
                %out_base_16_3 = arith.muli %tok_id_16_3, %N : index

                %tok_id_32_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_0] : memref<633xi32>
                %tok_id_32_0 = arith.index_cast %tok_id_32_0_i32 : i32 to index
                %out_base_32_0 = arith.muli %tok_id_32_0, %N : index
                %tok_id_32_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_1] : memref<633xi32>
                %tok_id_32_1 = arith.index_cast %tok_id_32_1_i32 : i32 to index
                %out_base_32_1 = arith.muli %tok_id_32_1, %N : index
                %tok_id_32_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_2] : memref<633xi32>
                %tok_id_32_2 = arith.index_cast %tok_id_32_2_i32 : i32 to index
                %out_base_32_2 = arith.muli %tok_id_32_2, %N : index
                %tok_id_32_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_3] : memref<633xi32>
                %tok_id_32_3 = arith.index_cast %tok_id_32_3_i32 : i32 to index
                %out_base_32_3 = arith.muli %tok_id_32_3, %N : index

                %tok_id_48_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_0] : memref<633xi32>
                %tok_id_48_0 = arith.index_cast %tok_id_48_0_i32 : i32 to index
                %out_base_48_0 = arith.muli %tok_id_48_0, %N : index
                %tok_id_48_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_1] : memref<633xi32>
                %tok_id_48_1 = arith.index_cast %tok_id_48_1_i32 : i32 to index
                %out_base_48_1 = arith.muli %tok_id_48_1, %N : index
                %tok_id_48_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_2] : memref<633xi32>
                %tok_id_48_2 = arith.index_cast %tok_id_48_2_i32 : i32 to index
                %out_base_48_2 = arith.muli %tok_id_48_2, %N : index
                %tok_id_48_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_3] : memref<633xi32>
                %tok_id_48_3 = arith.index_cast %tok_id_48_3_i32 : i32 to index
                %out_base_48_3 = arith.muli %tok_id_48_3, %N : index

                // pid_n determines which 64-neuron block we're computing
                %out_col_base = arith.muli %pid_n, %BLOCK_SIZE_N : index

                // Column offsets for the 4 column tiles
                %out_col_0 = arith.addi %out_col_base, %store_col_0 : index
                %out_col_1 = arith.addi %out_col_base, %store_col_1 : index
                %out_col_2 = arith.addi %out_col_base, %store_col_2 : index
                %out_col_3 = arith.addi %out_col_base, %store_col_3 : index

                // Write all 16 tiles using vector.store
                // Tile (0,0)
                %idx_00_0 = arith.addi %out_base_0_0, %out_col_0 : index
                vector.store %r00_0_f16, %c_flat[%idx_00_0] : memref<16896xf16>, vector<1xf16>
                %idx_00_1 = arith.addi %out_base_0_1, %out_col_0 : index
                vector.store %r00_1_f16, %c_flat[%idx_00_1] : memref<16896xf16>, vector<1xf16>
                %idx_00_2 = arith.addi %out_base_0_2, %out_col_0 : index
                vector.store %r00_2_f16, %c_flat[%idx_00_2] : memref<16896xf16>, vector<1xf16>
                %idx_00_3 = arith.addi %out_base_0_3, %out_col_0 : index
                vector.store %r00_3_f16, %c_flat[%idx_00_3] : memref<16896xf16>, vector<1xf16>

                // Tile (0,1)
                %idx_01_0 = arith.addi %out_base_0_0, %out_col_1 : index
                vector.store %r01_0_f16, %c_flat[%idx_01_0] : memref<16896xf16>, vector<1xf16>
                %idx_01_1 = arith.addi %out_base_0_1, %out_col_1 : index
                vector.store %r01_1_f16, %c_flat[%idx_01_1] : memref<16896xf16>, vector<1xf16>
                %idx_01_2 = arith.addi %out_base_0_2, %out_col_1 : index
                vector.store %r01_2_f16, %c_flat[%idx_01_2] : memref<16896xf16>, vector<1xf16>
                %idx_01_3 = arith.addi %out_base_0_3, %out_col_1 : index
                vector.store %r01_3_f16, %c_flat[%idx_01_3] : memref<16896xf16>, vector<1xf16>

                // Tile (0,2)
                %idx_02_0 = arith.addi %out_base_0_0, %out_col_2 : index
                vector.store %r02_0_f16, %c_flat[%idx_02_0] : memref<16896xf16>, vector<1xf16>
                %idx_02_1 = arith.addi %out_base_0_1, %out_col_2 : index
                vector.store %r02_1_f16, %c_flat[%idx_02_1] : memref<16896xf16>, vector<1xf16>
                %idx_02_2 = arith.addi %out_base_0_2, %out_col_2 : index
                vector.store %r02_2_f16, %c_flat[%idx_02_2] : memref<16896xf16>, vector<1xf16>
                %idx_02_3 = arith.addi %out_base_0_3, %out_col_2 : index
                vector.store %r02_3_f16, %c_flat[%idx_02_3] : memref<16896xf16>, vector<1xf16>

                // Tile (0,3)
                %idx_03_0 = arith.addi %out_base_0_0, %out_col_3 : index
                vector.store %r03_0_f16, %c_flat[%idx_03_0] : memref<16896xf16>, vector<1xf16>
                %idx_03_1 = arith.addi %out_base_0_1, %out_col_3 : index
                vector.store %r03_1_f16, %c_flat[%idx_03_1] : memref<16896xf16>, vector<1xf16>
                %idx_03_2 = arith.addi %out_base_0_2, %out_col_3 : index
                vector.store %r03_2_f16, %c_flat[%idx_03_2] : memref<16896xf16>, vector<1xf16>
                %idx_03_3 = arith.addi %out_base_0_3, %out_col_3 : index
                vector.store %r03_3_f16, %c_flat[%idx_03_3] : memref<16896xf16>, vector<1xf16>

                // Tile (1,0)
                %idx_10_0 = arith.addi %out_base_16_0, %out_col_0 : index
                vector.store %r10_0_f16, %c_flat[%idx_10_0] : memref<16896xf16>, vector<1xf16>
                %idx_10_1 = arith.addi %out_base_16_1, %out_col_0 : index
                vector.store %r10_1_f16, %c_flat[%idx_10_1] : memref<16896xf16>, vector<1xf16>
                %idx_10_2 = arith.addi %out_base_16_2, %out_col_0 : index
                vector.store %r10_2_f16, %c_flat[%idx_10_2] : memref<16896xf16>, vector<1xf16>
                %idx_10_3 = arith.addi %out_base_16_3, %out_col_0 : index
                vector.store %r10_3_f16, %c_flat[%idx_10_3] : memref<16896xf16>, vector<1xf16>

                // Tile (1,1)
                %idx_11_0 = arith.addi %out_base_16_0, %out_col_1 : index
                vector.store %r11_0_f16, %c_flat[%idx_11_0] : memref<16896xf16>, vector<1xf16>
                %idx_11_1 = arith.addi %out_base_16_1, %out_col_1 : index
                vector.store %r11_1_f16, %c_flat[%idx_11_1] : memref<16896xf16>, vector<1xf16>
                %idx_11_2 = arith.addi %out_base_16_2, %out_col_1 : index
                vector.store %r11_2_f16, %c_flat[%idx_11_2] : memref<16896xf16>, vector<1xf16>
                %idx_11_3 = arith.addi %out_base_16_3, %out_col_1 : index
                vector.store %r11_3_f16, %c_flat[%idx_11_3] : memref<16896xf16>, vector<1xf16>

                // Tile (1,2)
                %idx_12_0 = arith.addi %out_base_16_0, %out_col_2 : index
                vector.store %r12_0_f16, %c_flat[%idx_12_0] : memref<16896xf16>, vector<1xf16>
                %idx_12_1 = arith.addi %out_base_16_1, %out_col_2 : index
                vector.store %r12_1_f16, %c_flat[%idx_12_1] : memref<16896xf16>, vector<1xf16>
                %idx_12_2 = arith.addi %out_base_16_2, %out_col_2 : index
                vector.store %r12_2_f16, %c_flat[%idx_12_2] : memref<16896xf16>, vector<1xf16>
                %idx_12_3 = arith.addi %out_base_16_3, %out_col_2 : index
                vector.store %r12_3_f16, %c_flat[%idx_12_3] : memref<16896xf16>, vector<1xf16>

                // Tile (1,3)
                %idx_13_0 = arith.addi %out_base_16_0, %out_col_3 : index
                vector.store %r13_0_f16, %c_flat[%idx_13_0] : memref<16896xf16>, vector<1xf16>
                %idx_13_1 = arith.addi %out_base_16_1, %out_col_3 : index
                vector.store %r13_1_f16, %c_flat[%idx_13_1] : memref<16896xf16>, vector<1xf16>
                %idx_13_2 = arith.addi %out_base_16_2, %out_col_3 : index
                vector.store %r13_2_f16, %c_flat[%idx_13_2] : memref<16896xf16>, vector<1xf16>
                %idx_13_3 = arith.addi %out_base_16_3, %out_col_3 : index
                vector.store %r13_3_f16, %c_flat[%idx_13_3] : memref<16896xf16>, vector<1xf16>

                // Tile (2,0)
                %idx_20_0 = arith.addi %out_base_32_0, %out_col_0 : index
                vector.store %r20_0_f16, %c_flat[%idx_20_0] : memref<16896xf16>, vector<1xf16>
                %idx_20_1 = arith.addi %out_base_32_1, %out_col_0 : index
                vector.store %r20_1_f16, %c_flat[%idx_20_1] : memref<16896xf16>, vector<1xf16>
                %idx_20_2 = arith.addi %out_base_32_2, %out_col_0 : index
                vector.store %r20_2_f16, %c_flat[%idx_20_2] : memref<16896xf16>, vector<1xf16>
                %idx_20_3 = arith.addi %out_base_32_3, %out_col_0 : index
                vector.store %r20_3_f16, %c_flat[%idx_20_3] : memref<16896xf16>, vector<1xf16>

                // Tile (2,1)
                %idx_21_0 = arith.addi %out_base_32_0, %out_col_1 : index
                vector.store %r21_0_f16, %c_flat[%idx_21_0] : memref<16896xf16>, vector<1xf16>
                %idx_21_1 = arith.addi %out_base_32_1, %out_col_1 : index
                vector.store %r21_1_f16, %c_flat[%idx_21_1] : memref<16896xf16>, vector<1xf16>
                %idx_21_2 = arith.addi %out_base_32_2, %out_col_1 : index
                vector.store %r21_2_f16, %c_flat[%idx_21_2] : memref<16896xf16>, vector<1xf16>
                %idx_21_3 = arith.addi %out_base_32_3, %out_col_1 : index
                vector.store %r21_3_f16, %c_flat[%idx_21_3] : memref<16896xf16>, vector<1xf16>

                // Tile (2,2)
                %idx_22_0 = arith.addi %out_base_32_0, %out_col_2 : index
                vector.store %r22_0_f16, %c_flat[%idx_22_0] : memref<16896xf16>, vector<1xf16>
                %idx_22_1 = arith.addi %out_base_32_1, %out_col_2 : index
                vector.store %r22_1_f16, %c_flat[%idx_22_1] : memref<16896xf16>, vector<1xf16>
                %idx_22_2 = arith.addi %out_base_32_2, %out_col_2 : index
                vector.store %r22_2_f16, %c_flat[%idx_22_2] : memref<16896xf16>, vector<1xf16>
                %idx_22_3 = arith.addi %out_base_32_3, %out_col_2 : index
                vector.store %r22_3_f16, %c_flat[%idx_22_3] : memref<16896xf16>, vector<1xf16>

                // Tile (2,3)
                %idx_23_0 = arith.addi %out_base_32_0, %out_col_3 : index
                vector.store %r23_0_f16, %c_flat[%idx_23_0] : memref<16896xf16>, vector<1xf16>
                %idx_23_1 = arith.addi %out_base_32_1, %out_col_3 : index
                vector.store %r23_1_f16, %c_flat[%idx_23_1] : memref<16896xf16>, vector<1xf16>
                %idx_23_2 = arith.addi %out_base_32_2, %out_col_3 : index
                vector.store %r23_2_f16, %c_flat[%idx_23_2] : memref<16896xf16>, vector<1xf16>
                %idx_23_3 = arith.addi %out_base_32_3, %out_col_3 : index
                vector.store %r23_3_f16, %c_flat[%idx_23_3] : memref<16896xf16>, vector<1xf16>

                // Tile (3,0)
                %idx_30_0 = arith.addi %out_base_48_0, %out_col_0 : index
                vector.store %r30_0_f16, %c_flat[%idx_30_0] : memref<16896xf16>, vector<1xf16>
                %idx_30_1 = arith.addi %out_base_48_1, %out_col_0 : index
                vector.store %r30_1_f16, %c_flat[%idx_30_1] : memref<16896xf16>, vector<1xf16>
                %idx_30_2 = arith.addi %out_base_48_2, %out_col_0 : index
                vector.store %r30_2_f16, %c_flat[%idx_30_2] : memref<16896xf16>, vector<1xf16>
                %idx_30_3 = arith.addi %out_base_48_3, %out_col_0 : index
                vector.store %r30_3_f16, %c_flat[%idx_30_3] : memref<16896xf16>, vector<1xf16>

                // Tile (3,1)
                %idx_31_0 = arith.addi %out_base_48_0, %out_col_1 : index
                vector.store %r31_0_f16, %c_flat[%idx_31_0] : memref<16896xf16>, vector<1xf16>
                %idx_31_1 = arith.addi %out_base_48_1, %out_col_1 : index
                vector.store %r31_1_f16, %c_flat[%idx_31_1] : memref<16896xf16>, vector<1xf16>
                %idx_31_2 = arith.addi %out_base_48_2, %out_col_1 : index
                vector.store %r31_2_f16, %c_flat[%idx_31_2] : memref<16896xf16>, vector<1xf16>
                %idx_31_3 = arith.addi %out_base_48_3, %out_col_1 : index
                vector.store %r31_3_f16, %c_flat[%idx_31_3] : memref<16896xf16>, vector<1xf16>

                // Tile (3,2)
                %idx_32_0 = arith.addi %out_base_48_0, %out_col_2 : index
                vector.store %r32_0_f16, %c_flat[%idx_32_0] : memref<16896xf16>, vector<1xf16>
                %idx_32_1 = arith.addi %out_base_48_1, %out_col_2 : index
                vector.store %r32_1_f16, %c_flat[%idx_32_1] : memref<16896xf16>, vector<1xf16>
                %idx_32_2 = arith.addi %out_base_48_2, %out_col_2 : index
                vector.store %r32_2_f16, %c_flat[%idx_32_2] : memref<16896xf16>, vector<1xf16>
                %idx_32_3 = arith.addi %out_base_48_3, %out_col_2 : index
                vector.store %r32_3_f16, %c_flat[%idx_32_3] : memref<16896xf16>, vector<1xf16>

                // Tile (3,3)
                %idx_33_0 = arith.addi %out_base_48_0, %out_col_3 : index
                vector.store %r33_0_f16, %c_flat[%idx_33_0] : memref<16896xf16>, vector<1xf16>
                %idx_33_1 = arith.addi %out_base_48_1, %out_col_3 : index
                vector.store %r33_1_f16, %c_flat[%idx_33_1] : memref<16896xf16>, vector<1xf16>
                %idx_33_2 = arith.addi %out_base_48_2, %out_col_3 : index
                vector.store %r33_2_f16, %c_flat[%idx_33_2] : memref<16896xf16>, vector<1xf16>
                %idx_33_3 = arith.addi %out_base_48_3, %out_col_3 : index
                vector.store %r33_3_f16, %c_flat[%idx_33_3] : memref<16896xf16>, vector<1xf16>
                }
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
            // %a_ptr: memref<33x128xf16>,
            // %b_ptr: memref<8x256x128xf16>,
            // %sorted_token_ids_ptr: memref<633xi32>,
            // %expert_ids_ptr: memref<10xi32>,
            // %num_tokens_post_padded_ptr: memref<1xi32>,
            // %c_ptr: memref<33x2x256xf16>
            %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<33x128xf16>
            %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x256x128xf16>
            %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<633xi32>
            %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<10xi32>
            %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
            %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<33x2x256xf16>
            %6 = flow.dispatch @fused_moe_kernel::@fused_moe_kernel(%0, %1, %2, %3, %4, %5) : (tensor<33x128xf16>, tensor<8x256x128xf16>, tensor<633xi32>, tensor<10xi32>, tensor<1xi32>, tensor<33x2x256xf16>) -> %5
            %7 = hal.tensor.barrier join(%6 : tensor<33x2x256xf16>) => %arg7 : !hal.fence
            %8 = hal.tensor.export %7 : tensor<33x2x256xf16> -> !hal.buffer_view
            return %8 : !hal.buffer_view
        }
        }
            """
        )
    # --- symbolic sizes ---
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    E = tkl.sym.E
    EM = tkl.sym.EM
    TOPK = tkl.sym.TOPK
    MAX_M_BLOCKS = tkl.sym.MAX_M_BLOCKS

    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    NUM_TOKENS_BUF_SIZE = tkl.sym.NUM_TOKENS_BUF_SIZE

    # --- constraints / layout (match your working setup) ---
    constraints: list[tkw.Constraint] = [
        # 1D grid over EM*N in tiles of BLOCK_M*BLOCK_N (like your standalone)
        tkw.WorkgroupConstraint(EM * N, BLOCK_M * BLOCK_N, 0),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            vector_shapes={M: 0, TOPK: 0, N: 32},
        ),
    ]

    @tkw.wave(constraints)
    def fused_moe_kernel(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],                  # (33,128)
        b: tkl.Memory[E, N, K, ADDRESS_SPACE, tkl.f16],               # (8,256,128)
        sorted_token_ids: tkl.Memory[EM, ADDRESS_SPACE, tkl.i32],     # (633,)
        expert_ids: tkl.Memory[MAX_M_BLOCKS, ADDRESS_SPACE, tkl.i32], # (10,)
        num_tokens_post_padded: tkl.Memory[NUM_TOKENS_BUF_SIZE, ADDRESS_SPACE, tkl.i32],  # (1,)
        c: tkl.Memory[M, TOPK, N, GLOBAL_ADDRESS_SPACE, tkl.f16],     # (33,2,256)
    ):
        # Body is irrelevant when override_mlir is supplied; keep a no-op write to satisfy tracing.
        c_reg = tkl.Register[M, TOPK, N, tkl.f16](0.0)
        tkw.write(c_reg, c)

    # --- hard-coded hyperparams for this exact config ---
    EM_VAL = 633
    #MAX_M_BLOCKS_VAL = math.ceil(EM_VAL / 64)  # 10
    
    MAX_M_BLOCKS_VAL = -(EM_VAL // -64)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: 33,
        TOPK: 2,
        E: 8,
        K: 128,
        N: 256,
        EM: EM_VAL,
        MAX_M_BLOCKS: MAX_M_BLOCKS_VAL,
        NUM_TOKENS_BUF_SIZE: 1,
    }
    hyperparams.update(get_default_scheduling_params())


    # --- compile with your override MLIR ---
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        override_mlir=asm_dtype0_256_128_8_64_2_33_mfma,               # <— you provide the MLIR blob
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,                         # matches your MLIR path
        use_scheduling_barriers=enable_scheduling_barriers,
        print_mlir=False,
    )
    options = set_default_run_config(options)

    return wave_compile(options, fused_moe_kernel)


def build_wave_moe_gemm_fixed_1024_256_8_64_2_2048(max_num_tokens_padded:int, m:int, n:int, k:int ):
    """
    Returns a compiled Wave kernel specialized to:
      M=33, TOPK=2, E=8, K=128, N=256, BLOCK=64, BLOCK_K=32, EM=633, MAX_M_BLOCKS=10, dtype=fp16.

    You *must* call it with tensors of the exact shapes/dtypes shown below.
    """
    asm_dtype0_32768_6144_8_64_2_16384_mfma = (
            """
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

        #map_load_row = affine_map<()[s0] -> (s0 mod 16)>
        #map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

        #map_store_col = affine_map<()[s0] -> (s0 mod 16)>
        #map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

        module attributes {transform.with_named_sequence} {
        stream.executable private @fused_moe_kernel {
            stream.executable.export public @fused_moe_kernel workgroups() -> (index, index, index) {
            %c266752 = arith.constant 266752 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            stream.return %c266752, %c1, %c1 : index, index, index
            }
            builtin.module {
            func.func @fused_moe_kernel(
                // Input memrefs
            // %a_ptr: memref<16384x6144xf16>,
            // %b_ptr: memref<8x32768x6144xf16>,
            // %sorted_token_ids_ptr: memref<33335xi32>,
            // %expert_ids_ptr: memref<521xi32>,
            // %num_tokens_post_padded_ptr: memref<1xi32>,
            // %c_ptr: memref<16384x2x32768xf16>
                %arg0: !stream.binding,
                %arg1: !stream.binding,
                %arg2: !stream.binding,
                %arg3: !stream.binding,
                %arg4: !stream.binding,
                %arg5: !stream.binding
            ) attributes {translation_info = #translation} {
                // N = 32768
                // K = 6144
                // EM = 33335
                // top_k = 2
                // num_valid_tokens = 32768
                // GROUP_SIZE_M = 8
                // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
                // BLOCK_SIZE_K = 32
                %N = arith.constant 32768 : index
                %K = arith.constant 6144 : index
                %EM = arith.constant 33335 : index
                %top_k = arith.constant 2 : index
                %num_valid_tokens = arith.constant 32768 : index
                %GROUP_SIZE_M = arith.constant 8 : index
                %BLOCK_SIZE_M = arith.constant 64 : index
                %BLOCK_SIZE_N = arith.constant 64 : index
                %BLOCK_SIZE_K = arith.constant 32 : index

                %c16384 = arith.constant 16384 : index
                %c32768 = arith.constant 32768 : index
                %c0 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                %c2 = arith.constant 2 : index
                %c3 = arith.constant 3 : index
                %c16 = arith.constant 16 : index
                %c32 = arith.constant 32 : index
                %c48 = arith.constant 48 : index
                %c63 = arith.constant 63 : index
                %c127 = arith.constant 127 : index
                %f0 = arith.constant 0.0 : f32
                %f0_f16 = arith.constant 0.0 : f16
                %cst_mfma = arith.constant dense<0.000000e+00> : vector<4xf32>

                %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
                %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
                %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
                %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
                %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
                %num_tokens_post_padded_ptr = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1xi32>

                // Program ID mapping
                %pid = gpu.block_id x
                %num_pid_m = arith.ceildivui %EM, %BLOCK_SIZE_M : index
                %num_pid_n = arith.ceildivui %N, %BLOCK_SIZE_N : index
                %num_pid_in_group = arith.muli %GROUP_SIZE_M, %num_pid_n : index
                %group_id = arith.divui %pid, %num_pid_in_group : index
                %first_pid_m = arith.muli %group_id, %GROUP_SIZE_M : index
                %min_group_size_m = arith.subi %num_pid_m, %first_pid_m : index
                %group_size_m = arith.minui %GROUP_SIZE_M, %min_group_size_m : index
                %0 = arith.remsi %pid, %num_pid_in_group : index
                %1 = arith.remsi %0, %group_size_m : index
                %pid_m = arith.addi %first_pid_m, %1 : index
                %pid_n = arith.divui %0, %group_size_m : index

                %thread_id = gpu.thread_id x upper_bound 64

                // Early exit check
                %2 = memref.load %num_tokens_post_padded_ptr[%c0] : memref<1xi32>
                %num_tokens_post_padded = arith.index_cast %2 : i32 to index
                %pid_m_offset = arith.muli %pid_m, %BLOCK_SIZE_M : index
                %should_exit = arith.cmpi sge, %pid_m_offset, %num_tokens_post_padded : index
                scf.if %should_exit {
                scf.yield
                } else {
                // Compute token mask
                %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index
                %thread_token_id = arith.addi %offs_token_id_base, %thread_id : index

                // Load token ID for this row
                %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
                %token_id = arith.index_cast %token_id_val : i32 to index

                %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
                %token_mask = vector.broadcast %token_valid : i1 to vector<32xi1>

                // Compute A row index: token_id // top_k
                %a_row = arith.divui %token_id, %top_k : index

                // Load expert ID
                %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
                %expert_id = arith.index_cast %expert_id_val : i32 to index

                // Compute B row offset for this thread
                %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
                %b_row = arith.addi %offs_bn_base, %thread_id : index

                // Allocate shared memory: 64×32 for A, 64×32 for B (instead of 64×6144)
                %c4096 = arith.constant 4096 : index
                %alloc = memref.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
                %shared_a = memref.view %alloc[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>>
                    to memref<64x32xf16, #gpu.address_space<workgroup>>
                %shared_b = memref.view %alloc[%c4096][] : memref<8192xi8, #gpu.address_space<workgroup>>
                    to memref<64x32xf16, #gpu.address_space<workgroup>>

        %tid_cond = arith.cmpi eq, %thread_id, %c0 : index
        %pid_cond = arith.cmpi eq, %pid, %c0 : index
        %print = arith.andi %tid_cond, %pid_cond : i1
        //
        //scf.if %print { gpu.printf "pid %d\\n", %pid : index }
        //scf.if %print { gpu.printf "pid_m %d\\n", %pid_m : index }
        //scf.if %print { gpu.printf "pid_n %d\\n",  %pid_n : index }
        //
        //%a0.0 = memref.load %shared_a[%c0, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[0][0] %f\\n", %a0.0 : f16 }
        //
        //%a0.1 = memref.load %shared_a[%c0, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[0][1] %f\\n", %a0.1 : f16 }
        //
        //%a0.127 = memref.load %shared_a[%c0, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[0][127] %f\\n", %a0.127 : f16 }
        //
        //%a1.0 = memref.load %shared_a[%c1, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[1][0] %f\\n", %a1.0 : f16 }
        //
        //%a1.1 = memref.load %shared_a[%c1, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[1][1] %f\\n", %a1.1 : f16 }
        //
        //%a1.127 = memref.load %shared_a[%c1, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[1][127] %f\\n", %a1.127 : f16 }
        //
        //%a63.0 = memref.load %shared_a[%c63, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[63][0] %f\\n", %a63.0 : f16 }
        //
        //%a63.1 = memref.load %shared_a[%c63, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[63][1] %f\\n", %a63.1 : f16 }
        //
        //%a63.127 = memref.load %shared_a[%c63, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[63][127] %f\\n", %a63.127 : f16 }
        //
        //%b0.0 = memref.load %shared_b[%c0, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[0][0] %f\\n", %b0.0 : f16 }
        //
        //%b0.1 = memref.load %shared_b[%c0, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[0][1] %f\\n", %b0.1 : f16 }
        //
        //%b0.127 = memref.load %shared_b[%c0, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[0][127] %f\\n", %b0.127 : f16 }
        //
        //%b1.0 = memref.load %shared_b[%c1, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[1][0] %f\\n", %b1.0 : f16 }
        //
        //%b1.1 = memref.load %shared_b[%c1, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[1][1] %f\\n", %b1.1 : f16 }
        //
        //%b1.127 = memref.load %shared_b[%c1, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[1][127] %f\\n", %b1.127 : f16 }
        //
        //%b63.0 = memref.load %shared_b[%c63, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[63][0] %f\\n", %b63.0 : f16 }
        //
        //%b63.1 = memref.load %shared_b[%c63, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[63][1] %f\\n", %b63.1 : f16 }
        //
        //%b63.127 = memref.load %shared_b[%c63, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[63][127] %f\\n", %b63.127 : f16 }
        //
        //amdgpu.lds_barrier

                // Thread-level indices for MFMA loading
                %load_col = affine.apply #map_load_col()[%thread_id]  // 0, 4, 8, 12 (first 16 elements of K)
                %load_row = affine.apply #map_load_row()[%thread_id]
                %load_row_1 = arith.addi %load_row, %c16 : index
                %load_row_2 = arith.addi %load_row, %c32 : index
                %load_row_3 = arith.addi %load_row, %c48 : index

                // Compute column indices for first and second half of K (split 32 into 16+16)
                %load_col_k = arith.addi %load_col, %c16 : index  // 16, 20, 24, 28 (second 16 elements of K)

                // =========================================================================
                // PROLOGUE: Load first iteration (K=0)
                // =========================================================================
                %k_start_0 = arith.constant 0 : index

                // Each thread loads its 32-element slice from A (from k_start to k_start+32)
                %a_row_vec_0 = vector.transfer_read %a_ptr[%a_row, %k_start_0], %f0_f16, %token_mask :
                    memref<16384x6144xf16>, vector<32xf16>
                // Store to shared memory
                vector.store %a_row_vec_0, %shared_a[%thread_id, %c0] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<32xf16>


                // Each thread loads its 32-element slice from B (from k_start to k_start+32)
                // B is [8, 32768, 6144], we need [expert_id, b_row, k_start]
                // Note: b_row is always < 32768 since pid_n * 64 + thread_id_x < 32768
                %b_row_vec_0 = vector.transfer_read %b_ptr[%expert_id, %b_row, %k_start_0], %f0_f16 :
                    memref<8x32768x6144xf16>, vector<32xf16>
                // Store to shared memory
                vector.store %b_row_vec_0, %shared_b[%thread_id, %c0] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<32xf16>

                amdgpu.lds_barrier

                %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
                %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

                // =========================================================================
                // MAIN LOOP: Process iterations 0 to N-2
                // =========================================================================
                %result:16 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
                    iter_args(%a00=%cst_mfma, %a01=%cst_mfma, %a02=%cst_mfma, %a03=%cst_mfma,
                                %a10=%cst_mfma, %a11=%cst_mfma, %a12=%cst_mfma, %a13=%cst_mfma,
                                %a20=%cst_mfma, %a21=%cst_mfma, %a22=%cst_mfma, %a23=%cst_mfma,
                                %a30=%cst_mfma, %a31=%cst_mfma, %a32=%cst_mfma, %a33=%cst_mfma)
                    -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {

                    // Compute K offset for this iteration
                    %k_start = arith.muli %k_block, %BLOCK_SIZE_K : index
                    %k_col = arith.addi %k_start, %load_col : index
                    %k_col_k = arith.addi %k_start, %load_col_k : index

        //gpu.printf "pid_m %d pid_n %d thread %d k_start %d load_col %d\\n", %pid_m, %pid_n, %thread_id, %k_start, %load_col : index, index, index, index, index

                    // =========================================================================
                    // FIRST HALF: K[0:16] - Load from shared memory
                    // =========================================================================

                    // Load A vectors for first half: 4 M tiles
                    %a0 = vector.load %shared_a[%load_row, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a1 = vector.load %shared_a[%load_row_1, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a2 = vector.load %shared_a[%load_row_2, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a3 = vector.load %shared_a[%load_row_3, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // Load B vectors for first half: 4 N tiles
                    // Note: B is stored as [64, 32] where rows are output features
                    // For MFMA, we need B[n, k], which maps to shared_b[load_row, load_col]
                    %b0 = vector.load %shared_b[%load_row, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b1 = vector.load %shared_b[%load_row_1, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b2 = vector.load %shared_b[%load_row_2, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b3 = vector.load %shared_b[%load_row_3, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // =========================================================================
                    // PREFETCH NEXT ITERATION from global memory
                    // =========================================================================
                    %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index

                    %a_row_vec_next = vector.transfer_read %a_ptr[%a_row, %k_start_next], %f0_f16, %token_mask :
                    memref<16384x6144xf16>, vector<32xf16>
                    %b_row_vec_next = vector.transfer_read %b_ptr[%expert_id, %b_row, %k_start_next], %f0_f16 :
                    memref<8x32768x6144xf16>, vector<32xf16>

                    // =========================================================================
                    // SECOND HALF: K[16:32] - Load from shared memory
                    // =========================================================================

                    // Load A vectors for second half: 4 M tiles
                    %a0k = vector.load %shared_a[%load_row, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a1k = vector.load %shared_a[%load_row_1, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a2k = vector.load %shared_a[%load_row_2, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a3k = vector.load %shared_a[%load_row_3, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // Load B vectors for second half: 4 N tiles
                    %b0k = vector.load %shared_b[%load_row, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b1k = vector.load %shared_b[%load_row_1, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b2k = vector.load %shared_b[%load_row_2, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b3k = vector.load %shared_b[%load_row_3, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // =========================================================================
                    // MFMA OPERATIONS - FIRST HALF (K[0:16])
                    // =========================================================================

                    // Tile (0,0)
                    %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,1)
                    %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,2)
                    %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,3)
                    %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,0)
                    %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,1)
                    %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,2)
                    %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,3)
                    %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,0)
                    %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,1)
                    %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,2)
                    %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,3)
                    %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,0)
                    %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,1)
                    %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,2)
                    %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,3)
                    %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // =========================================================================
                    // STORE PREFETCHED DATA to shared memory (after first half compute)
                    // =========================================================================
                    amdgpu.lds_barrier

                    vector.store %a_row_vec_next, %shared_a[%thread_id, %c0] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<32xf16>
                    vector.store %b_row_vec_next, %shared_b[%thread_id, %c0] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<32xf16>

                    amdgpu.lds_barrier

                    // =========================================================================
                    // MFMA OPERATIONS - SECOND HALF (K[16:32]) accumulate on 1st half results
                    // =========================================================================
                    // Tile (0,0)
                    %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,1)
                    %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,2)
                    %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,3)
                    %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,0)
                    %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,1)
                    %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,2)
                    %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,3)
                    %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,0)
                    %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,1)
                    %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,2)
                    %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,3)
                    %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,0)
                    %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,1)
                    %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,2)
                    %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,3)
                    %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                            %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
                }

                // =========================================================================
                // EPILOGUE: Process last iteration (K = num_blocks - 1)
                // =========================================================================
                // Load first half from shared memory
                %a0_last = vector.load %shared_a[%load_row, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a1_last = vector.load %shared_a[%load_row_1, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a2_last = vector.load %shared_a[%load_row_2, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a3_last = vector.load %shared_a[%load_row_3, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                %b0_last = vector.load %shared_b[%load_row, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b1_last = vector.load %shared_b[%load_row_1, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b2_last = vector.load %shared_b[%load_row_2, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b3_last = vector.load %shared_b[%load_row_3, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                // Load second half from shared memory
                %a0_k_last = vector.load %shared_a[%load_row, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a1_k_last = vector.load %shared_a[%load_row_1, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a2_k_last = vector.load %shared_a[%load_row_2, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a3_k_last = vector.load %shared_a[%load_row_3, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                %b0_k_last = vector.load %shared_b[%load_row, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b1_k_last = vector.load %shared_b[%load_row_1, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b2_k_last = vector.load %shared_b[%load_row_2, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b3_k_last = vector.load %shared_b[%load_row_3, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                // Compute first half
                %r00_0_last = amdgpu.mfma %a0_last * %b0_last + %result#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r01_0_last = amdgpu.mfma %a0_last * %b1_last + %result#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r02_0_last = amdgpu.mfma %a0_last * %b2_last + %result#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r03_0_last = amdgpu.mfma %a0_last * %b3_last + %result#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r10_0_last = amdgpu.mfma %a1_last * %b0_last + %result#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r11_0_last = amdgpu.mfma %a1_last * %b1_last + %result#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r12_0_last = amdgpu.mfma %a1_last * %b2_last + %result#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r13_0_last = amdgpu.mfma %a1_last * %b3_last + %result#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r20_0_last = amdgpu.mfma %a2_last * %b0_last + %result#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r21_0_last = amdgpu.mfma %a2_last * %b1_last + %result#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r22_0_last = amdgpu.mfma %a2_last * %b2_last + %result#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r23_0_last = amdgpu.mfma %a2_last * %b3_last + %result#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r30_0_last = amdgpu.mfma %a3_last * %b0_last + %result#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r31_0_last = amdgpu.mfma %a3_last * %b1_last + %result#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r32_0_last = amdgpu.mfma %a3_last * %b2_last + %result#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r33_0_last = amdgpu.mfma %a3_last * %b3_last + %result#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                // Compute second half (final results)
                %r00_final = amdgpu.mfma %a0_k_last * %b0_k_last + %r00_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r01_final = amdgpu.mfma %a0_k_last * %b1_k_last + %r01_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r02_final = amdgpu.mfma %a0_k_last * %b2_k_last + %r02_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r03_final = amdgpu.mfma %a0_k_last * %b3_k_last + %r03_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r10_final = amdgpu.mfma %a1_k_last * %b0_k_last + %r10_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r11_final = amdgpu.mfma %a1_k_last * %b1_k_last + %r11_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r12_final = amdgpu.mfma %a1_k_last * %b2_k_last + %r12_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r13_final = amdgpu.mfma %a1_k_last * %b3_k_last + %r13_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r20_final = amdgpu.mfma %a2_k_last * %b0_k_last + %r20_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r21_final = amdgpu.mfma %a2_k_last * %b1_k_last + %r21_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r22_final = amdgpu.mfma %a2_k_last * %b2_k_last + %r22_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r23_final = amdgpu.mfma %a2_k_last * %b3_k_last + %r23_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r30_final = amdgpu.mfma %a3_k_last * %b0_k_last + %r30_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r31_final = amdgpu.mfma %a3_k_last * %b1_k_last + %r31_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r32_final = amdgpu.mfma %a3_k_last * %b2_k_last + %r32_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r33_final = amdgpu.mfma %a3_k_last * %b3_k_last + %r33_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                // =========================================================================
                // STORE RESULTS
                // =========================================================================

                // Truncate to f16
                %r00_f16 = arith.truncf %r00_final : vector<4xf32> to vector<4xf16>
                %r01_f16 = arith.truncf %r01_final : vector<4xf32> to vector<4xf16>
                %r02_f16 = arith.truncf %r02_final : vector<4xf32> to vector<4xf16>
                %r03_f16 = arith.truncf %r03_final : vector<4xf32> to vector<4xf16>
                %r10_f16 = arith.truncf %r10_final : vector<4xf32> to vector<4xf16>
                %r11_f16 = arith.truncf %r11_final : vector<4xf32> to vector<4xf16>
                %r12_f16 = arith.truncf %r12_final : vector<4xf32> to vector<4xf16>
                %r13_f16 = arith.truncf %r13_final : vector<4xf32> to vector<4xf16>
                %r20_f16 = arith.truncf %r20_final : vector<4xf32> to vector<4xf16>
                %r21_f16 = arith.truncf %r21_final : vector<4xf32> to vector<4xf16>
                %r22_f16 = arith.truncf %r22_final : vector<4xf32> to vector<4xf16>
                %r23_f16 = arith.truncf %r23_final : vector<4xf32> to vector<4xf16>
                %r30_f16 = arith.truncf %r30_final : vector<4xf32> to vector<4xf16>
                %r31_f16 = arith.truncf %r31_final : vector<4xf32> to vector<4xf16>
                %r32_f16 = arith.truncf %r32_final : vector<4xf32> to vector<4xf16>
                %r33_f16 = arith.truncf %r33_final : vector<4xf32> to vector<4xf16>

                %store_col_0 = affine.apply #map_store_col()[%thread_id]
                %store_col_1 = arith.addi %store_col_0, %c16 : index
                %store_col_2 = arith.addi %store_col_0, %c32 : index
                %store_col_3 = arith.addi %store_col_0, %c48 : index
                %store_row_0_0 = affine.apply #map_store_row()[%thread_id]
                %store_row_0_1 = arith.addi %store_row_0_0, %c1 : index
                %store_row_0_2 = arith.addi %store_row_0_0, %c2 : index
                %store_row_0_3 = arith.addi %store_row_0_0, %c3 : index
                %store_row_16_0 = arith.addi %store_row_0_0, %c16 : index
                %store_row_16_1 = arith.addi %store_row_16_0, %c1 : index
                %store_row_16_2 = arith.addi %store_row_16_0, %c2 : index
                %store_row_16_3 = arith.addi %store_row_16_0, %c3 : index
                %store_row_32_0 = arith.addi %store_row_0_0, %c32 : index
                %store_row_32_1 = arith.addi %store_row_32_0, %c1 : index
                %store_row_32_2 = arith.addi %store_row_32_0, %c2 : index
                %store_row_32_3 = arith.addi %store_row_32_0, %c3 : index
                %store_row_48_0 = arith.addi %store_row_0_0, %c48 : index
                %store_row_48_1 = arith.addi %store_row_48_0, %c1 : index
                %store_row_48_2 = arith.addi %store_row_48_0, %c2 : index
                %store_row_48_3 = arith.addi %store_row_48_0, %c3 : index

                %r00_0_f16 = vector.extract_strided_slice %r00_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_1_f16 = vector.extract_strided_slice %r00_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_2_f16 = vector.extract_strided_slice %r00_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_3_f16 = vector.extract_strided_slice %r00_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_0_f16 = vector.extract_strided_slice %r01_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_1_f16 = vector.extract_strided_slice %r01_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_2_f16 = vector.extract_strided_slice %r01_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_3_f16 = vector.extract_strided_slice %r01_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_0_f16 = vector.extract_strided_slice %r02_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_1_f16 = vector.extract_strided_slice %r02_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_2_f16 = vector.extract_strided_slice %r02_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_3_f16 = vector.extract_strided_slice %r02_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_0_f16 = vector.extract_strided_slice %r03_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_1_f16 = vector.extract_strided_slice %r03_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_2_f16 = vector.extract_strided_slice %r03_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_3_f16 = vector.extract_strided_slice %r03_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r10_0_f16 = vector.extract_strided_slice %r10_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_1_f16 = vector.extract_strided_slice %r10_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_2_f16 = vector.extract_strided_slice %r10_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_3_f16 = vector.extract_strided_slice %r10_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_0_f16 = vector.extract_strided_slice %r11_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_1_f16 = vector.extract_strided_slice %r11_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_2_f16 = vector.extract_strided_slice %r11_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_3_f16 = vector.extract_strided_slice %r11_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_0_f16 = vector.extract_strided_slice %r12_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_1_f16 = vector.extract_strided_slice %r12_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_2_f16 = vector.extract_strided_slice %r12_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_3_f16 = vector.extract_strided_slice %r12_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_0_f16 = vector.extract_strided_slice %r13_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_1_f16 = vector.extract_strided_slice %r13_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_2_f16 = vector.extract_strided_slice %r13_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_3_f16 = vector.extract_strided_slice %r13_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r20_0_f16 = vector.extract_strided_slice %r20_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_1_f16 = vector.extract_strided_slice %r20_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_2_f16 = vector.extract_strided_slice %r20_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_3_f16 = vector.extract_strided_slice %r20_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_0_f16 = vector.extract_strided_slice %r21_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_1_f16 = vector.extract_strided_slice %r21_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_2_f16 = vector.extract_strided_slice %r21_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_3_f16 = vector.extract_strided_slice %r21_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_0_f16 = vector.extract_strided_slice %r22_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_1_f16 = vector.extract_strided_slice %r22_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_2_f16 = vector.extract_strided_slice %r22_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_3_f16 = vector.extract_strided_slice %r22_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_0_f16 = vector.extract_strided_slice %r23_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_1_f16 = vector.extract_strided_slice %r23_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_2_f16 = vector.extract_strided_slice %r23_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_3_f16 = vector.extract_strided_slice %r23_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r30_0_f16 = vector.extract_strided_slice %r30_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_1_f16 = vector.extract_strided_slice %r30_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_2_f16 = vector.extract_strided_slice %r30_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_3_f16 = vector.extract_strided_slice %r30_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_0_f16 = vector.extract_strided_slice %r31_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_1_f16 = vector.extract_strided_slice %r31_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_2_f16 = vector.extract_strided_slice %r31_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_3_f16 = vector.extract_strided_slice %r31_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_0_f16 = vector.extract_strided_slice %r32_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_1_f16 = vector.extract_strided_slice %r32_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_2_f16 = vector.extract_strided_slice %r32_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_3_f16 = vector.extract_strided_slice %r32_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_0_f16 = vector.extract_strided_slice %r33_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_1_f16 = vector.extract_strided_slice %r33_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_2_f16 = vector.extract_strided_slice %r33_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_3_f16 = vector.extract_strided_slice %r33_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                // Flatten c_ptr for easier indexing
                %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>

                // Each thread writes to 4 different rows (from load_row, load_row+16, load_row+32, load_row+48)
                // across 4 column groups (base, base+16, base+32, base+48)

                // Get token indices for output rows
                %out_token_0_0 = arith.addi %offs_token_id_base, %store_row_0_0 : index
                %out_token_0_1 = arith.addi %offs_token_id_base, %store_row_0_1 : index
                %out_token_0_2 = arith.addi %offs_token_id_base, %store_row_0_2 : index
                %out_token_0_3 = arith.addi %offs_token_id_base, %store_row_0_3 : index
                %out_token_16_0 = arith.addi %offs_token_id_base, %store_row_16_0 : index
                %out_token_16_1 = arith.addi %offs_token_id_base, %store_row_16_1 : index
                %out_token_16_2 = arith.addi %offs_token_id_base, %store_row_16_2 : index
                %out_token_16_3 = arith.addi %offs_token_id_base, %store_row_16_3 : index
                %out_token_32_0 = arith.addi %offs_token_id_base, %store_row_32_0 : index
                %out_token_32_1 = arith.addi %offs_token_id_base, %store_row_32_1 : index
                %out_token_32_2 = arith.addi %offs_token_id_base, %store_row_32_2 : index
                %out_token_32_3 = arith.addi %offs_token_id_base, %store_row_32_3 : index
                %out_token_48_0 = arith.addi %offs_token_id_base, %store_row_48_0 : index
                %out_token_48_1 = arith.addi %offs_token_id_base, %store_row_48_1 : index
                %out_token_48_2 = arith.addi %offs_token_id_base, %store_row_48_2 : index
                %out_token_48_3 = arith.addi %offs_token_id_base, %store_row_48_3 : index

                %tok_id_0_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_0] : memref<33335xi32>
                %tok_id_0_0 = arith.index_cast %tok_id_0_0_i32 : i32 to index
                %out_base_0_0 = arith.muli %tok_id_0_0, %N : index
                %out_valid_0_0 = arith.cmpi slt, %tok_id_0_0, %num_valid_tokens : index
                %out_mask_0_0 = vector.broadcast %out_valid_0_0 : i1 to vector<1xi1>
                %tok_id_0_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_1] : memref<33335xi32>
                %tok_id_0_1 = arith.index_cast %tok_id_0_1_i32 : i32 to index
                %out_base_0_1 = arith.muli %tok_id_0_1, %N : index
                %out_valid_0_1 = arith.cmpi slt, %tok_id_0_1, %num_valid_tokens : index
                %out_mask_0_1 = vector.broadcast %out_valid_0_1 : i1 to vector<1xi1>
                %tok_id_0_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_2] : memref<33335xi32>
                %tok_id_0_2 = arith.index_cast %tok_id_0_2_i32 : i32 to index
                %out_base_0_2 = arith.muli %tok_id_0_2, %N : index
                %out_valid_0_2 = arith.cmpi slt, %tok_id_0_2, %num_valid_tokens : index
                %out_mask_0_2 = vector.broadcast %out_valid_0_2 : i1 to vector<1xi1>
                %tok_id_0_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_3] : memref<33335xi32>
                %tok_id_0_3 = arith.index_cast %tok_id_0_3_i32 : i32 to index
                %out_base_0_3 = arith.muli %tok_id_0_3, %N : index
                %out_valid_0_3 = arith.cmpi slt, %tok_id_0_3, %num_valid_tokens : index
                %out_mask_0_3 = vector.broadcast %out_valid_0_3 : i1 to vector<1xi1>

                %tok_id_16_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_0] : memref<33335xi32>
                %tok_id_16_0 = arith.index_cast %tok_id_16_0_i32 : i32 to index
                %out_base_16_0 = arith.muli %tok_id_16_0, %N : index
                %out_valid_16_0 = arith.cmpi slt, %tok_id_16_0, %num_valid_tokens : index
                %out_mask_16_0 = vector.broadcast %out_valid_16_0 : i1 to vector<1xi1>
                %tok_id_16_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_1] : memref<33335xi32>
                %tok_id_16_1 = arith.index_cast %tok_id_16_1_i32 : i32 to index
                %out_base_16_1 = arith.muli %tok_id_16_1, %N : index
                %out_valid_16_1 = arith.cmpi slt, %tok_id_16_1, %num_valid_tokens : index
                %out_mask_16_1 = vector.broadcast %out_valid_16_1 : i1 to vector<1xi1>
                %tok_id_16_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_2] : memref<33335xi32>
                %tok_id_16_2 = arith.index_cast %tok_id_16_2_i32 : i32 to index
                %out_base_16_2 = arith.muli %tok_id_16_2, %N : index
                %out_valid_16_2 = arith.cmpi slt, %tok_id_16_2, %num_valid_tokens : index
                %out_mask_16_2 = vector.broadcast %out_valid_16_2 : i1 to vector<1xi1>
                %tok_id_16_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_3] : memref<33335xi32>
                %tok_id_16_3 = arith.index_cast %tok_id_16_3_i32 : i32 to index
                %out_base_16_3 = arith.muli %tok_id_16_3, %N : index
                %out_valid_16_3 = arith.cmpi slt, %tok_id_16_3, %num_valid_tokens : index
                %out_mask_16_3 = vector.broadcast %out_valid_16_3 : i1 to vector<1xi1>

                %tok_id_32_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_0] : memref<33335xi32>
                %tok_id_32_0 = arith.index_cast %tok_id_32_0_i32 : i32 to index
                %out_base_32_0 = arith.muli %tok_id_32_0, %N : index
                %out_valid_32_0 = arith.cmpi slt, %tok_id_32_0, %num_valid_tokens : index
                %out_mask_32_0 = vector.broadcast %out_valid_32_0 : i1 to vector<1xi1>
                %tok_id_32_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_1] : memref<33335xi32>
                %tok_id_32_1 = arith.index_cast %tok_id_32_1_i32 : i32 to index
                %out_base_32_1 = arith.muli %tok_id_32_1, %N : index
                %out_valid_32_1 = arith.cmpi slt, %tok_id_32_1, %num_valid_tokens : index
                %out_mask_32_1 = vector.broadcast %out_valid_32_1 : i1 to vector<1xi1>
                %tok_id_32_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_2] : memref<33335xi32>
                %tok_id_32_2 = arith.index_cast %tok_id_32_2_i32 : i32 to index
                %out_base_32_2 = arith.muli %tok_id_32_2, %N : index
                %out_valid_32_2 = arith.cmpi slt, %tok_id_32_2, %num_valid_tokens : index
                %out_mask_32_2 = vector.broadcast %out_valid_32_2 : i1 to vector<1xi1>
                %tok_id_32_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_3] : memref<33335xi32>
                %tok_id_32_3 = arith.index_cast %tok_id_32_3_i32 : i32 to index
                %out_base_32_3 = arith.muli %tok_id_32_3, %N : index
                %out_valid_32_3 = arith.cmpi slt, %tok_id_32_3, %num_valid_tokens : index
                %out_mask_32_3 = vector.broadcast %out_valid_32_3 : i1 to vector<1xi1>

                %tok_id_48_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_0] : memref<33335xi32>
                %tok_id_48_0 = arith.index_cast %tok_id_48_0_i32 : i32 to index
                %out_base_48_0 = arith.muli %tok_id_48_0, %N : index
                %out_valid_48_0 = arith.cmpi slt, %tok_id_48_0, %num_valid_tokens : index
                %out_mask_48_0 = vector.broadcast %out_valid_48_0 : i1 to vector<1xi1>
                %tok_id_48_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_1] : memref<33335xi32>
                %tok_id_48_1 = arith.index_cast %tok_id_48_1_i32 : i32 to index
                %out_base_48_1 = arith.muli %tok_id_48_1, %N : index
                %out_valid_48_1 = arith.cmpi slt, %tok_id_48_1, %num_valid_tokens : index
                %out_mask_48_1 = vector.broadcast %out_valid_48_1 : i1 to vector<1xi1>
                %tok_id_48_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_2] : memref<33335xi32>
                %tok_id_48_2 = arith.index_cast %tok_id_48_2_i32 : i32 to index
                %out_base_48_2 = arith.muli %tok_id_48_2, %N : index
                %out_valid_48_2 = arith.cmpi slt, %tok_id_48_2, %num_valid_tokens : index
                %out_mask_48_2 = vector.broadcast %out_valid_48_2 : i1 to vector<1xi1>
                %tok_id_48_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_3] : memref<33335xi32>
                %tok_id_48_3 = arith.index_cast %tok_id_48_3_i32 : i32 to index
                %out_base_48_3 = arith.muli %tok_id_48_3, %N : index
                %out_valid_48_3 = arith.cmpi slt, %tok_id_48_3, %num_valid_tokens : index
                %out_mask_48_3 = vector.broadcast %out_valid_48_3 : i1 to vector<1xi1>

                // pid_n determines which 64-neuron block we're computing
                %out_col_base = arith.muli %pid_n, %BLOCK_SIZE_N : index

                // Column offsets for the 4 column tiles
                %out_col_0 = arith.addi %out_col_base, %store_col_0 : index
                %out_col_1 = arith.addi %out_col_base, %store_col_1 : index
                %out_col_2 = arith.addi %out_col_base, %store_col_2 : index
                %out_col_3 = arith.addi %out_col_base, %store_col_3 : index

                // Write all 16 tiles using vector.store
                // Tile (0,0)
                %idx_00_0 = arith.addi %out_base_0_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_0], %out_mask_0_0, %r00_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_00_1 = arith.addi %out_base_0_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_1], %out_mask_0_1, %r00_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_00_2 = arith.addi %out_base_0_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_2], %out_mask_0_2, %r00_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_00_3 = arith.addi %out_base_0_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_3], %out_mask_0_3, %r00_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (0,1)
                %idx_01_0 = arith.addi %out_base_0_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_0], %out_mask_0_0, %r01_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_01_1 = arith.addi %out_base_0_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_1], %out_mask_0_1, %r01_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_01_2 = arith.addi %out_base_0_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_2], %out_mask_0_2, %r01_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_01_3 = arith.addi %out_base_0_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_3], %out_mask_0_3, %r01_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (0,2)
                %idx_02_0 = arith.addi %out_base_0_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_0], %out_mask_0_0, %r02_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_02_1 = arith.addi %out_base_0_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_1], %out_mask_0_1, %r02_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_02_2 = arith.addi %out_base_0_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_2], %out_mask_0_2, %r02_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_02_3 = arith.addi %out_base_0_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_3], %out_mask_0_3, %r02_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (0,3)
                %idx_03_0 = arith.addi %out_base_0_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_0], %out_mask_0_0, %r03_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_03_1 = arith.addi %out_base_0_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_1], %out_mask_0_1, %r03_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_03_2 = arith.addi %out_base_0_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_2], %out_mask_0_2, %r03_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_03_3 = arith.addi %out_base_0_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_3], %out_mask_0_3, %r03_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,0)
                %idx_10_0 = arith.addi %out_base_16_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_0], %out_mask_16_0, %r10_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_10_1 = arith.addi %out_base_16_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_1], %out_mask_16_1, %r10_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_10_2 = arith.addi %out_base_16_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_2], %out_mask_16_2, %r10_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_10_3 = arith.addi %out_base_16_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_3], %out_mask_16_3, %r10_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,1)
                %idx_11_0 = arith.addi %out_base_16_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_0], %out_mask_16_0, %r11_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_11_1 = arith.addi %out_base_16_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_1], %out_mask_16_1, %r11_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_11_2 = arith.addi %out_base_16_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_2], %out_mask_16_2, %r11_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_11_3 = arith.addi %out_base_16_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_3], %out_mask_16_3, %r11_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,2)
                %idx_12_0 = arith.addi %out_base_16_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_0], %out_mask_16_0, %r12_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_12_1 = arith.addi %out_base_16_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_1], %out_mask_16_1, %r12_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_12_2 = arith.addi %out_base_16_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_2], %out_mask_16_2, %r12_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_12_3 = arith.addi %out_base_16_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_3], %out_mask_16_3, %r12_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,3)
                %idx_13_0 = arith.addi %out_base_16_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_0], %out_mask_16_0, %r13_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_13_1 = arith.addi %out_base_16_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_1], %out_mask_16_1, %r13_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_13_2 = arith.addi %out_base_16_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_2], %out_mask_16_2, %r13_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_13_3 = arith.addi %out_base_16_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_3], %out_mask_16_3, %r13_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,0)
                %idx_20_0 = arith.addi %out_base_32_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_0], %out_mask_32_0, %r20_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_20_1 = arith.addi %out_base_32_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_1], %out_mask_32_1, %r20_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_20_2 = arith.addi %out_base_32_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_2], %out_mask_32_2, %r20_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_20_3 = arith.addi %out_base_32_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_3], %out_mask_32_3, %r20_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,1)
                %idx_21_0 = arith.addi %out_base_32_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_0], %out_mask_32_0, %r21_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_21_1 = arith.addi %out_base_32_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_1], %out_mask_32_1, %r21_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_21_2 = arith.addi %out_base_32_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_2], %out_mask_32_2, %r21_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_21_3 = arith.addi %out_base_32_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_3], %out_mask_32_3, %r21_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,2)
                %idx_22_0 = arith.addi %out_base_32_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_0], %out_mask_32_0, %r22_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_22_1 = arith.addi %out_base_32_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_1], %out_mask_32_1, %r22_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_22_2 = arith.addi %out_base_32_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_2], %out_mask_32_2, %r22_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_22_3 = arith.addi %out_base_32_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_3], %out_mask_32_3, %r22_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,3)
                %idx_23_0 = arith.addi %out_base_32_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_0], %out_mask_32_0, %r23_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_23_1 = arith.addi %out_base_32_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_1], %out_mask_32_1, %r23_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_23_2 = arith.addi %out_base_32_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_2], %out_mask_32_2, %r23_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_23_3 = arith.addi %out_base_32_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_3], %out_mask_32_3, %r23_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,0)
                %idx_30_0 = arith.addi %out_base_48_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_0], %out_mask_48_0, %r30_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_30_1 = arith.addi %out_base_48_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_1], %out_mask_48_1, %r30_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_30_2 = arith.addi %out_base_48_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_2], %out_mask_48_2, %r30_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_30_3 = arith.addi %out_base_48_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_3], %out_mask_48_3, %r30_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,1)
                %idx_31_0 = arith.addi %out_base_48_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_0], %out_mask_48_0, %r31_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_31_1 = arith.addi %out_base_48_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_1], %out_mask_48_1, %r31_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_31_2 = arith.addi %out_base_48_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_2], %out_mask_48_2, %r31_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_31_3 = arith.addi %out_base_48_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_3], %out_mask_48_3, %r31_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,2)
                %idx_32_0 = arith.addi %out_base_48_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_0], %out_mask_48_0, %r32_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_32_1 = arith.addi %out_base_48_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_1], %out_mask_48_1, %r32_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_32_2 = arith.addi %out_base_48_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_2], %out_mask_48_2, %r32_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_32_3 = arith.addi %out_base_48_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_3], %out_mask_48_3, %r32_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,3)
                %idx_33_0 = arith.addi %out_base_48_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_0], %out_mask_48_0, %r33_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_33_1 = arith.addi %out_base_48_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_1], %out_mask_48_1, %r33_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_33_2 = arith.addi %out_base_48_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_2], %out_mask_48_2, %r33_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                %idx_33_3 = arith.addi %out_base_48_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_3], %out_mask_48_3, %r33_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
                }
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
            // %a_ptr: memref<16384x6144xf16>,
            // %b_ptr: memref<8x32768x6144xf16>,
            // %sorted_token_ids_ptr: memref<33335xi32>,
            // %expert_ids_ptr: memref<521xi32>,
            // %num_tokens_post_padded_ptr: memref<1xi32>,
            // %c_ptr: memref<16384x2x32768xf16>
            %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
            %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
            %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
            %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
            %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
            %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
            %6 = flow.dispatch @fused_moe_kernel::@fused_moe_kernel(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
            %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
            %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
            return %8 : !hal.buffer_view
        }
        }
            """
        )
        
    asm_dtype0_1024_256_8_64_2_2048_mfma = (
            """
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

        #map_load_row = affine_map<()[s0] -> (s0 mod 16)>
        #map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

        #map_store_col = affine_map<()[s0] -> (s0 mod 16)>
        #map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

        module attributes {transform.with_named_sequence} {
        stream.executable private @fused_moe_kernel {
            stream.executable.export public @fused_moe_kernel workgroups() -> (index, index, index) {
            %c1168 = arith.constant 1168 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            stream.return %c1168, %c1, %c1 : index, index, index
            }
            builtin.module {
            func.func @fused_moe_kernel(
                // Input memrefs
            // %a_ptr: memref<2048x256xf16>,
            // %b_ptr: memref<8x1024x256xf16>,
            // %sorted_token_ids_ptr: memref<4663xi32>,
            // %expert_ids_ptr: memref<10xi32>,
            // %num_tokens_post_padded_ptr: memref<1xi32>,
            // %c_ptr: memref<2048x2x1024xf16>
                %arg0: !stream.binding,
                %arg1: !stream.binding,
                %arg2: !stream.binding,
                %arg3: !stream.binding,
                %arg4: !stream.binding,
                %arg5: !stream.binding
            ) attributes {translation_info = #translation} {
                // N = 1024
                // K = 256
                // EM = 4663
                // top_k = 2
                // num_valid_tokens = 4096
                // GROUP_SIZE_M = 8
                // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
                // BLOCK_SIZE_K = 32
                %N = arith.constant 1024 : index
                %K = arith.constant 256 : index
                %EM = arith.constant 4663 : index
                %top_k = arith.constant 2 : index
                %num_valid_tokens = arith.constant 4096 : index
                %GROUP_SIZE_M = arith.constant 8 : index
                %BLOCK_SIZE_M = arith.constant 64 : index
                %BLOCK_SIZE_N = arith.constant 64 : index
                %BLOCK_SIZE_K = arith.constant 32 : index

                %c32768 = arith.constant 32768 : index
                %c0 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                %c2 = arith.constant 2 : index
                %c3 = arith.constant 3 : index
                %c16 = arith.constant 16 : index
                %c32 = arith.constant 32 : index
                %c48 = arith.constant 48 : index
                %c63 = arith.constant 63 : index
                %c127 = arith.constant 127 : index
                %c256 = arith.constant 256 : index
                %c511 = arith.constant 511 : index
                %f0 = arith.constant 0.0 : f32
                %f0_f16 = arith.constant 0.0 : f16
                %cst_mfma = arith.constant dense<0.000000e+00> : vector<4xf32>

                %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<2048x256xf16>
                %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x1024x256xf16>
                %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<2048x2x1024xf16>
                %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<4663xi32>
                %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<73xi32>
                %num_tokens_post_padded_ptr = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1xi32>

                // Program ID mapping
                %pid = gpu.block_id x
                %num_pid_m = arith.ceildivui %EM, %BLOCK_SIZE_M : index
                %num_pid_n = arith.ceildivui %N, %BLOCK_SIZE_N : index
                %num_pid_in_group = arith.muli %GROUP_SIZE_M, %num_pid_n : index
                %group_id = arith.divui %pid, %num_pid_in_group : index
                %first_pid_m = arith.muli %group_id, %GROUP_SIZE_M : index
                %min_group_size_m = arith.subi %num_pid_m, %first_pid_m : index
                %group_size_m = arith.minui %GROUP_SIZE_M, %min_group_size_m : index
                %0 = arith.remsi %pid, %num_pid_in_group : index
                %1 = arith.remsi %0, %group_size_m : index
                %pid_m = arith.addi %first_pid_m, %1 : index
                %pid_n = arith.divui %0, %group_size_m : index

                %thread_id = gpu.thread_id x upper_bound 64

                // Early exit check
                %2 = memref.load %num_tokens_post_padded_ptr[%c0] : memref<1xi32>
                %num_tokens_post_padded = arith.index_cast %2 : i32 to index
                %pid_m_offset = arith.muli %pid_m, %BLOCK_SIZE_M : index
                %should_exit = arith.cmpi sge, %pid_m_offset, %num_tokens_post_padded : index
                scf.if %should_exit {
                scf.yield
                } else {
                // Compute token mask
                %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index
                %thread_token_id = arith.addi %offs_token_id_base, %thread_id : index

                // Load token ID for this row
                %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<4663xi32>
                %token_id = arith.index_cast %token_id_val : i32 to index

                %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
                %token_mask = vector.broadcast %token_valid : i1 to vector<256xi1>

                // Compute A row index: token_id // top_k
                %a_row = arith.divui %token_id, %top_k : index

                // Load expert ID
                %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<73xi32>
                %expert_id = arith.index_cast %expert_id_val : i32 to index

                // Compute B row offset for this thread
                %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
                %b_row = arith.addi %offs_bn_base, %thread_id : index

                // Allocate shared memory: 64×256 for A, 64×256 for B
                %alloc = memref.alloc() : memref<65536xi8, #gpu.address_space<workgroup>>
                %shared_a = memref.view %alloc[%c0][] : memref<65536xi8, #gpu.address_space<workgroup>>
                    to memref<64x256xf16, #gpu.address_space<workgroup>>
                %shared_b = memref.view %alloc[%c32768][] : memref<65536xi8, #gpu.address_space<workgroup>>
                    to memref<64x256xf16, #gpu.address_space<workgroup>>
        //%alloc_c = memref.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
        //%shared_c = memref.view %alloc_c[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>>
        //  to memref<64x64xf16, #gpu.address_space<workgroup>>

                // Each thread loads its full row from A (256 f16)
                %a_row_vec = vector.transfer_read %a_ptr[%a_row, %c0], %f0_f16, %token_mask :
                    memref<2048x256xf16>, vector<256xf16>
                // Store to shared memory
                vector.store %a_row_vec, %shared_a[%thread_id, %c0] :
                    memref<64x256xf16, #gpu.address_space<workgroup>>, vector<256xf16>

                // Each thread loads its row from B (256 f16)
                // B is [8, 1024, 256], we need [expert_id, b_row, :]
                // Note: b_row is always < 1024 since pid_n * 64 + thread_id_x < 1024
                %b_row_vec = vector.transfer_read %b_ptr[%expert_id, %b_row, %c0], %f0_f16 :
                    memref<8x1024x256xf16>, vector<256xf16>
                // Store to shared memory
                vector.store %b_row_vec, %shared_b[%thread_id, %c0] :
                    memref<64x256xf16, #gpu.address_space<workgroup>>, vector<256xf16>

                amdgpu.lds_barrier


                // Thread-level indices for MFMA loading
                %load_col = affine.apply #map_load_col()[%thread_id]
                %load_row = affine.apply #map_load_row()[%thread_id]
                %load_row_1 = arith.addi %load_row, %c16 : index
                %load_row_2 = arith.addi %load_row, %c32 : index
                %load_row_3 = arith.addi %load_row, %c48 : index
        //gpu.printf "T%d load_col %d load_row %d\\n", %thread_id, %load_col, %load_row : index, index, index

                // =========================================================================
                // MFMA COMPUTATION
                // =========================================================================
                %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index

                %result:16 = scf.for %k_block = %c0 to %num_blocks step %c1
                    iter_args(%a00=%cst_mfma, %a01=%cst_mfma, %a02=%cst_mfma, %a03=%cst_mfma,
                                %a10=%cst_mfma, %a11=%cst_mfma, %a12=%cst_mfma, %a13=%cst_mfma,
                                %a20=%cst_mfma, %a21=%cst_mfma, %a22=%cst_mfma, %a23=%cst_mfma,
                                %a30=%cst_mfma, %a31=%cst_mfma, %a32=%cst_mfma, %a33=%cst_mfma)
                    -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {

                    // Compute K offset for this iteration
                    %k_start = arith.muli %k_block, %BLOCK_SIZE_K : index
                    %k_col = arith.addi %k_start, %load_col : index
                    %k_col_k = arith.addi %k_col, %c16 : index

                    // Load A vectors: 4 M tiles × 2 K slices (columns k_col and k_col+16)
                    %a0 = vector.load %shared_a[%load_row, %k_col] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a1 = vector.load %shared_a[%load_row_1, %k_col] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a2 = vector.load %shared_a[%load_row_2, %k_col] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a3 = vector.load %shared_a[%load_row_3, %k_col] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    %a0k = vector.load %shared_a[%load_row, %k_col_k] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a1k = vector.load %shared_a[%load_row_1, %k_col_k] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a2k = vector.load %shared_a[%load_row_2, %k_col_k] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a3k = vector.load %shared_a[%load_row_3, %k_col_k] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // Load B vectors: 4 N tiles × 2 K slices
                    // Note: B is stored as [64, 256] where rows are output features
                    // For MFMA, we need B[n, k], which maps to shared_b[row, k_col]
                    %b0 = vector.load %shared_b[%load_row, %k_col] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b1 = vector.load %shared_b[%load_row_1, %k_col] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b2 = vector.load %shared_b[%load_row_2, %k_col] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b3 = vector.load %shared_b[%load_row_3, %k_col] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    %b0k = vector.load %shared_b[%load_row, %k_col_k] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b1k = vector.load %shared_b[%load_row_1, %k_col_k] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b2k = vector.load %shared_b[%load_row_2, %k_col_k] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b3k = vector.load %shared_b[%load_row_3, %k_col_k] :
                        memref<64x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // MFMA operations: 4×4 tile grid
                    // Tile (0,0)
                    %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (0,1)
                    %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (0,2)
                    %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (0,3)
                    %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,0)
                    %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,1)
                    %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,2)
                    %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,3)
                    %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,0)
                    %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,1)
                    %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,2)
                    %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,3)
                    %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,0)
                    %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,1)
                    %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,2)
                    %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,3)
                    %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                            %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
                }

                // =========================================================================
                // STORE RESULTS
                // =========================================================================

                // Truncate to f16
                %r00_f16 = arith.truncf %result#0 : vector<4xf32> to vector<4xf16>
                %r01_f16 = arith.truncf %result#1 : vector<4xf32> to vector<4xf16>
                %r02_f16 = arith.truncf %result#2 : vector<4xf32> to vector<4xf16>
                %r03_f16 = arith.truncf %result#3 : vector<4xf32> to vector<4xf16>
                %r10_f16 = arith.truncf %result#4 : vector<4xf32> to vector<4xf16>
                %r11_f16 = arith.truncf %result#5 : vector<4xf32> to vector<4xf16>
                %r12_f16 = arith.truncf %result#6 : vector<4xf32> to vector<4xf16>
                %r13_f16 = arith.truncf %result#7 : vector<4xf32> to vector<4xf16>
                %r20_f16 = arith.truncf %result#8 : vector<4xf32> to vector<4xf16>
                %r21_f16 = arith.truncf %result#9 : vector<4xf32> to vector<4xf16>
                %r22_f16 = arith.truncf %result#10 : vector<4xf32> to vector<4xf16>
                %r23_f16 = arith.truncf %result#11 : vector<4xf32> to vector<4xf16>
                %r30_f16 = arith.truncf %result#12 : vector<4xf32> to vector<4xf16>
                %r31_f16 = arith.truncf %result#13 : vector<4xf32> to vector<4xf16>
                %r32_f16 = arith.truncf %result#14 : vector<4xf32> to vector<4xf16>
                %r33_f16 = arith.truncf %result#15 : vector<4xf32> to vector<4xf16>

                %store_col_0 = affine.apply #map_store_col()[%thread_id]
                %store_col_1 = arith.addi %store_col_0, %c16 : index
                %store_col_2 = arith.addi %store_col_0, %c32 : index
                %store_col_3 = arith.addi %store_col_0, %c48 : index
                %store_row_0_0 = affine.apply #map_store_row()[%thread_id]
                %store_row_0_1 = arith.addi %store_row_0_0, %c1 : index
                %store_row_0_2 = arith.addi %store_row_0_0, %c2 : index
                %store_row_0_3 = arith.addi %store_row_0_0, %c3 : index
                %store_row_16_0 = arith.addi %store_row_0_0, %c16 : index
                %store_row_16_1 = arith.addi %store_row_16_0, %c1 : index
                %store_row_16_2 = arith.addi %store_row_16_0, %c2 : index
                %store_row_16_3 = arith.addi %store_row_16_0, %c3 : index
                %store_row_32_0 = arith.addi %store_row_0_0, %c32 : index
                %store_row_32_1 = arith.addi %store_row_32_0, %c1 : index
                %store_row_32_2 = arith.addi %store_row_32_0, %c2 : index
                %store_row_32_3 = arith.addi %store_row_32_0, %c3 : index
                %store_row_48_0 = arith.addi %store_row_0_0, %c48 : index
                %store_row_48_1 = arith.addi %store_row_48_0, %c1 : index
                %store_row_48_2 = arith.addi %store_row_48_0, %c2 : index
                %store_row_48_3 = arith.addi %store_row_48_0, %c3 : index

                %r00_0_f16 = vector.extract_strided_slice %r00_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_1_f16 = vector.extract_strided_slice %r00_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_2_f16 = vector.extract_strided_slice %r00_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_3_f16 = vector.extract_strided_slice %r00_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_0_f16 = vector.extract_strided_slice %r01_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_1_f16 = vector.extract_strided_slice %r01_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_2_f16 = vector.extract_strided_slice %r01_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_3_f16 = vector.extract_strided_slice %r01_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_0_f16 = vector.extract_strided_slice %r02_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_1_f16 = vector.extract_strided_slice %r02_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_2_f16 = vector.extract_strided_slice %r02_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_3_f16 = vector.extract_strided_slice %r02_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_0_f16 = vector.extract_strided_slice %r03_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_1_f16 = vector.extract_strided_slice %r03_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_2_f16 = vector.extract_strided_slice %r03_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_3_f16 = vector.extract_strided_slice %r03_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r10_0_f16 = vector.extract_strided_slice %r10_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_1_f16 = vector.extract_strided_slice %r10_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_2_f16 = vector.extract_strided_slice %r10_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_3_f16 = vector.extract_strided_slice %r10_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_0_f16 = vector.extract_strided_slice %r11_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_1_f16 = vector.extract_strided_slice %r11_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_2_f16 = vector.extract_strided_slice %r11_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_3_f16 = vector.extract_strided_slice %r11_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_0_f16 = vector.extract_strided_slice %r12_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_1_f16 = vector.extract_strided_slice %r12_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_2_f16 = vector.extract_strided_slice %r12_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_3_f16 = vector.extract_strided_slice %r12_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_0_f16 = vector.extract_strided_slice %r13_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_1_f16 = vector.extract_strided_slice %r13_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_2_f16 = vector.extract_strided_slice %r13_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_3_f16 = vector.extract_strided_slice %r13_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r20_0_f16 = vector.extract_strided_slice %r20_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_1_f16 = vector.extract_strided_slice %r20_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_2_f16 = vector.extract_strided_slice %r20_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_3_f16 = vector.extract_strided_slice %r20_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_0_f16 = vector.extract_strided_slice %r21_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_1_f16 = vector.extract_strided_slice %r21_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_2_f16 = vector.extract_strided_slice %r21_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_3_f16 = vector.extract_strided_slice %r21_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_0_f16 = vector.extract_strided_slice %r22_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_1_f16 = vector.extract_strided_slice %r22_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_2_f16 = vector.extract_strided_slice %r22_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_3_f16 = vector.extract_strided_slice %r22_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_0_f16 = vector.extract_strided_slice %r23_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_1_f16 = vector.extract_strided_slice %r23_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_2_f16 = vector.extract_strided_slice %r23_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_3_f16 = vector.extract_strided_slice %r23_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r30_0_f16 = vector.extract_strided_slice %r30_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_1_f16 = vector.extract_strided_slice %r30_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_2_f16 = vector.extract_strided_slice %r30_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_3_f16 = vector.extract_strided_slice %r30_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_0_f16 = vector.extract_strided_slice %r31_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_1_f16 = vector.extract_strided_slice %r31_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_2_f16 = vector.extract_strided_slice %r31_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_3_f16 = vector.extract_strided_slice %r31_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_0_f16 = vector.extract_strided_slice %r32_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_1_f16 = vector.extract_strided_slice %r32_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_2_f16 = vector.extract_strided_slice %r32_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_3_f16 = vector.extract_strided_slice %r32_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_0_f16 = vector.extract_strided_slice %r33_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_1_f16 = vector.extract_strided_slice %r33_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_2_f16 = vector.extract_strided_slice %r33_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_3_f16 = vector.extract_strided_slice %r33_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

    
        //amdgpu.lds_barrier

                // Flatten c_ptr for easier indexing
                %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<2048x2x1024xf16> into memref<4194304xf16>

                // Each thread writes to 4 different rows (from load_row, load_row+16, load_row+32, load_row+48)
                // across 4 column groups (base, base+16, base+32, base+48)

                // Get token indices for output rows
                %out_token_0_0 = arith.addi %offs_token_id_base, %store_row_0_0 : index
                %out_token_0_1 = arith.addi %offs_token_id_base, %store_row_0_1 : index
                %out_token_0_2 = arith.addi %offs_token_id_base, %store_row_0_2 : index
                %out_token_0_3 = arith.addi %offs_token_id_base, %store_row_0_3 : index
                %out_token_16_0 = arith.addi %offs_token_id_base, %store_row_16_0 : index
                %out_token_16_1 = arith.addi %offs_token_id_base, %store_row_16_1 : index
                %out_token_16_2 = arith.addi %offs_token_id_base, %store_row_16_2 : index
                %out_token_16_3 = arith.addi %offs_token_id_base, %store_row_16_3 : index
                %out_token_32_0 = arith.addi %offs_token_id_base, %store_row_32_0 : index
                %out_token_32_1 = arith.addi %offs_token_id_base, %store_row_32_1 : index
                %out_token_32_2 = arith.addi %offs_token_id_base, %store_row_32_2 : index
                %out_token_32_3 = arith.addi %offs_token_id_base, %store_row_32_3 : index
                %out_token_48_0 = arith.addi %offs_token_id_base, %store_row_48_0 : index
                %out_token_48_1 = arith.addi %offs_token_id_base, %store_row_48_1 : index
                %out_token_48_2 = arith.addi %offs_token_id_base, %store_row_48_2 : index
                %out_token_48_3 = arith.addi %offs_token_id_base, %store_row_48_3 : index

                %tok_id_0_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_0] : memref<4663xi32>
                %tok_id_0_0 = arith.index_cast %tok_id_0_0_i32 : i32 to index
                %out_base_0_0 = arith.muli %tok_id_0_0, %N : index
                %out_valid_0_0 = arith.cmpi slt, %tok_id_0_0, %num_valid_tokens : index
                %out_mask_0_0 = vector.broadcast %out_valid_0_0 : i1 to vector<1xi1>
                %tok_id_0_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_1] : memref<4663xi32>
                %tok_id_0_1 = arith.index_cast %tok_id_0_1_i32 : i32 to index
                %out_base_0_1 = arith.muli %tok_id_0_1, %N : index
                %out_valid_0_1 = arith.cmpi slt, %tok_id_0_1, %num_valid_tokens : index
                %out_mask_0_1 = vector.broadcast %out_valid_0_1 : i1 to vector<1xi1>
                %tok_id_0_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_2] : memref<4663xi32>
                %tok_id_0_2 = arith.index_cast %tok_id_0_2_i32 : i32 to index
                %out_base_0_2 = arith.muli %tok_id_0_2, %N : index
                %out_valid_0_2 = arith.cmpi slt, %tok_id_0_2, %num_valid_tokens : index
                %out_mask_0_2 = vector.broadcast %out_valid_0_2 : i1 to vector<1xi1>
                %tok_id_0_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_3] : memref<4663xi32>
                %tok_id_0_3 = arith.index_cast %tok_id_0_3_i32 : i32 to index
                %out_base_0_3 = arith.muli %tok_id_0_3, %N : index
                %out_valid_0_3 = arith.cmpi slt, %tok_id_0_3, %num_valid_tokens : index
                %out_mask_0_3 = vector.broadcast %out_valid_0_3 : i1 to vector<1xi1>

                %tok_id_16_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_0] : memref<4663xi32>
                %tok_id_16_0 = arith.index_cast %tok_id_16_0_i32 : i32 to index
                %out_base_16_0 = arith.muli %tok_id_16_0, %N : index
                %out_valid_16_0 = arith.cmpi slt, %tok_id_16_0, %num_valid_tokens : index
                %out_mask_16_0 = vector.broadcast %out_valid_16_0 : i1 to vector<1xi1>
                %tok_id_16_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_1] : memref<4663xi32>
                %tok_id_16_1 = arith.index_cast %tok_id_16_1_i32 : i32 to index
                %out_base_16_1 = arith.muli %tok_id_16_1, %N : index
                %out_valid_16_1 = arith.cmpi slt, %tok_id_16_1, %num_valid_tokens : index
                %out_mask_16_1 = vector.broadcast %out_valid_16_1 : i1 to vector<1xi1>
                %tok_id_16_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_2] : memref<4663xi32>
                %tok_id_16_2 = arith.index_cast %tok_id_16_2_i32 : i32 to index
                %out_base_16_2 = arith.muli %tok_id_16_2, %N : index
                %out_valid_16_2 = arith.cmpi slt, %tok_id_16_2, %num_valid_tokens : index
                %out_mask_16_2 = vector.broadcast %out_valid_16_2 : i1 to vector<1xi1>
                %tok_id_16_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_3] : memref<4663xi32>
                %tok_id_16_3 = arith.index_cast %tok_id_16_3_i32 : i32 to index
                %out_base_16_3 = arith.muli %tok_id_16_3, %N : index
                %out_valid_16_3 = arith.cmpi slt, %tok_id_16_3, %num_valid_tokens : index
                %out_mask_16_3 = vector.broadcast %out_valid_16_3 : i1 to vector<1xi1>

                %tok_id_32_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_0] : memref<4663xi32>
                %tok_id_32_0 = arith.index_cast %tok_id_32_0_i32 : i32 to index
                %out_base_32_0 = arith.muli %tok_id_32_0, %N : index
                %out_valid_32_0 = arith.cmpi slt, %tok_id_32_0, %num_valid_tokens : index
                %out_mask_32_0 = vector.broadcast %out_valid_32_0 : i1 to vector<1xi1>
                %tok_id_32_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_1] : memref<4663xi32>
                %tok_id_32_1 = arith.index_cast %tok_id_32_1_i32 : i32 to index
                %out_base_32_1 = arith.muli %tok_id_32_1, %N : index
                %out_valid_32_1 = arith.cmpi slt, %tok_id_32_1, %num_valid_tokens : index
                %out_mask_32_1 = vector.broadcast %out_valid_32_1 : i1 to vector<1xi1>
                %tok_id_32_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_2] : memref<4663xi32>
                %tok_id_32_2 = arith.index_cast %tok_id_32_2_i32 : i32 to index
                %out_base_32_2 = arith.muli %tok_id_32_2, %N : index
                %out_valid_32_2 = arith.cmpi slt, %tok_id_32_2, %num_valid_tokens : index
                %out_mask_32_2 = vector.broadcast %out_valid_32_2 : i1 to vector<1xi1>
                %tok_id_32_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_3] : memref<4663xi32>
                %tok_id_32_3 = arith.index_cast %tok_id_32_3_i32 : i32 to index
                %out_base_32_3 = arith.muli %tok_id_32_3, %N : index
                %out_valid_32_3 = arith.cmpi slt, %tok_id_32_3, %num_valid_tokens : index
                %out_mask_32_3 = vector.broadcast %out_valid_32_3 : i1 to vector<1xi1>

                %tok_id_48_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_0] : memref<4663xi32>
                %tok_id_48_0 = arith.index_cast %tok_id_48_0_i32 : i32 to index
                %out_base_48_0 = arith.muli %tok_id_48_0, %N : index
                %out_valid_48_0 = arith.cmpi slt, %tok_id_48_0, %num_valid_tokens : index
                %out_mask_48_0 = vector.broadcast %out_valid_48_0 : i1 to vector<1xi1>
                %tok_id_48_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_1] : memref<4663xi32>
                %tok_id_48_1 = arith.index_cast %tok_id_48_1_i32 : i32 to index
                %out_base_48_1 = arith.muli %tok_id_48_1, %N : index
                %out_valid_48_1 = arith.cmpi slt, %tok_id_48_1, %num_valid_tokens : index
                %out_mask_48_1 = vector.broadcast %out_valid_48_1 : i1 to vector<1xi1>
                %tok_id_48_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_2] : memref<4663xi32>
                %tok_id_48_2 = arith.index_cast %tok_id_48_2_i32 : i32 to index
                %out_base_48_2 = arith.muli %tok_id_48_2, %N : index
                %out_valid_48_2 = arith.cmpi slt, %tok_id_48_2, %num_valid_tokens : index
                %out_mask_48_2 = vector.broadcast %out_valid_48_2 : i1 to vector<1xi1>
                %tok_id_48_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_3] : memref<4663xi32>
                %tok_id_48_3 = arith.index_cast %tok_id_48_3_i32 : i32 to index
                %out_base_48_3 = arith.muli %tok_id_48_3, %N : index
                %out_valid_48_3 = arith.cmpi slt, %tok_id_48_3, %num_valid_tokens : index
                %out_mask_48_3 = vector.broadcast %out_valid_48_3 : i1 to vector<1xi1>

                // pid_n determines which 64-neuron block we're computing
                %out_col_base = arith.muli %pid_n, %BLOCK_SIZE_N : index

                // Column offsets for the 4 column tiles
                %out_col_0 = arith.addi %out_col_base, %store_col_0 : index
                %out_col_1 = arith.addi %out_col_base, %store_col_1 : index
                %out_col_2 = arith.addi %out_col_base, %store_col_2 : index
                %out_col_3 = arith.addi %out_col_base, %store_col_3 : index

                // Write all 16 tiles using vector.store
                // Tile (0,0)
                %idx_00_0 = arith.addi %out_base_0_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_0], %out_mask_0_0, %r00_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_00_1 = arith.addi %out_base_0_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_1], %out_mask_0_1, %r00_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_00_2 = arith.addi %out_base_0_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_2], %out_mask_0_2, %r00_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_00_3 = arith.addi %out_base_0_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_3], %out_mask_0_3, %r00_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (0,1)
                %idx_01_0 = arith.addi %out_base_0_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_0], %out_mask_0_0, %r01_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_01_1 = arith.addi %out_base_0_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_1], %out_mask_0_1, %r01_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_01_2 = arith.addi %out_base_0_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_2], %out_mask_0_2, %r01_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_01_3 = arith.addi %out_base_0_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_3], %out_mask_0_3, %r01_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (0,2)
                %idx_02_0 = arith.addi %out_base_0_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_0], %out_mask_0_0, %r02_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_02_1 = arith.addi %out_base_0_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_1], %out_mask_0_1, %r02_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_02_2 = arith.addi %out_base_0_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_2], %out_mask_0_2, %r02_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_02_3 = arith.addi %out_base_0_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_3], %out_mask_0_3, %r02_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (0,3)
                %idx_03_0 = arith.addi %out_base_0_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_0], %out_mask_0_0, %r03_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_03_1 = arith.addi %out_base_0_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_1], %out_mask_0_1, %r03_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_03_2 = arith.addi %out_base_0_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_2], %out_mask_0_2, %r03_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_03_3 = arith.addi %out_base_0_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_3], %out_mask_0_3, %r03_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,0)
                %idx_10_0 = arith.addi %out_base_16_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_0], %out_mask_16_0, %r10_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_10_1 = arith.addi %out_base_16_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_1], %out_mask_16_0, %r10_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_10_2 = arith.addi %out_base_16_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_2], %out_mask_16_0, %r10_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_10_3 = arith.addi %out_base_16_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_3], %out_mask_16_0, %r10_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,1)
                %idx_11_0 = arith.addi %out_base_16_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_0], %out_mask_16_0, %r11_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_11_1 = arith.addi %out_base_16_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_1], %out_mask_16_1, %r11_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_11_2 = arith.addi %out_base_16_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_2], %out_mask_16_2, %r11_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_11_3 = arith.addi %out_base_16_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_3], %out_mask_16_3, %r11_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,2)
                %idx_12_0 = arith.addi %out_base_16_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_0], %out_mask_16_0, %r12_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_12_1 = arith.addi %out_base_16_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_1], %out_mask_16_1, %r12_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_12_2 = arith.addi %out_base_16_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_2], %out_mask_16_2, %r12_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_12_3 = arith.addi %out_base_16_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_3], %out_mask_16_3, %r12_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,3)
                %idx_13_0 = arith.addi %out_base_16_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_0], %out_mask_16_0, %r13_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_13_1 = arith.addi %out_base_16_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_1], %out_mask_16_1, %r13_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_13_2 = arith.addi %out_base_16_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_2], %out_mask_16_2, %r13_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_13_3 = arith.addi %out_base_16_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_3], %out_mask_16_3, %r13_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,0)
                %idx_20_0 = arith.addi %out_base_32_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_0], %out_mask_32_0, %r20_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_20_1 = arith.addi %out_base_32_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_1], %out_mask_32_1, %r20_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_20_2 = arith.addi %out_base_32_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_2], %out_mask_32_2, %r20_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_20_3 = arith.addi %out_base_32_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_3], %out_mask_32_3, %r20_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,1)
                %idx_21_0 = arith.addi %out_base_32_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_0], %out_mask_32_0, %r21_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_21_1 = arith.addi %out_base_32_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_1], %out_mask_32_1, %r21_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_21_2 = arith.addi %out_base_32_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_2], %out_mask_32_2, %r21_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_21_3 = arith.addi %out_base_32_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_3], %out_mask_32_3, %r21_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,2)
                %idx_22_0 = arith.addi %out_base_32_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_0], %out_mask_32_0, %r22_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_22_1 = arith.addi %out_base_32_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_1], %out_mask_32_1, %r22_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_22_2 = arith.addi %out_base_32_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_2], %out_mask_32_2, %r22_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_22_3 = arith.addi %out_base_32_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_3], %out_mask_32_3, %r22_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,3)
                %idx_23_0 = arith.addi %out_base_32_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_0], %out_mask_32_0, %r23_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_23_1 = arith.addi %out_base_32_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_1], %out_mask_32_1, %r23_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_23_2 = arith.addi %out_base_32_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_2], %out_mask_32_2, %r23_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_23_3 = arith.addi %out_base_32_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_3], %out_mask_32_3, %r23_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,0)
                %idx_30_0 = arith.addi %out_base_48_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_0], %out_mask_48_0, %r30_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_30_1 = arith.addi %out_base_48_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_1], %out_mask_48_1, %r30_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_30_2 = arith.addi %out_base_48_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_2], %out_mask_48_2, %r30_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_30_3 = arith.addi %out_base_48_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_3], %out_mask_48_3, %r30_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,1)
                %idx_31_0 = arith.addi %out_base_48_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_0], %out_mask_48_0, %r31_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_31_1 = arith.addi %out_base_48_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_1], %out_mask_48_1, %r31_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_31_2 = arith.addi %out_base_48_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_2], %out_mask_48_2, %r31_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_31_3 = arith.addi %out_base_48_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_3], %out_mask_48_3, %r31_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,2)
                %idx_32_0 = arith.addi %out_base_48_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_0], %out_mask_48_0, %r32_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_32_1 = arith.addi %out_base_48_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_1], %out_mask_48_1, %r32_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_32_2 = arith.addi %out_base_48_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_2], %out_mask_48_2, %r32_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_32_3 = arith.addi %out_base_48_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_3], %out_mask_48_3, %r32_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,3)
                %idx_33_0 = arith.addi %out_base_48_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_0], %out_mask_48_0, %r33_0_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_33_1 = arith.addi %out_base_48_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_1], %out_mask_48_1, %r33_1_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_33_2 = arith.addi %out_base_48_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_2], %out_mask_48_2, %r33_2_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                %idx_33_3 = arith.addi %out_base_48_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_3], %out_mask_48_3, %r33_3_f16 : memref<4194304xf16>, vector<1xi1>, vector<1xf16>
                }
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
            // %a_ptr: memref<2048x256xf16>,
            // %b_ptr: memref<8x1024x256xf16>,
            // %sorted_token_ids_ptr: memref<4663xi32>,
            // %expert_ids_ptr: memref<73xi32>,
            // %num_tokens_post_padded_ptr: memref<1xi32>,
            // %c_ptr: memref<2048x2x1024xf16>
            %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<2048x256xf16>
            %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x1024x256xf16>
            %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<4663xi32>
            %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<73xi32>
            %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
            %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<2048x2x1024xf16>
            %6 = flow.dispatch @fused_moe_kernel::@fused_moe_kernel(%0, %1, %2, %3, %4, %5) : (tensor<2048x256xf16>, tensor<8x1024x256xf16>, tensor<4663xi32>, tensor<73xi32>, tensor<1xi32>, tensor<2048x2x1024xf16>) -> %5
            %7 = hal.tensor.barrier join(%6 : tensor<2048x2x1024xf16>) => %arg7 : !hal.fence
            %8 = hal.tensor.export %7 : tensor<2048x2x1024xf16> -> !hal.buffer_view
            return %8 : !hal.buffer_view
        }
        }
            """
        )
        
    asm_dtype0_256_128_8_64_2_33_mfma = (
            """
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

        #map_load_row = affine_map<()[s0] -> (s0 mod 16)>
        #map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

        #map_store_col = affine_map<()[s0] -> (s0 mod 16)>
        #map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

        module attributes {transform.with_named_sequence} {
        stream.executable private @fused_moe_kernel {
            stream.executable.export public @fused_moe_kernel workgroups() -> (index, index, index) {
            %c40 = arith.constant 40 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            stream.return %c40, %c1, %c1 : index, index, index
            }
            builtin.module {
            func.func @fused_moe_kernel(
                // Input memrefs
            // %a_ptr: memref<33x128xf16>,
            // %b_ptr: memref<8x256x128xf16>,
            // %sorted_token_ids_ptr: memref<633xi32>,
            // %expert_ids_ptr: memref<10xi32>,
            // %num_tokens_post_padded_ptr: memref<1xi32>,
            // %c_ptr: memref<33x2x256xf16>
                %arg0: !stream.binding,
                %arg1: !stream.binding,
                %arg2: !stream.binding,
                %arg3: !stream.binding,
                %arg4: !stream.binding,
                %arg5: !stream.binding
            ) attributes {translation_info = #translation} {
                // N = 256
                // K = 128
                // EM = 633
                // top_k = 2
                // num_valid_tokens = 66
                // GROUP_SIZE_M = 8
                // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
                // BLOCK_SIZE_K = 32
                %N = arith.constant 256 : index
                %K = arith.constant 128 : index
                %EM = arith.constant 633 : index
                %top_k = arith.constant 2 : index
                %num_valid_tokens = arith.constant 66 : index
                %GROUP_SIZE_M = arith.constant 8 : index
                %BLOCK_SIZE_M = arith.constant 64 : index
                %BLOCK_SIZE_N = arith.constant 64 : index
                %BLOCK_SIZE_K = arith.constant 32 : index

                %c16384 = arith.constant 16384 : index
                %c32768 = arith.constant 32768 : index
                %c0 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                %c2 = arith.constant 2 : index
                %c3 = arith.constant 3 : index
                %c16 = arith.constant 16 : index
                %c32 = arith.constant 32 : index
                %c48 = arith.constant 48 : index
                %c63 = arith.constant 63 : index
                %c127 = arith.constant 127 : index
                %f0 = arith.constant 0.0 : f32
                %f0_f16 = arith.constant 0.0 : f16
                %cst_mfma = arith.constant dense<0.000000e+00> : vector<4xf32>

                %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<33x128xf16>
                %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x256x128xf16>
                %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<33x2x256xf16>
                %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<633xi32>
                %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<10xi32>
                %num_tokens_post_padded_ptr = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1xi32>

                // Program ID mapping
                %pid = gpu.block_id x
                %num_pid_m = arith.ceildivui %EM, %BLOCK_SIZE_M : index
                %num_pid_n = arith.ceildivui %N, %BLOCK_SIZE_N : index
                %num_pid_in_group = arith.muli %GROUP_SIZE_M, %num_pid_n : index
                %group_id = arith.divui %pid, %num_pid_in_group : index
                %first_pid_m = arith.muli %group_id, %GROUP_SIZE_M : index
                %min_group_size_m = arith.subi %num_pid_m, %first_pid_m : index
                %group_size_m = arith.minui %GROUP_SIZE_M, %min_group_size_m : index
                %0 = arith.remsi %pid, %num_pid_in_group : index
                %1 = arith.remsi %0, %group_size_m : index
                %pid_m = arith.addi %first_pid_m, %1 : index
                %pid_n = arith.divui %0, %group_size_m : index

                %thread_id = gpu.thread_id x upper_bound 64

                // Early exit check
                %2 = memref.load %num_tokens_post_padded_ptr[%c0] : memref<1xi32>
                %num_tokens_post_padded = arith.index_cast %2 : i32 to index
                %pid_m_offset = arith.muli %pid_m, %BLOCK_SIZE_M : index
                %should_exit = arith.cmpi sge, %pid_m_offset, %num_tokens_post_padded : index
                scf.if %should_exit {
                scf.yield
                } else {
                // Compute token mask
                %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index
                %thread_token_id = arith.addi %offs_token_id_base, %thread_id : index

                // Load token ID for this row
                %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<633xi32>
                %token_id = arith.index_cast %token_id_val : i32 to index

                %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
                %token_mask = vector.broadcast %token_valid : i1 to vector<32xi1>

                // Compute A row index: token_id // top_k
                %a_row = arith.divui %token_id, %top_k : index

                // Load expert ID
                %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<10xi32>
                %expert_id = arith.index_cast %expert_id_val : i32 to index

                // Compute B row offset for this thread
                %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
                %b_row = arith.addi %offs_bn_base, %thread_id : index

                // Allocate shared memory: 64×32 for A, 64×32 for B (instead of 64×128)
                %c4096 = arith.constant 4096 : index
                %alloc = memref.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
                %shared_a = memref.view %alloc[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>>
                    to memref<64x32xf16, #gpu.address_space<workgroup>>
                %shared_b = memref.view %alloc[%c4096][] : memref<8192xi8, #gpu.address_space<workgroup>>
                    to memref<64x32xf16, #gpu.address_space<workgroup>>

        //%print = arith.cmpi eq, %thread_id, %c1 : index
        //
        //scf.if %print { gpu.printf "pid %d\\n", %pid : index }
        //scf.if %print { gpu.printf "pid_m %d\\n", %pid_m : index }
        //scf.if %print { gpu.printf "pid_n %d\\n",  %pid_n : index }
        //
        //%a0.0 = memref.load %shared_a[%c0, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[0][0] %f\\n", %a0.0 : f16 }
        //
        //%a0.1 = memref.load %shared_a[%c0, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[0][1] %f\\n", %a0.1 : f16 }
        //
        //%a0.127 = memref.load %shared_a[%c0, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[0][127] %f\\n", %a0.127 : f16 }
        //
        //%a1.0 = memref.load %shared_a[%c1, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[1][0] %f\\n", %a1.0 : f16 }
        //
        //%a1.1 = memref.load %shared_a[%c1, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[1][1] %f\\n", %a1.1 : f16 }
        //
        //%a1.127 = memref.load %shared_a[%c1, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[1][127] %f\\n", %a1.127 : f16 }
        //
        //%a63.0 = memref.load %shared_a[%c63, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[63][0] %f\\n", %a63.0 : f16 }
        //
        //%a63.1 = memref.load %shared_a[%c63, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[63][1] %f\\n", %a63.1 : f16 }
        //
        //%a63.127 = memref.load %shared_a[%c63, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "a[63][127] %f\\n", %a63.127 : f16 }
        //
        //%b0.0 = memref.load %shared_b[%c0, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[0][0] %f\\n", %b0.0 : f16 }
        //
        //%b0.1 = memref.load %shared_b[%c0, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[0][1] %f\\n", %b0.1 : f16 }
        //
        //%b0.127 = memref.load %shared_b[%c0, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[0][127] %f\\n", %b0.127 : f16 }
        //
        //%b1.0 = memref.load %shared_b[%c1, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[1][0] %f\\n", %b1.0 : f16 }
        //
        //%b1.1 = memref.load %shared_b[%c1, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[1][1] %f\\n", %b1.1 : f16 }
        //
        //%b1.127 = memref.load %shared_b[%c1, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[1][127] %f\\n", %b1.127 : f16 }
        //
        //%b63.0 = memref.load %shared_b[%c63, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[63][0] %f\\n", %b63.0 : f16 }
        //
        //%b63.1 = memref.load %shared_b[%c63, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[63][1] %f\\n", %b63.1 : f16 }
        //
        //%b63.127 = memref.load %shared_b[%c63, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "b[63][127] %f\\n", %b63.127 : f16 }
        //
        //amdgpu.lds_barrier

                // Thread-level indices for MFMA loading
                %load_col = affine.apply #map_load_col()[%thread_id]  // 0, 4, 8, 12 (first 16 elements of K)
                %load_row = affine.apply #map_load_row()[%thread_id]
                %load_row_1 = arith.addi %load_row, %c16 : index
                %load_row_2 = arith.addi %load_row, %c32 : index
                %load_row_3 = arith.addi %load_row, %c48 : index

                // Compute column indices for first and second half of K (split 32 into 16+16)
                %load_col_k = arith.addi %load_col, %c16 : index  // 16, 20, 24, 28 (second 16 elements of K)

                // =========================================================================
                // PROLOGUE: Load first iteration (K=0)
                // =========================================================================
                %k_start_0 = arith.constant 0 : index

                // Each thread loads its 32-element slice from A (from k_start to k_start+32)
                %a_row_vec_0 = vector.transfer_read %a_ptr[%a_row, %k_start_0], %f0_f16, %token_mask :
                    memref<33x128xf16>, vector<32xf16>
                // Store to shared memory
                vector.store %a_row_vec_0, %shared_a[%thread_id, %c0] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<32xf16>


                // Each thread loads its 32-element slice from B (from k_start to k_start+32)
                // B is [8, 256, 128], we need [expert_id, b_row, k_start]
                // Note: b_row is always < 256 since pid_n * 64 + thread_id_x < 256
                %b_row_vec_0 = vector.transfer_read %b_ptr[%expert_id, %b_row, %k_start_0], %f0_f16 :
                    memref<8x256x128xf16>, vector<32xf16>
                // Store to shared memory
                vector.store %b_row_vec_0, %shared_b[%thread_id, %c0] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<32xf16>

                amdgpu.lds_barrier

                %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
                %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

                // =========================================================================
                // MAIN LOOP: Process iterations 0 to N-2
                // =========================================================================
                %result:16 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
                    iter_args(%a00=%cst_mfma, %a01=%cst_mfma, %a02=%cst_mfma, %a03=%cst_mfma,
                                %a10=%cst_mfma, %a11=%cst_mfma, %a12=%cst_mfma, %a13=%cst_mfma,
                                %a20=%cst_mfma, %a21=%cst_mfma, %a22=%cst_mfma, %a23=%cst_mfma,
                                %a30=%cst_mfma, %a31=%cst_mfma, %a32=%cst_mfma, %a33=%cst_mfma)
                    -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {

                    // Compute K offset for this iteration
                    %k_start = arith.muli %k_block, %BLOCK_SIZE_K : index
                    %k_col = arith.addi %k_start, %load_col : index
                    %k_col_k = arith.addi %k_start, %load_col_k : index

        //gpu.printf "pid_m %d pid_n %d thread %d k_start %d load_col %d\\n", %pid_m, %pid_n, %thread_id, %k_start, %load_col : index, index, index, index, index

                    // =========================================================================
                    // FIRST HALF: K[0:16] - Load from shared memory
                    // =========================================================================

                    // Load A vectors for first half: 4 M tiles
                    %a0 = vector.load %shared_a[%load_row, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a1 = vector.load %shared_a[%load_row_1, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a2 = vector.load %shared_a[%load_row_2, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a3 = vector.load %shared_a[%load_row_3, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // Load B vectors for first half: 4 N tiles
                    // Note: B is stored as [64, 32] where rows are output features
                    // For MFMA, we need B[n, k], which maps to shared_b[load_row, load_col]
                    %b0 = vector.load %shared_b[%load_row, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b1 = vector.load %shared_b[%load_row_1, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b2 = vector.load %shared_b[%load_row_2, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b3 = vector.load %shared_b[%load_row_3, %load_col] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // =========================================================================
                    // PREFETCH NEXT ITERATION from global memory
                    // =========================================================================
                    %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index

                    %a_row_vec_next = vector.transfer_read %a_ptr[%a_row, %k_start_next], %f0_f16, %token_mask :
                    memref<33x128xf16>, vector<32xf16>
                    %b_row_vec_next = vector.transfer_read %b_ptr[%expert_id, %b_row, %k_start_next], %f0_f16 :
                    memref<8x256x128xf16>, vector<32xf16>

                    // =========================================================================
                    // SECOND HALF: K[16:32] - Load from shared memory
                    // =========================================================================

                    // Load A vectors for second half: 4 M tiles
                    %a0k = vector.load %shared_a[%load_row, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a1k = vector.load %shared_a[%load_row_1, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a2k = vector.load %shared_a[%load_row_2, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %a3k = vector.load %shared_a[%load_row_3, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // Load B vectors for second half: 4 N tiles
                    %b0k = vector.load %shared_b[%load_row, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b1k = vector.load %shared_b[%load_row_1, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b2k = vector.load %shared_b[%load_row_2, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                    %b3k = vector.load %shared_b[%load_row_3, %load_col_k] :
                        memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                    // =========================================================================
                    // MFMA OPERATIONS - FIRST HALF (K[0:16])
                    // =========================================================================

                    // Tile (0,0)
                    %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,1)
                    %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,2)
                    %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,3)
                    %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,0)
                    %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,1)
                    %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,2)
                    %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,3)
                    %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,0)
                    %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,1)
                    %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,2)
                    %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,3)
                    %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,0)
                    %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,1)
                    %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,2)
                    %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,3)
                    %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // =========================================================================
                    // STORE PREFETCHED DATA to shared memory (after first half compute)
                    // =========================================================================
                    amdgpu.lds_barrier

                    vector.store %a_row_vec_next, %shared_a[%thread_id, %c0] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<32xf16>
                    vector.store %b_row_vec_next, %shared_b[%thread_id, %c0] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<32xf16>

                    amdgpu.lds_barrier

                    // =========================================================================
                    // MFMA OPERATIONS - SECOND HALF (K[16:32]) accumulate on 1st half results
                    // =========================================================================
                    // Tile (0,0)
                    %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,1)
                    %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,2)
                    %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (0,3)
                    %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (1,0)
                    %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,1)
                    %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,2)
                    %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (1,3)
                    %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (2,0)
                    %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,1)
                    %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,2)
                    %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (2,3)
                    %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    // Tile (3,0)
                    %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,1)
                    %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,2)
                    %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                    // Tile (3,3)
                    %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                    scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                            %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
                }

                // =========================================================================
                // EPILOGUE: Process last iteration (K = num_blocks - 1)
                // =========================================================================
                // Load first half from shared memory
                %a0_last = vector.load %shared_a[%load_row, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a1_last = vector.load %shared_a[%load_row_1, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a2_last = vector.load %shared_a[%load_row_2, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a3_last = vector.load %shared_a[%load_row_3, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                %b0_last = vector.load %shared_b[%load_row, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b1_last = vector.load %shared_b[%load_row_1, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b2_last = vector.load %shared_b[%load_row_2, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b3_last = vector.load %shared_b[%load_row_3, %load_col] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                // Load second half from shared memory
                %a0_k_last = vector.load %shared_a[%load_row, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a1_k_last = vector.load %shared_a[%load_row_1, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a2_k_last = vector.load %shared_a[%load_row_2, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %a3_k_last = vector.load %shared_a[%load_row_3, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                %b0_k_last = vector.load %shared_b[%load_row, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b1_k_last = vector.load %shared_b[%load_row_1, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b2_k_last = vector.load %shared_b[%load_row_2, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
                %b3_k_last = vector.load %shared_b[%load_row_3, %load_col_k] :
                    memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

                // Compute first half
                %r00_0_last = amdgpu.mfma %a0_last * %b0_last + %result#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r01_0_last = amdgpu.mfma %a0_last * %b1_last + %result#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r02_0_last = amdgpu.mfma %a0_last * %b2_last + %result#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r03_0_last = amdgpu.mfma %a0_last * %b3_last + %result#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r10_0_last = amdgpu.mfma %a1_last * %b0_last + %result#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r11_0_last = amdgpu.mfma %a1_last * %b1_last + %result#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r12_0_last = amdgpu.mfma %a1_last * %b2_last + %result#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r13_0_last = amdgpu.mfma %a1_last * %b3_last + %result#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r20_0_last = amdgpu.mfma %a2_last * %b0_last + %result#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r21_0_last = amdgpu.mfma %a2_last * %b1_last + %result#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r22_0_last = amdgpu.mfma %a2_last * %b2_last + %result#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r23_0_last = amdgpu.mfma %a2_last * %b3_last + %result#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r30_0_last = amdgpu.mfma %a3_last * %b0_last + %result#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r31_0_last = amdgpu.mfma %a3_last * %b1_last + %result#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r32_0_last = amdgpu.mfma %a3_last * %b2_last + %result#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r33_0_last = amdgpu.mfma %a3_last * %b3_last + %result#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                // Compute second half (final results)
                %r00_final = amdgpu.mfma %a0_k_last * %b0_k_last + %r00_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r01_final = amdgpu.mfma %a0_k_last * %b1_k_last + %r01_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r02_final = amdgpu.mfma %a0_k_last * %b2_k_last + %r02_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r03_final = amdgpu.mfma %a0_k_last * %b3_k_last + %r03_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r10_final = amdgpu.mfma %a1_k_last * %b0_k_last + %r10_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r11_final = amdgpu.mfma %a1_k_last * %b1_k_last + %r11_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r12_final = amdgpu.mfma %a1_k_last * %b2_k_last + %r12_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r13_final = amdgpu.mfma %a1_k_last * %b3_k_last + %r13_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r20_final = amdgpu.mfma %a2_k_last * %b0_k_last + %r20_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r21_final = amdgpu.mfma %a2_k_last * %b1_k_last + %r21_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r22_final = amdgpu.mfma %a2_k_last * %b2_k_last + %r22_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r23_final = amdgpu.mfma %a2_k_last * %b3_k_last + %r23_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                %r30_final = amdgpu.mfma %a3_k_last * %b0_k_last + %r30_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r31_final = amdgpu.mfma %a3_k_last * %b1_k_last + %r31_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r32_final = amdgpu.mfma %a3_k_last * %b2_k_last + %r32_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
                %r33_final = amdgpu.mfma %a3_k_last * %b3_k_last + %r33_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

                // =========================================================================
                // STORE RESULTS
                // =========================================================================

                // Truncate to f16
                %r00_f16 = arith.truncf %r00_final : vector<4xf32> to vector<4xf16>
                %r01_f16 = arith.truncf %r01_final : vector<4xf32> to vector<4xf16>
                %r02_f16 = arith.truncf %r02_final : vector<4xf32> to vector<4xf16>
                %r03_f16 = arith.truncf %r03_final : vector<4xf32> to vector<4xf16>
                %r10_f16 = arith.truncf %r10_final : vector<4xf32> to vector<4xf16>
                %r11_f16 = arith.truncf %r11_final : vector<4xf32> to vector<4xf16>
                %r12_f16 = arith.truncf %r12_final : vector<4xf32> to vector<4xf16>
                %r13_f16 = arith.truncf %r13_final : vector<4xf32> to vector<4xf16>
                %r20_f16 = arith.truncf %r20_final : vector<4xf32> to vector<4xf16>
                %r21_f16 = arith.truncf %r21_final : vector<4xf32> to vector<4xf16>
                %r22_f16 = arith.truncf %r22_final : vector<4xf32> to vector<4xf16>
                %r23_f16 = arith.truncf %r23_final : vector<4xf32> to vector<4xf16>
                %r30_f16 = arith.truncf %r30_final : vector<4xf32> to vector<4xf16>
                %r31_f16 = arith.truncf %r31_final : vector<4xf32> to vector<4xf16>
                %r32_f16 = arith.truncf %r32_final : vector<4xf32> to vector<4xf16>
                %r33_f16 = arith.truncf %r33_final : vector<4xf32> to vector<4xf16>

                %store_col_0 = affine.apply #map_store_col()[%thread_id]
                %store_col_1 = arith.addi %store_col_0, %c16 : index
                %store_col_2 = arith.addi %store_col_0, %c32 : index
                %store_col_3 = arith.addi %store_col_0, %c48 : index
                %store_row_0_0 = affine.apply #map_store_row()[%thread_id]
                %store_row_0_1 = arith.addi %store_row_0_0, %c1 : index
                %store_row_0_2 = arith.addi %store_row_0_0, %c2 : index
                %store_row_0_3 = arith.addi %store_row_0_0, %c3 : index
                %store_row_16_0 = arith.addi %store_row_0_0, %c16 : index
                %store_row_16_1 = arith.addi %store_row_16_0, %c1 : index
                %store_row_16_2 = arith.addi %store_row_16_0, %c2 : index
                %store_row_16_3 = arith.addi %store_row_16_0, %c3 : index
                %store_row_32_0 = arith.addi %store_row_0_0, %c32 : index
                %store_row_32_1 = arith.addi %store_row_32_0, %c1 : index
                %store_row_32_2 = arith.addi %store_row_32_0, %c2 : index
                %store_row_32_3 = arith.addi %store_row_32_0, %c3 : index
                %store_row_48_0 = arith.addi %store_row_0_0, %c48 : index
                %store_row_48_1 = arith.addi %store_row_48_0, %c1 : index
                %store_row_48_2 = arith.addi %store_row_48_0, %c2 : index
                %store_row_48_3 = arith.addi %store_row_48_0, %c3 : index

                %r00_0_f16 = vector.extract_strided_slice %r00_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_1_f16 = vector.extract_strided_slice %r00_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_2_f16 = vector.extract_strided_slice %r00_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r00_3_f16 = vector.extract_strided_slice %r00_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_0_f16 = vector.extract_strided_slice %r01_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_1_f16 = vector.extract_strided_slice %r01_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_2_f16 = vector.extract_strided_slice %r01_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r01_3_f16 = vector.extract_strided_slice %r01_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_0_f16 = vector.extract_strided_slice %r02_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_1_f16 = vector.extract_strided_slice %r02_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_2_f16 = vector.extract_strided_slice %r02_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r02_3_f16 = vector.extract_strided_slice %r02_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_0_f16 = vector.extract_strided_slice %r03_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_1_f16 = vector.extract_strided_slice %r03_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_2_f16 = vector.extract_strided_slice %r03_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r03_3_f16 = vector.extract_strided_slice %r03_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r10_0_f16 = vector.extract_strided_slice %r10_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_1_f16 = vector.extract_strided_slice %r10_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_2_f16 = vector.extract_strided_slice %r10_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r10_3_f16 = vector.extract_strided_slice %r10_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_0_f16 = vector.extract_strided_slice %r11_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_1_f16 = vector.extract_strided_slice %r11_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_2_f16 = vector.extract_strided_slice %r11_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r11_3_f16 = vector.extract_strided_slice %r11_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_0_f16 = vector.extract_strided_slice %r12_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_1_f16 = vector.extract_strided_slice %r12_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_2_f16 = vector.extract_strided_slice %r12_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r12_3_f16 = vector.extract_strided_slice %r12_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_0_f16 = vector.extract_strided_slice %r13_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_1_f16 = vector.extract_strided_slice %r13_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_2_f16 = vector.extract_strided_slice %r13_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r13_3_f16 = vector.extract_strided_slice %r13_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r20_0_f16 = vector.extract_strided_slice %r20_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_1_f16 = vector.extract_strided_slice %r20_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_2_f16 = vector.extract_strided_slice %r20_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r20_3_f16 = vector.extract_strided_slice %r20_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_0_f16 = vector.extract_strided_slice %r21_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_1_f16 = vector.extract_strided_slice %r21_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_2_f16 = vector.extract_strided_slice %r21_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r21_3_f16 = vector.extract_strided_slice %r21_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_0_f16 = vector.extract_strided_slice %r22_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_1_f16 = vector.extract_strided_slice %r22_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_2_f16 = vector.extract_strided_slice %r22_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r22_3_f16 = vector.extract_strided_slice %r22_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_0_f16 = vector.extract_strided_slice %r23_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_1_f16 = vector.extract_strided_slice %r23_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_2_f16 = vector.extract_strided_slice %r23_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r23_3_f16 = vector.extract_strided_slice %r23_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

                %r30_0_f16 = vector.extract_strided_slice %r30_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_1_f16 = vector.extract_strided_slice %r30_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_2_f16 = vector.extract_strided_slice %r30_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r30_3_f16 = vector.extract_strided_slice %r30_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_0_f16 = vector.extract_strided_slice %r31_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_1_f16 = vector.extract_strided_slice %r31_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_2_f16 = vector.extract_strided_slice %r31_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r31_3_f16 = vector.extract_strided_slice %r31_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_0_f16 = vector.extract_strided_slice %r32_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_1_f16 = vector.extract_strided_slice %r32_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_2_f16 = vector.extract_strided_slice %r32_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r32_3_f16 = vector.extract_strided_slice %r32_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_0_f16 = vector.extract_strided_slice %r33_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_1_f16 = vector.extract_strided_slice %r33_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_2_f16 = vector.extract_strided_slice %r33_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
                %r33_3_f16 = vector.extract_strided_slice %r33_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

        //vector.store %r00_0_f16, %shared_c[%store_row_0_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r00_1_f16, %shared_c[%store_row_0_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r00_2_f16, %shared_c[%store_row_0_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r00_3_f16, %shared_c[%store_row_0_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r01_0_f16, %shared_c[%store_row_0_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r01_1_f16, %shared_c[%store_row_0_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r01_2_f16, %shared_c[%store_row_0_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r01_3_f16, %shared_c[%store_row_0_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r02_0_f16, %shared_c[%store_row_0_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r02_1_f16, %shared_c[%store_row_0_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r02_2_f16, %shared_c[%store_row_0_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r02_3_f16, %shared_c[%store_row_0_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r03_0_f16, %shared_c[%store_row_0_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r03_1_f16, %shared_c[%store_row_0_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r03_2_f16, %shared_c[%store_row_0_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r03_3_f16, %shared_c[%store_row_0_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //
        //vector.store %r10_0_f16, %shared_c[%store_row_16_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r10_1_f16, %shared_c[%store_row_16_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r10_2_f16, %shared_c[%store_row_16_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r10_3_f16, %shared_c[%store_row_16_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r11_0_f16, %shared_c[%store_row_16_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r11_1_f16, %shared_c[%store_row_16_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r11_2_f16, %shared_c[%store_row_16_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r11_3_f16, %shared_c[%store_row_16_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r12_0_f16, %shared_c[%store_row_16_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r12_1_f16, %shared_c[%store_row_16_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r12_2_f16, %shared_c[%store_row_16_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r12_3_f16, %shared_c[%store_row_16_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r13_0_f16, %shared_c[%store_row_16_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r13_1_f16, %shared_c[%store_row_16_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r13_2_f16, %shared_c[%store_row_16_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r13_3_f16, %shared_c[%store_row_16_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //
        //vector.store %r20_0_f16, %shared_c[%store_row_32_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r20_1_f16, %shared_c[%store_row_32_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r20_2_f16, %shared_c[%store_row_32_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r20_3_f16, %shared_c[%store_row_32_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r21_0_f16, %shared_c[%store_row_32_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r21_1_f16, %shared_c[%store_row_32_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r21_2_f16, %shared_c[%store_row_32_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r21_3_f16, %shared_c[%store_row_32_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r22_0_f16, %shared_c[%store_row_32_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r22_1_f16, %shared_c[%store_row_32_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r22_2_f16, %shared_c[%store_row_32_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r22_3_f16, %shared_c[%store_row_32_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r23_0_f16, %shared_c[%store_row_32_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r23_1_f16, %shared_c[%store_row_32_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r23_2_f16, %shared_c[%store_row_32_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r23_3_f16, %shared_c[%store_row_32_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //
        //vector.store %r30_0_f16, %shared_c[%store_row_48_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r30_1_f16, %shared_c[%store_row_48_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r30_2_f16, %shared_c[%store_row_48_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r30_3_f16, %shared_c[%store_row_48_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r31_0_f16, %shared_c[%store_row_48_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r31_1_f16, %shared_c[%store_row_48_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r31_2_f16, %shared_c[%store_row_48_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r31_3_f16, %shared_c[%store_row_48_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r32_0_f16, %shared_c[%store_row_48_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r32_1_f16, %shared_c[%store_row_48_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r32_2_f16, %shared_c[%store_row_48_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r32_3_f16, %shared_c[%store_row_48_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r33_0_f16, %shared_c[%store_row_48_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r33_1_f16, %shared_c[%store_row_48_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r33_2_f16, %shared_c[%store_row_48_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //vector.store %r33_3_f16, %shared_c[%store_row_48_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
        //
        //%c0.0 = memref.load %shared_c[%c0, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[0][0] %f\\n", %c0.0 : f16 }
        //
        //%c0.1 = memref.load %shared_c[%c0, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[0][1] %f\\n", %c0.1 : f16 }
        //
        //%c0.63 = memref.load %shared_c[%c0, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[0][63] %f\\n", %c0.63 : f16 }
        //
        //%c1.0 = memref.load %shared_c[%c1, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[1][0] %f\\n", %c1.0 : f16 }
        //
        //%c1.1 = memref.load %shared_c[%c1, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[1][1] %f\\n", %c1.1 : f16 }
        //
        //%c1.63 = memref.load %shared_c[%c1, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[1][63] %f\\n", %c1.63 : f16 }
        //
        //%c2.0 = memref.load %shared_c[%c2, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[2][0] %f\\n", %c2.0 : f16 }
        //
        //%c2.1 = memref.load %shared_c[%c2, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[2][1] %f\\n", %c2.1 : f16 }
        //
        //%c2.63 = memref.load %shared_c[%c2, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[2][63] %f\\n", %c2.63 : f16 }
        //
        //%c63.0 = memref.load %shared_c[%c63, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[63][0] %f\\n", %c63.0 : f16 }
        //
        //%c63.1 = memref.load %shared_c[%c63, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[63][1] %f\\n", %c63.1 : f16 }
        //
        //%c63.63 = memref.load %shared_c[%c63, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
        //scf.if %print { gpu.printf "c[63][63] %f\\n", %c63.63 : f16 }
        //
        //amdgpu.lds_barrier

                // Flatten c_ptr for easier indexing
                %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<33x2x256xf16> into memref<16896xf16>

                // Each thread writes to 4 different rows (from load_row, load_row+16, load_row+32, load_row+48)
                // across 4 column groups (base, base+16, base+32, base+48)

                // Get token indices for output rows
                %out_token_0_0 = arith.addi %offs_token_id_base, %store_row_0_0 : index
                %out_token_0_1 = arith.addi %offs_token_id_base, %store_row_0_1 : index
                %out_token_0_2 = arith.addi %offs_token_id_base, %store_row_0_2 : index
                %out_token_0_3 = arith.addi %offs_token_id_base, %store_row_0_3 : index
                %out_token_16_0 = arith.addi %offs_token_id_base, %store_row_16_0 : index
                %out_token_16_1 = arith.addi %offs_token_id_base, %store_row_16_1 : index
                %out_token_16_2 = arith.addi %offs_token_id_base, %store_row_16_2 : index
                %out_token_16_3 = arith.addi %offs_token_id_base, %store_row_16_3 : index
                %out_token_32_0 = arith.addi %offs_token_id_base, %store_row_32_0 : index
                %out_token_32_1 = arith.addi %offs_token_id_base, %store_row_32_1 : index
                %out_token_32_2 = arith.addi %offs_token_id_base, %store_row_32_2 : index
                %out_token_32_3 = arith.addi %offs_token_id_base, %store_row_32_3 : index
                %out_token_48_0 = arith.addi %offs_token_id_base, %store_row_48_0 : index
                %out_token_48_1 = arith.addi %offs_token_id_base, %store_row_48_1 : index
                %out_token_48_2 = arith.addi %offs_token_id_base, %store_row_48_2 : index
                %out_token_48_3 = arith.addi %offs_token_id_base, %store_row_48_3 : index

                %tok_id_0_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_0] : memref<633xi32>
                %tok_id_0_0 = arith.index_cast %tok_id_0_0_i32 : i32 to index
                %out_base_0_0 = arith.muli %tok_id_0_0, %N : index
                %out_valid_0_0 = arith.cmpi slt, %tok_id_0_0, %num_valid_tokens : index
                %out_mask_0_0 = vector.broadcast %out_valid_0_0 : i1 to vector<1xi1>
                %tok_id_0_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_1] : memref<633xi32>
                %tok_id_0_1 = arith.index_cast %tok_id_0_1_i32 : i32 to index
                %out_base_0_1 = arith.muli %tok_id_0_1, %N : index
                %out_valid_0_1 = arith.cmpi slt, %tok_id_0_1, %num_valid_tokens : index
                %out_mask_0_1 = vector.broadcast %out_valid_0_1 : i1 to vector<1xi1>
                %tok_id_0_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_2] : memref<633xi32>
                %tok_id_0_2 = arith.index_cast %tok_id_0_2_i32 : i32 to index
                %out_base_0_2 = arith.muli %tok_id_0_2, %N : index
                %out_valid_0_2 = arith.cmpi slt, %tok_id_0_2, %num_valid_tokens : index
                %out_mask_0_2 = vector.broadcast %out_valid_0_2 : i1 to vector<1xi1>
                %tok_id_0_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_3] : memref<633xi32>
                %tok_id_0_3 = arith.index_cast %tok_id_0_3_i32 : i32 to index
                %out_base_0_3 = arith.muli %tok_id_0_3, %N : index
                %out_valid_0_3 = arith.cmpi slt, %tok_id_0_3, %num_valid_tokens : index
                %out_mask_0_3 = vector.broadcast %out_valid_0_3 : i1 to vector<1xi1>

                %tok_id_16_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_0] : memref<633xi32>
                %tok_id_16_0 = arith.index_cast %tok_id_16_0_i32 : i32 to index
                %out_base_16_0 = arith.muli %tok_id_16_0, %N : index
                %out_valid_16_0 = arith.cmpi slt, %tok_id_16_0, %num_valid_tokens : index
                %out_mask_16_0 = vector.broadcast %out_valid_16_0 : i1 to vector<1xi1>
                %tok_id_16_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_1] : memref<633xi32>
                %tok_id_16_1 = arith.index_cast %tok_id_16_1_i32 : i32 to index
                %out_base_16_1 = arith.muli %tok_id_16_1, %N : index
                %out_valid_16_1 = arith.cmpi slt, %tok_id_16_1, %num_valid_tokens : index
                %out_mask_16_1 = vector.broadcast %out_valid_16_1 : i1 to vector<1xi1>
                %tok_id_16_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_2] : memref<633xi32>
                %tok_id_16_2 = arith.index_cast %tok_id_16_2_i32 : i32 to index
                %out_base_16_2 = arith.muli %tok_id_16_2, %N : index
                %out_valid_16_2 = arith.cmpi slt, %tok_id_16_2, %num_valid_tokens : index
                %out_mask_16_2 = vector.broadcast %out_valid_16_2 : i1 to vector<1xi1>
                %tok_id_16_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_3] : memref<633xi32>
                %tok_id_16_3 = arith.index_cast %tok_id_16_3_i32 : i32 to index
                %out_base_16_3 = arith.muli %tok_id_16_3, %N : index
                %out_valid_16_3 = arith.cmpi slt, %tok_id_16_3, %num_valid_tokens : index
                %out_mask_16_3 = vector.broadcast %out_valid_16_3 : i1 to vector<1xi1>

                %tok_id_32_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_0] : memref<633xi32>
                %tok_id_32_0 = arith.index_cast %tok_id_32_0_i32 : i32 to index
                %out_base_32_0 = arith.muli %tok_id_32_0, %N : index
                %out_valid_32_0 = arith.cmpi slt, %tok_id_32_0, %num_valid_tokens : index
                %out_mask_32_0 = vector.broadcast %out_valid_32_0 : i1 to vector<1xi1>
                %tok_id_32_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_1] : memref<633xi32>
                %tok_id_32_1 = arith.index_cast %tok_id_32_1_i32 : i32 to index
                %out_base_32_1 = arith.muli %tok_id_32_1, %N : index
                %out_valid_32_1 = arith.cmpi slt, %tok_id_32_1, %num_valid_tokens : index
                %out_mask_32_1 = vector.broadcast %out_valid_32_1 : i1 to vector<1xi1>
                %tok_id_32_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_2] : memref<633xi32>
                %tok_id_32_2 = arith.index_cast %tok_id_32_2_i32 : i32 to index
                %out_base_32_2 = arith.muli %tok_id_32_2, %N : index
                %out_valid_32_2 = arith.cmpi slt, %tok_id_32_2, %num_valid_tokens : index
                %out_mask_32_2 = vector.broadcast %out_valid_32_2 : i1 to vector<1xi1>
                %tok_id_32_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_3] : memref<633xi32>
                %tok_id_32_3 = arith.index_cast %tok_id_32_3_i32 : i32 to index
                %out_base_32_3 = arith.muli %tok_id_32_3, %N : index
                %out_valid_32_3 = arith.cmpi slt, %tok_id_32_3, %num_valid_tokens : index
                %out_mask_32_3 = vector.broadcast %out_valid_32_3 : i1 to vector<1xi1>

                %tok_id_48_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_0] : memref<633xi32>
                %tok_id_48_0 = arith.index_cast %tok_id_48_0_i32 : i32 to index
                %out_base_48_0 = arith.muli %tok_id_48_0, %N : index
                %out_valid_48_0 = arith.cmpi slt, %tok_id_48_0, %num_valid_tokens : index
                %out_mask_48_0 = vector.broadcast %out_valid_48_0 : i1 to vector<1xi1>
                %tok_id_48_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_1] : memref<633xi32>
                %tok_id_48_1 = arith.index_cast %tok_id_48_1_i32 : i32 to index
                %out_base_48_1 = arith.muli %tok_id_48_1, %N : index
                %out_valid_48_1 = arith.cmpi slt, %tok_id_48_1, %num_valid_tokens : index
                %out_mask_48_1 = vector.broadcast %out_valid_48_1 : i1 to vector<1xi1>
                %tok_id_48_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_2] : memref<633xi32>
                %tok_id_48_2 = arith.index_cast %tok_id_48_2_i32 : i32 to index
                %out_base_48_2 = arith.muli %tok_id_48_2, %N : index
                %out_valid_48_2 = arith.cmpi slt, %tok_id_48_2, %num_valid_tokens : index
                %out_mask_48_2 = vector.broadcast %out_valid_48_2 : i1 to vector<1xi1>
                %tok_id_48_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_3] : memref<633xi32>
                %tok_id_48_3 = arith.index_cast %tok_id_48_3_i32 : i32 to index
                %out_base_48_3 = arith.muli %tok_id_48_3, %N : index
                %out_valid_48_3 = arith.cmpi slt, %tok_id_48_3, %num_valid_tokens : index
                %out_mask_48_3 = vector.broadcast %out_valid_48_3 : i1 to vector<1xi1>

                // pid_n determines which 64-neuron block we're computing
                %out_col_base = arith.muli %pid_n, %BLOCK_SIZE_N : index

                // Column offsets for the 4 column tiles
                %out_col_0 = arith.addi %out_col_base, %store_col_0 : index
                %out_col_1 = arith.addi %out_col_base, %store_col_1 : index
                %out_col_2 = arith.addi %out_col_base, %store_col_2 : index
                %out_col_3 = arith.addi %out_col_base, %store_col_3 : index

                // Write all 16 tiles using vector.store
                // Tile (0,0)
                %idx_00_0 = arith.addi %out_base_0_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_0], %out_mask_0_0, %r00_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_00_1 = arith.addi %out_base_0_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_1], %out_mask_0_1, %r00_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_00_2 = arith.addi %out_base_0_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_2], %out_mask_0_2, %r00_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_00_3 = arith.addi %out_base_0_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_00_3], %out_mask_0_3, %r00_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (0,1)
                %idx_01_0 = arith.addi %out_base_0_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_0], %out_mask_0_0, %r01_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_01_1 = arith.addi %out_base_0_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_1], %out_mask_0_1, %r01_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_01_2 = arith.addi %out_base_0_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_2], %out_mask_0_2, %r01_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_01_3 = arith.addi %out_base_0_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_01_3], %out_mask_0_3, %r01_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (0,2)
                %idx_02_0 = arith.addi %out_base_0_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_0], %out_mask_0_0, %r02_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_02_1 = arith.addi %out_base_0_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_1], %out_mask_0_1, %r02_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_02_2 = arith.addi %out_base_0_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_2], %out_mask_0_2, %r02_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_02_3 = arith.addi %out_base_0_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_02_3], %out_mask_0_3, %r02_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (0,3)
                %idx_03_0 = arith.addi %out_base_0_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_0], %out_mask_0_0, %r03_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_03_1 = arith.addi %out_base_0_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_1], %out_mask_0_1, %r03_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_03_2 = arith.addi %out_base_0_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_2], %out_mask_0_2, %r03_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_03_3 = arith.addi %out_base_0_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_03_3], %out_mask_0_3, %r03_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,0)
                %idx_10_0 = arith.addi %out_base_16_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_0], %out_mask_16_0, %r10_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_10_1 = arith.addi %out_base_16_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_1], %out_mask_16_0, %r10_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_10_2 = arith.addi %out_base_16_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_2], %out_mask_16_0, %r10_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_10_3 = arith.addi %out_base_16_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_10_3], %out_mask_16_0, %r10_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,1)
                %idx_11_0 = arith.addi %out_base_16_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_0], %out_mask_16_0, %r11_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_11_1 = arith.addi %out_base_16_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_1], %out_mask_16_1, %r11_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_11_2 = arith.addi %out_base_16_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_2], %out_mask_16_2, %r11_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_11_3 = arith.addi %out_base_16_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_11_3], %out_mask_16_3, %r11_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,2)
                %idx_12_0 = arith.addi %out_base_16_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_0], %out_mask_16_0, %r12_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_12_1 = arith.addi %out_base_16_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_1], %out_mask_16_1, %r12_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_12_2 = arith.addi %out_base_16_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_2], %out_mask_16_2, %r12_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_12_3 = arith.addi %out_base_16_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_12_3], %out_mask_16_3, %r12_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (1,3)
                %idx_13_0 = arith.addi %out_base_16_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_0], %out_mask_16_0, %r13_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_13_1 = arith.addi %out_base_16_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_1], %out_mask_16_1, %r13_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_13_2 = arith.addi %out_base_16_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_2], %out_mask_16_2, %r13_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_13_3 = arith.addi %out_base_16_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_13_3], %out_mask_16_3, %r13_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,0)
                %idx_20_0 = arith.addi %out_base_32_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_0], %out_mask_32_0, %r20_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_20_1 = arith.addi %out_base_32_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_1], %out_mask_32_1, %r20_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_20_2 = arith.addi %out_base_32_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_2], %out_mask_32_2, %r20_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_20_3 = arith.addi %out_base_32_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_20_3], %out_mask_32_3, %r20_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,1)
                %idx_21_0 = arith.addi %out_base_32_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_0], %out_mask_32_0, %r21_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_21_1 = arith.addi %out_base_32_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_1], %out_mask_32_1, %r21_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_21_2 = arith.addi %out_base_32_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_2], %out_mask_32_2, %r21_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_21_3 = arith.addi %out_base_32_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_21_3], %out_mask_32_3, %r21_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,2)
                %idx_22_0 = arith.addi %out_base_32_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_0], %out_mask_32_0, %r22_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_22_1 = arith.addi %out_base_32_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_1], %out_mask_32_1, %r22_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_22_2 = arith.addi %out_base_32_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_2], %out_mask_32_2, %r22_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_22_3 = arith.addi %out_base_32_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_22_3], %out_mask_32_3, %r22_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (2,3)
                %idx_23_0 = arith.addi %out_base_32_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_0], %out_mask_32_0, %r23_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_23_1 = arith.addi %out_base_32_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_1], %out_mask_32_1, %r23_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_23_2 = arith.addi %out_base_32_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_2], %out_mask_32_2, %r23_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_23_3 = arith.addi %out_base_32_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_23_3], %out_mask_32_3, %r23_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,0)
                %idx_30_0 = arith.addi %out_base_48_0, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_0], %out_mask_48_0, %r30_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_30_1 = arith.addi %out_base_48_1, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_1], %out_mask_48_1, %r30_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_30_2 = arith.addi %out_base_48_2, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_2], %out_mask_48_2, %r30_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_30_3 = arith.addi %out_base_48_3, %out_col_0 : index
                vector.maskedstore %c_flat[%idx_30_3], %out_mask_48_3, %r30_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,1)
                %idx_31_0 = arith.addi %out_base_48_0, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_0], %out_mask_48_0, %r31_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_31_1 = arith.addi %out_base_48_1, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_1], %out_mask_48_1, %r31_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_31_2 = arith.addi %out_base_48_2, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_2], %out_mask_48_2, %r31_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_31_3 = arith.addi %out_base_48_3, %out_col_1 : index
                vector.maskedstore %c_flat[%idx_31_3], %out_mask_48_3, %r31_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,2)
                %idx_32_0 = arith.addi %out_base_48_0, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_0], %out_mask_48_0, %r32_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_32_1 = arith.addi %out_base_48_1, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_1], %out_mask_48_1, %r32_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_32_2 = arith.addi %out_base_48_2, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_2], %out_mask_48_2, %r32_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_32_3 = arith.addi %out_base_48_3, %out_col_2 : index
                vector.maskedstore %c_flat[%idx_32_3], %out_mask_48_3, %r32_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>

                // Tile (3,3)
                %idx_33_0 = arith.addi %out_base_48_0, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_0], %out_mask_48_0, %r33_0_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_33_1 = arith.addi %out_base_48_1, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_1], %out_mask_48_1, %r33_1_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_33_2 = arith.addi %out_base_48_2, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_2], %out_mask_48_2, %r33_2_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                %idx_33_3 = arith.addi %out_base_48_3, %out_col_3 : index
                vector.maskedstore %c_flat[%idx_33_3], %out_mask_48_3, %r33_3_f16 : memref<16896xf16>, vector<1xi1>, vector<1xf16>
                }
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
            // %a_ptr: memref<33x128xf16>,
            // %b_ptr: memref<8x256x128xf16>,
            // %sorted_token_ids_ptr: memref<633xi32>,
            // %expert_ids_ptr: memref<10xi32>,
            // %num_tokens_post_padded_ptr: memref<1xi32>,
            // %c_ptr: memref<33x2x256xf16>
            %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<33x128xf16>
            %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x256x128xf16>
            %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<633xi32>
            %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<10xi32>
            %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
            %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<33x2x256xf16>
            %6 = flow.dispatch @fused_moe_kernel::@fused_moe_kernel(%0, %1, %2, %3, %4, %5) : (tensor<33x128xf16>, tensor<8x256x128xf16>, tensor<633xi32>, tensor<10xi32>, tensor<1xi32>, tensor<33x2x256xf16>) -> %5
            %7 = hal.tensor.barrier join(%6 : tensor<33x2x256xf16>) => %arg7 : !hal.fence
            %8 = hal.tensor.export %7 : tensor<33x2x256xf16> -> !hal.buffer_view
            return %8 : !hal.buffer_view
        }
        }
            """
    )


# --- symbolic sizes ---
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    E = tkl.sym.E
    EM = tkl.sym.EM
    TOPK = tkl.sym.TOPK
    MAX_M_BLOCKS = tkl.sym.MAX_M_BLOCKS

    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    NUM_TOKENS_BUF_SIZE = tkl.sym.NUM_TOKENS_BUF_SIZE

    # --- constraints / layout (match your working setup) ---
    constraints: list[tkw.Constraint] = [
        # 1D grid over EM*N in tiles of BLOCK_M*BLOCK_N (like your standalone)
        tkw.WorkgroupConstraint(EM * N, BLOCK_M * BLOCK_N, 0),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            vector_shapes={M: 0, TOPK: 0, N: 32},
        ),
    ]

    @tkw.wave(constraints)
    def fused_moe_kernel(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],                  # (33,128)
        b: tkl.Memory[E, N, K, ADDRESS_SPACE, tkl.f16],               # (8,256,128)
        sorted_token_ids: tkl.Memory[EM, ADDRESS_SPACE, tkl.i32],     # (633,)
        expert_ids: tkl.Memory[MAX_M_BLOCKS, ADDRESS_SPACE, tkl.i32], # (10,)
        num_tokens_post_padded: tkl.Memory[NUM_TOKENS_BUF_SIZE, ADDRESS_SPACE, tkl.i32],  # (1,)
        c: tkl.Memory[M, TOPK, N, GLOBAL_ADDRESS_SPACE, tkl.f16],     # (33,2,256)
    ):
        # Body is irrelevant when override_mlir is supplied; keep a no-op write to satisfy tracing.
        c_reg = tkl.Register[M, TOPK, N, tkl.f16](0.0)
        tkw.write(c_reg, c)

    # --- hard-coded hyperparams for this exact config ---
    #EM_VAL = 633
    EM_VAL= max_num_tokens_padded
    #MAX_M_BLOCKS_VAL = math.ceil(EM_VAL / 64)  # 10
    
    MAX_M_BLOCKS_VAL = -(EM_VAL // -64)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        TOPK: 2,
        E: 8,
        K: k,
        N: n,
        EM: EM_VAL,
        MAX_M_BLOCKS: MAX_M_BLOCKS_VAL,
        NUM_TOKENS_BUF_SIZE: 1,
    }
    hyperparams.update(get_default_scheduling_params())


    # --- compile with your override MLIR ---
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma,               # <— you provide the MLIR blob
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,                         # matches your MLIR path
        use_scheduling_barriers=enable_scheduling_barriers,
        print_mlir=False,
    )
    options = set_default_run_config(options)

    return wave_compile(options, fused_moe_kernel)

# --------- Triton kernel: MoE GEMM (fp16/bf16, no quant/bias) ---------
@triton.jit
def _moe_gemm_kernel_triton(
    a_ptr,              # (M, K)
    b_ptr,              # (E, N, K)
    c2d_ptr,            # (M*topk, N)  -- 2D view of C to match offs_token indexing
    sorted_token_ids_ptr,   # (EM,)
    expert_ids_ptr,         # (num_blocks_m,)
    num_tokens_post_padded_ptr,  # (1,)

    # dimensions
    N, K, EM, num_valid_tokens,

    # strides
    stride_am, stride_ak,     # A
    stride_be, stride_bk, stride_bn,  # B  (E,N,K) with layout E-major
    stride_c2dm, stride_c2dn, # C2D  (M*topk, N)

    # meta
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,     # tl.float32 (accumulator)
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # early-out if padded rows
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # per-block token slice
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # expert for this M-block
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    # if -1, nothing to compute (not in this expert-parallel shard)
    if off_expert == -1:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c2d_ptr + offs_token[:, None] * stride_c2dm + offs_n[None, :] * stride_c2dn
        c_mask = token_mask[:, None] & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)
        return

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A pointers (flattened token pairs // top_k → original token)
    a_ptrs = a_ptr + ( (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak )

    # B pointers: expert-major, NxK per expert
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + (offs_k[:, None] * stride_bk + (offs_n[None, :] % N) * stride_bn)
    )

    # fp32 accumulation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)

    # K loop
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for kt in range(0, num_k_tiles):
        # for last tile, mask loads if K not a multiple of BLOCK_SIZE_K
        k_left = K - kt * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_left), other=0)
        b = tl.load(b_ptrs,  mask=(offs_k[:, None] < k_left), other=0)
        acc += tl.dot(a, b)  # fp16/bf16→fp32

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # write-back into (M*topk, N) view using offs_token as row index
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c2d_ptr + offs_token[:, None] * stride_c2dm + offs_n[None, :] * stride_c2dn
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# --------- Python launcher that uses your test’s tensors ---------
def moe_gemm_triton(a, b, c, sorted_ids, expert_ids, num_tokens_post_padded, topk,
                    block_m=64, block_n=64, block_k=32, group_m=8):
    """
    a: (M, K)          dtype: fp16/bf16
    b: (E, N, K)       dtype: fp16/bf16 (same as a)
    c: (M, topk, N)    dtype: same as a (or fp16/bf16); accumulator is fp32
    sorted_ids: (EM,)  int32/int64
    expert_ids: (ceil(EM/block_m),) int32/int64  (one expert id per M-block)
    num_tokens_post_padded: (1,) int32/int64
    """
    assert a.device.type == "cuda" and b.device.type == "cuda" and c.device.type == "cuda"
    M, K = a.shape
    E, N, Kb = b.shape
    assert K == Kb, "B last dim must equal K"
    assert c.shape == (M, topk, N)
    EM = sorted_ids.numel()
    compute_type = tl.float32

    # convenience: make 2D view of c as (M*topk, N) to match offs_token addressing
    c2d = c.view(M * topk, N)

    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, block_n),)

    _moe_gemm_kernel_triton[grid](
        a, b, c2d,
        sorted_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, M * topk,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(2), b.stride(1),
        c2d.stride(0), c2d.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        top_k=topk,
        compute_type=compute_type,
        num_warps=4,      # good defaults for 64x64x32 tiles on RDNA/CDNA
        num_stages=2,
    )




def moe_gemm_pytorch(
    a,  # Input tokens: [M, K]
    b,  # Expert weights: [E, N, K]
    c,  # Output: [M, topk, N]
    sorted_token_ids,  # Sorted token-expert pair indices: [EM] (padded)
    expert_ids,  # Expert ID for each block: [num_blocks]
    num_tokens_post_padded,  # Total padded length: [1]
    top_k,  # Number of experts per token
    block_size_m=64,
    block_size_n=64,
    block_size_k=64,
):
    """
    PyTorch equivalent of the Triton fused MoE kernel.

    Args:
        a: Input token embeddings [M, K]
        b: Expert weight matrices [E, N, K]
        sorted_token_ids: Token-expert pair indices sorted by expert [EM] (padded)
        expert_ids: Expert ID for each block [num_blocks]
        num_tokens_post_padded: Total padded length [1]
        top_k: Number of experts each token is routed to

    Returns:
        c: Output tensor [M, topk, N]
    """
    M, K = a.shape
    E, N, _ = b.shape
    EM = sorted_token_ids.shape[0]
    num_valid_tokens = M * top_k

    # Process tokens in blocks
    num_blocks = (EM + block_size_m - 1) // block_size_m

    for i, idx in enumerate(sorted_token_ids.tolist()):
        if i % block_size_m == 0:
            block_token_ids = sorted_token_ids[i : i + block_size_m]
            orig_token_ids = torch.clamp(block_token_ids // top_k, 0, M - 1)
            valid_mask = block_token_ids < num_valid_tokens
    for block_idx in range(num_blocks):
        # Determine token range for this block
        start_idx = block_idx * block_size_m
        end_idx = min(start_idx + block_size_m, EM)

        # Skip if we're past valid tokens
        if start_idx >= num_tokens_post_padded.item():
            continue

        # Get token-expert pair indices for this block
        block_token_ids = sorted_token_ids[start_idx:end_idx]

        # Create mask for valid token-expert pairs
        valid_mask = block_token_ids < num_valid_tokens

        if not valid_mask.any():
            continue

        # Get the expert ID for this block
        expert_id = expert_ids[block_idx].item()

        # Initialize accumulator for this block
        accumulator = torch.zeros(block_size_m, N, dtype=a.dtype, device=a.device)

        # Process K dimension in chunks (simulating the K-loop in Triton)
        for k_start in range(0, K, block_size_k):
            k_end = min(k_start + block_size_k, K)
            actual_k_size = k_end - k_start

            # Load block from A
            # Map token-expert pair indices to original token indices
            orig_token_ids = torch.clamp(block_token_ids // top_k, 0, M - 1)
            block_a = a[orig_token_ids, k_start:k_end]  # [block_size_m, actual_k_size]

            # Apply valid mask to A
            block_a = block_a * valid_mask.to(a.dtype).unsqueeze(1)

            # Load block from B (expert weights)
            # Process N dimension in chunks
            for n_start in range(0, N, block_size_n):
                n_end = min(n_start + block_size_n, N)
                actual_n_size = n_end - n_start

                # Get expert weights: B[expert_id, n_start:n_end, k_start:k_end]
                # Need to transpose to [k, n] for matrix multiplication
                block_b = b[
                    expert_id, n_start:n_end, k_start:k_end
                ].t()  # [actual_k_size, actual_n_size]

                # Compute matrix multiplication: [block_size_m, k] @ [k, n] = [block_size_m, n]
                partial_result = torch.matmul(
                    block_a, block_b
                )  # [block_size_m, actual_n_size]

                # Accumulate in the correct position
                accumulator[:, n_start:n_end] += partial_result

        # Write back to output tensor using the stride-based mapping
        # This is the key: sorted_token_ids encodes the mapping to 3D positions
        for i, token_id in enumerate(block_token_ids):
            if token_id >= num_valid_tokens:
                continue

            # Decode the 3D position from the flat index
            # Which original token (0 to M-1)
            orig_token = token_id // top_k
            # Which expert slot for that token (0 to top_k-1)
            expert_slot = token_id % top_k

            c[orig_token, expert_slot] = accumulator[i]



    
@torch.no_grad()
def build_sorted_ids_and_expert_blocks(topk_ids: torch.Tensor, num_experts: int, block_m: int):
    """
    Given topk_ids: (M, topk) of expert indices, build:
      - sorted_ids: (EM_padded,) int32 — token*topk+slot, grouped by expert, padded per-expert to multiples of block_m
      - expert_ids: (num_m_blocks,) int32 — expert id for each M-block (64 rows), -1 for host-padding blocks
      - num_tokens_post_padded: (1,) int32 — total length of sorted_ids after padding
    """
    device = topk_ids.device
    M, topk = topk_ids.shape
    torch.manual_seed(0)
    
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_m - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = -(max_num_tokens_padded // -block_m)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    fuse_sorted_ids_padding = sorted_ids.shape[0] <= 4096
    if not fuse_sorted_ids_padding:
        sorted_ids.fill_(topk_ids.numel())
    
    # Populate using the same routine as the test harness
    moe_align_block_size_pytorch(
        topk_ids, num_experts, block_m, sorted_ids, expert_ids, num_tokens_post_pad
    )
    print("expert ids",expert_ids)
    print("num_tokenspost padded", num_tokens_post_pad)
    print("sorted ids",sorted_ids.shape)
    return sorted_ids, expert_ids, num_tokens_post_pad, max_num_tokens_padded
    

def show_outputs(c_ref, c_tri, full=False, rows=4, cols=16):
    print("\n=== Outputs ===")
    print(f"c_ref   shape: {tuple(c_ref.shape)}  dtype: {c_ref.dtype}")
    print(f"c_triton shape: {tuple(c_tri.shape)} dtype: {c_tri.dtype}")

    # numeric check
    diff = (c_tri.float() - c_ref.float()).abs()
    print(f"max|diff|: {diff.max().item():.6e}, mean|diff|: {diff.mean().item():.6e}")

    if full:
        # print the whole tensors (can be huge)
        torch.set_printoptions(sci_mode=False, linewidth=200, precision=4)
        print("\n--- c_ref ---")
        print(c_ref)
        print("\n--- c_triton ---")
        print(c_tri)
    else:
        # print a focused slice so it’s readable
        torch.set_printoptions(sci_mode=False, linewidth=200, precision=4)
        M, topk, N = c_ref.shape
        r = min(rows, M)
        k = min(2, topk)
        c = min(cols, N)

        print(f"\nShowing slice: [:{r}, :{k}, :{c}]")
        print("\n--- c_ref (slice) ---")
        print(c_ref[:r, :k, :c])
        print("\n--- c_triton (slice) ---")
        print(c_tri[:r, :k, :c])

# After computing c_ref and c_tri:





def bench(M, N, K, provider):
    """
    M: number of tokens (== num_tokens)
    N: output features
    K: input features
    provider: "triton" or "torch"
    """
    # --- fixed test knobs (match your pytest-ish setup) ---
    topk = 2
    num_experts = 8
    block_m = 64       # Triton block size for M; also routing pad
    block_n = 64       # Triton block size for N
    block_k = 32       # Triton block size for K
    dtype = torch.float16

    # --- inputs ---
    assert torch.cuda.is_available(), "No HIP/CUDA device visible"
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Router -> topk expert ids
    scores = torch.rand(M, num_experts, device=device)
    _, topk_ids = torch.topk(scores, k=topk, dim=1)          # (M, topk), values in [0, E-1]

    # Build routing structures
    sorted_ids, expert_ids, num_tokens_post_padded, max_num_tokens_padded = build_sorted_ids_and_expert_blocks(
        topk_ids, num_experts=num_experts, block_m=block_m
    )

    # print("max_num_tokens_padded",max_num_tokens_padded)
    # print("num_tokens_post_padded",num_tokens_post_padded)
    
    # Model tensors
    a = torch.randn(M, K, dtype=dtype, device=device)        # tokens
    b = torch.randn(num_experts, N, K, dtype=dtype, device=device)  

    # Outputs
    c_wave = torch.empty(M, topk, N, dtype=dtype, device="cuda")
    c_ref = torch.empty(M, topk, N, dtype=dtype, device="cuda")
    c_tri = torch.empty(M, topk, N, dtype=dtype, device="cuda")
    
    quantiles = [0.5, 0.2, 0.8]
 
    if provider == "triton":
        # warmup
        moe_gemm_triton(
            a, b, c_tri,
            sorted_ids, expert_ids, num_tokens_post_padded,
            topk,
            block_m=block_m, block_n=block_n, block_k=block_k, group_m=8
        )
        #save_triton_ir(_moe_gemm_kernel, basename="moe_gemm")  
        # bench
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: moe_gemm_triton(
                a, b, c_tri,
                sorted_ids, expert_ids, num_tokens_post_padded,
                topk,
                block_m=block_m, block_n=block_n, block_k=block_k, group_m=8
            ),
            quantiles=quantiles,
        )

    elif provider == "torch":
        # warmup
        moe_gemm_pytorch(
            a, b, c_ref,
            sorted_ids, expert_ids, num_tokens_post_padded,
            topk,
            block_size_m=block_m, block_size_n=block_n, block_size_k=block_k
        )
        # bench
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: moe_gemm_pytorch(
                a, b, c_ref,
                sorted_ids, expert_ids, num_tokens_post_padded,
                topk,
                block_size_m=block_m, block_size_n=block_n, block_size_k=block_k
            ),
            quantiles=quantiles,
        )
    elif provider =="wave":
        
        #wave_kernel = build_wave_moe_gemm_fixed_fp16_33x2_E8_K128_N256()
        wave_kernel= build_wave_moe_gemm_fixed_1024_256_8_64_2_2048(max_num_tokens_padded,M,N,K)
        wave_kernel(a, b, sorted_ids, expert_ids, num_tokens_post_padded, c_wave)

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda:         wave_kernel(a, b, sorted_ids, expert_ids, num_tokens_post_padded, c_wave),
            quantiles=quantiles,
        )

    else:
        raise ValueError(f"unknown provider: {provider}")

    return ms, min_ms, max_ms


def compare_once(M, N, K):
    topk = 2
    num_experts = 8
    block_m, block_n, block_k = 64, 64, 32
    dtype = torch.float16
    device = torch.device("cuda")

    scores = torch.rand(M, num_experts, device=device)
    _, topk_ids = torch.topk(scores, k=topk, dim=1)

    sorted_ids, expert_ids, num_tokens_post_padded, max_num_tokens_padded = build_sorted_ids_and_expert_blocks(
        topk_ids, num_experts, block_m
    )

    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(num_experts, N, K, dtype=dtype, device=device)
    # c_ref = torch.zeros(M, topk, N, dtype=dtype, device=device)
    # c_tri = torch.zeros_like(c_ref)
    # c_wave= torch.zeros_like(c_ref)
    c_wave = torch.empty(M, topk, N, dtype=dtype, device="cuda")
    c_ref = torch.empty(M, topk, N, dtype=dtype, device="cuda")
    c_tri = torch.empty(M, topk, N, dtype=dtype, device="cuda")



    #moe_gemm_pytorch(a, b, c_ref, sorted_ids, expert_ids, num_tokens_post_padded, topk,block_size_m=block_m, block_size_n=block_n, block_size_k=block_k)
    moe_gemm_triton(a, b, c_tri, sorted_ids, expert_ids, num_tokens_post_padded, topk,
                    block_m=block_m, block_n=block_n, block_k=block_k, group_m=8)
    
    wave_kernel= build_wave_moe_gemm_fixed_1024_256_8_64_2_2048(max_num_tokens_padded,M,N,K)
    #wave_kernel=build_wave_moe_gemm_fixed_fp16_33x2_E8_K128_N256()
    wave_kernel(a, b, sorted_ids, expert_ids, num_tokens_post_padded, c_wave)
    
    rtol, atol = 1e-1, 1e-2


    # print slice + diffs
    def show_outputs(ref, tri,wave, rows=4, cols=16):
        torch.set_printoptions(sci_mode=False, linewidth=200, precision=4)
        print("\n=== sanity slice [0:%d, 0:2, 0:%d] ===" % (rows, cols))
        print("--- c_ref ---")
        #print(ref[:rows, :2, :cols])
        print("--- c_triton ---")
        print(tri[:rows, :2, :cols])
        print("--- wave ---")
        print(wave[:rows, :2, :cols])
        diff = (tri.float() - ref.float()).abs()
        print("max|diff|:", diff.max().item(), "mean|diff|:", diff.mean().item())

    show_outputs(c_ref, c_tri, c_wave)
    
    # torch.testing.assert_close(c_tri, c_wave, rtol=rtol, atol=atol)
    # torch.testing.assert_close(c_tri, c_ref, rtol=rtol, atol=atol)
    
    

if __name__ == "__main__":
    import torch

    # Reproducibility + device sanity
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "No HIP/CUDA device visible"

    # Match your test case
    #M, N, K = 33, 256, 128
    
    
    #M,N,K= 2048,1024,256
    M,N,K= 16384,32768,6144


    #compare_once(M, N, K)

    #2) Time both implementations with the same inputs
    for provider in ["triton"]:
        ms, min_ms, max_ms = bench(M, N, K, provider)
        print(f"[{provider}]  median: {ms:.3f} ms   best: {min_ms:.3f} ms   worst: {max_ms:.3f} ms")