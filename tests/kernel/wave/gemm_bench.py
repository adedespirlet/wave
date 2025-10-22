import triton
import triton.language as tl
import torch
import itertools

import triton.compiler as tc

from torch.nn import functional as F
import wave_lang.kernel as tk
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.templates.reordered_gemm import get_reordered_matmul

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


def triton_matmul_abt(A: torch.Tensor, B: torch.Tensor, BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=8):
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


def get_wave_gemm_pipelined_32x32x16(shape: tuple[int, int, int],dtype: torch.dtype, dynamic_dims: bool | tuple[bool, bool, bool],mfma_variant: MMAType):
    
    pipelined_version_32x32x16="""
    #map = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 4) mod 64)>
    #map1 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>
    #map2 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 4 - ((s1 * 32 + s0 floordiv 4) floordiv 64) * 64)>
    #map3 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 32) * 32)>
    #map4 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8)>
    #map5 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 16)>
    #map6 = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
    #map7 = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 8 - (s1 floordiv 4) * 32)>
    #map8 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
    #map9 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 32 - (s0 floordiv 32) * 32)>
    #map10 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
    #map11 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
    #map12 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
    #map13 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
    #map14 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
    #map15 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
    #map16 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
    #map17 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
    #map18 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
    #map19 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
    #map20 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
    #map21 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
    #map22 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
    #map23 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
    #map24 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
    #prefetch_current = affine_map<()[s1] -> (0 * 32 + s1 * 8 - (s1 floordiv 4) * 32)>
    #prefetch_next = affine_map<()[s1] -> (1 * 32 + s1 * 8 - (s1 floordiv 4) * 32)>
    #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>
    module attributes {transform.with_named_sequence} {
    stream.executable private @gemm {
        stream.executable.export public @gemm workgroups() -> (index, index, index) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        stream.return %c1, %c2, %c1 : index, index, index
        }
        builtin.module {
        func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
            %cst = arith.constant dense<0.000000e+00> : vector<8xbf16>
            %cst_0 = arith.constant dense<511> : vector<8xindex>
            %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c16 = arith.constant 16 : index
            %c4608 = arith.constant 4608 : index
            %c0 = arith.constant 0 : index
            %pipeline_depth= arith.constant 2 : index
            %cst_2 = arith.constant dense<0.000000e+00> : vector<16xf32>
            %block_id_y = gpu.block_id  y upper_bound 2
            %thread_id_x = gpu.thread_id  x upper_bound 128
            %thread_id_y = gpu.thread_id  y upper_bound 2
            %alloc = memref.alloc() : memref<9216xi8, #gpu.address_space<workgroup>>
            %view = memref.view %alloc[%c0][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xbf16, #gpu.address_space<workgroup>>
            %view_3 = memref.view %alloc[%c4608][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xbf16, #gpu.address_space<workgroup>>
            %0 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<128x511xbf16, strided<[511, 1], offset: ?>>
            %1 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<64x511xbf16, strided<[511, 1], offset: ?>>
            %2 = affine.apply #map()[%thread_id_x, %thread_id_y]
            %3 = affine.apply #map1()[%thread_id_x]
            %4 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y]
            %5 = affine.apply #map3()[%thread_id_x, %thread_id_y]
            %6 = affine.apply #map4()[%thread_id_x]
            %7 = affine.apply #map5()[%thread_id_x]
            %8 = affine.apply #map6()[%thread_id_x]    
            // pre-load 
            %next = affine.apply #prefetch_current()[%thread_id_x]
            %current = affine.apply #prefetch_next()[%thread_id_x]
            //normally %39 depends on arg3 loop It K , replace it with 0 and 1 for two first it
            %prefetchA_0 = vector.load %1[%2, %current] : memref<64x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
            %prefetchB_0 = vector.load %0[%4, %current] : memref<128x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
            %prefetchA_1 = vector.load %1[%2, %next] : memref<64x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
            %prefetchB_1 = vector.load %0[%4, %next] : memref<128x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
            %9:5 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %cst_2, %prefetchAcurrent=%prefetchA_0 , %prefetchBcurrent= %prefetchB_0 , %prefetchAnext=%prefetchA_1 , %prefetchBnext= %prefetchB_1) -> (vector<16xf32>,vector<8xbf16>,vector<8xbf16>,vector<8xbf16>,vector<8xbf16>) {
                %K_iter= arith.addi %arg3, %pipeline_depth : index
                %44 = affine.apply #map7()[%K_iter, %thread_id_x]
                %45 = vector.broadcast %44 : index to vector<8xindex>
                %46 = arith.addi %45, %cst_1 overflow<nsw, nuw> : vector<8xindex>
                %47 = arith.cmpi slt, %46, %cst_0 : vector<8xindex>
                %48 = vector.maskedload %1[%2, %44], %47, %cst : memref<64x511xbf16, strided<[511, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
                %49 = vector.maskedload %0[%4, %44], %47, %cst : memref<128x511xbf16, strided<[511, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
                vector.store %prefetchAcurrent, %view_3[%2, %3] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %prefetchBcurrent, %view[%2, %3] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                amdgpu.lds_barrier
                %50 = vector.load %view[%5, %6] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %51 = vector.load %view[%5, %7] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %52 = vector.load %view_3[%8, %6] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %53 = vector.load %view_3[%8, %7] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %56 = amdgpu.mfma %52 * %50 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %62 = amdgpu.mfma %53 * %51 + %56 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                scf.yield %62 ,%prefetchAnext, %prefetchBnext, %48 , %49: vector<16xf32>,vector<8xbf16>,vector<8xbf16>,vector<8xbf16>,vector<8xbf16>
            }
            %10 = vector.extract_strided_slice %9#0 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %11 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<64x128xf32, strided<[128, 1], offset: ?>>
            %12 = affine.apply #map8()[%thread_id_x]
            %13 = affine.apply #map9()[%thread_id_x, %block_id_y, %thread_id_y]
            vector.store %10, %11[%12, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %14 = vector.extract_strided_slice %9#0 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %15 = affine.apply #map10()[%thread_id_x]
            vector.store %14, %11[%15, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %16 = vector.extract_strided_slice %9#0 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %17 = affine.apply #map11()[%thread_id_x]
            vector.store %16, %11[%17, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %18 = vector.extract_strided_slice %9#0 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %19 = affine.apply #map12()[%thread_id_x]
            vector.store %18, %11[%19, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %20 = vector.extract_strided_slice %9#0 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %21 = affine.apply #map13()[%thread_id_x]
            vector.store %20, %11[%21, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %22 = vector.extract_strided_slice %9#0 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %23 = affine.apply #map14()[%thread_id_x]
            vector.store %22, %11[%23, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %24 = vector.extract_strided_slice %9#0 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %25 = affine.apply #map15()[%thread_id_x]
            vector.store %24, %11[%25, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %26 = vector.extract_strided_slice %9#0 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %27 = affine.apply #map16()[%thread_id_x]
            vector.store %26, %11[%27, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %28 = vector.extract_strided_slice %9#0 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %29 = affine.apply #map17()[%thread_id_x]
            vector.store %28, %11[%29, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %30 = vector.extract_strided_slice %9#0 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %31 = affine.apply #map18()[%thread_id_x]
            vector.store %30, %11[%31, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %32 = vector.extract_strided_slice %9#0 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %33 = affine.apply #map19()[%thread_id_x]
            vector.store %32, %11[%33, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %34 = vector.extract_strided_slice %9#0 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %35 = affine.apply #map20()[%thread_id_x]
            vector.store %34, %11[%35, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %36 = vector.extract_strided_slice %9#0 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %37 = affine.apply #map21()[%thread_id_x]
            vector.store %36, %11[%37, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %38 = vector.extract_strided_slice %9#0 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %39 = affine.apply #map22()[%thread_id_x]
            vector.store %38, %11[%39, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %40 = vector.extract_strided_slice %9#0 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %41 = affine.apply #map23()[%thread_id_x]
            vector.store %40, %11[%41, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %42 = vector.extract_strided_slice %9#0 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %43 = affine.apply #map24()[%thread_id_x]
            vector.store %42, %11[%43, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            return
        }
        }
    }
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<64x511xbf16>
        %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<128x511xbf16>
        %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<64x128xf32>
        %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<64x511xbf16>, tensor<128x511xbf16>, tensor<64x128xf32>) -> %2
        %4 = hal.tensor.barrier join(%3 : tensor<64x128xf32>) => %arg4 : !hal.fence
        %5 = hal.tensor.export %4 : tensor<64x128xf32> -> !hal.buffer_view
        return %5 : !hal.buffer_view
    }
    }
    """
    
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
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        override_mlir=pipelined_version_32x32x16,
        wave_runtime=True,
        use_scheduling_barriers=enable_scheduling_barriers,
    )
    
    options = set_default_run_config(options)
    return wave_compile(options, gemm)

def get_wave_gemm_pipelined_harsh(shape: tuple[int, int, int],dtype: torch.dtype, dynamic_dims: bool | tuple[bool, bool, bool],mfma_variant: MMAType):
    harshversion="""
    #map = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 4) mod 64)>
    #map1 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>
    #map2 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 4 - ((s1 * 32 + s0 floordiv 4) floordiv 64) * 64)>
    #map3 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 32) * 32)>
    #map4 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8)>
    #map5 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 16)>
    #map6 = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
    #map7 = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 8 - (s1 floordiv 4) * 32)>
    #map8 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
    #map9 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 32 - (s0 floordiv 32) * 32)>
    #map10 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
    #map11 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
    #map12 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
    #map13 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
    #map14 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
    #map15 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
    #map16 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
    #map17 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
    #map18 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
    #map19 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
    #map20 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
    #map21 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
    #map22 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
    #map23 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
    #map24 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
    #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64, {llvm_func_attrs = {"denormal-fp-math-f32" = "preserve-sign"}}>
    module attributes {transform.with_named_sequence} {
    stream.executable private @gemm {
        stream.executable.export public @gemm workgroups() -> (index, index, index) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        stream.return %c1, %c2, %c1 : index, index, index
        }
        builtin.module {
        func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
            %cst = arith.constant dense<0.000000e+00> : vector<8xbf16>
            %cst_0 = arith.constant dense<511> : vector<8xindex>
            %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
            %c1 = arith.constant 1 : index
            %c16 = arith.constant 16 : index
            %c4608 = arith.constant 4608 : index
            %c0 = arith.constant 0 : index
            %cst_2 = arith.constant dense<0.000000e+00> : vector<16xf32>
            %block_id_y = gpu.block_id  y upper_bound 2
            %thread_id_x = gpu.thread_id  x upper_bound 128
            %thread_id_y = gpu.thread_id  y upper_bound 2
            %alloc = memref.alloc() : memref<9216xi8, #gpu.address_space<workgroup>>
            %view = memref.view %alloc[%c0][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xbf16, #gpu.address_space<workgroup>>
            %view_3 = memref.view %alloc[%c4608][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xbf16, #gpu.address_space<workgroup>>
            %0 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<128x511xbf16, strided<[511, 1], offset: ?>>
            %1 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<64x511xbf16, strided<[511, 1], offset: ?>>
            %2 = affine.apply #map()[%thread_id_x, %thread_id_y]
            %3 = affine.apply #map1()[%thread_id_x]
            %4 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y]
            %5 = affine.apply #map3()[%thread_id_x, %thread_id_y]
            %6 = affine.apply #map4()[%thread_id_x]
            %7 = affine.apply #map5()[%thread_id_x]
            %8 = affine.apply #map6()[%thread_id_x]
            %9 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %cst_2) -> (vector<16xf32>) {
            %44 = affine.apply #map7()[%arg3, %thread_id_x]
            %45 = vector.broadcast %44 : index to vector<8xindex>
            %46 = arith.addi %45, %cst_1 overflow<nsw, nuw> : vector<8xindex>
            %47 = arith.cmpi slt, %46, %cst_0 : vector<8xindex>
            %48 = vector.maskedload %1[%2, %44], %47, %cst : memref<64x511xbf16, strided<[511, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
            amdgpu.lds_barrier
            vector.store %48, %view_3[%2, %3] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
            %49 = vector.maskedload %0[%4, %44], %47, %cst : memref<128x511xbf16, strided<[511, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
            vector.store %49, %view[%2, %3] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
            amdgpu.lds_barrier
            %50 = vector.load %view[%5, %6] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
            %51 = vector.load %view[%5, %7] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
            %52 = vector.load %view_3[%8, %6] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
            %53 = vector.load %view_3[%8, %7] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
            %54 = vector.extract_strided_slice %52 {offsets = [0], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
            %55 = vector.extract_strided_slice %50 {offsets = [0], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
            %56 = amdgpu.mfma %54 * %55 + %arg4 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<16xf32>
            %57 = vector.extract_strided_slice %52 {offsets = [4], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
            %58 = vector.extract_strided_slice %50 {offsets = [4], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
            %59 = amdgpu.mfma %57 * %58 + %56 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<16xf32>
            %60 = vector.extract_strided_slice %53 {offsets = [0], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
            %61 = vector.extract_strided_slice %51 {offsets = [0], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
            %62 = amdgpu.mfma %60 * %61 + %59 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<16xf32>
            %63 = vector.extract_strided_slice %53 {offsets = [4], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
            %64 = vector.extract_strided_slice %51 {offsets = [4], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
            %65 = amdgpu.mfma %63 * %64 + %62 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<16xf32>
            scf.yield %65 : vector<16xf32>
            }
            %10 = vector.extract_strided_slice %9 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %11 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<64x128xf32, strided<[128, 1], offset: ?>>
            %12 = affine.apply #map8()[%thread_id_x]
            %13 = affine.apply #map9()[%thread_id_x, %block_id_y, %thread_id_y]
            vector.store %10, %11[%12, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %14 = vector.extract_strided_slice %9 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %15 = affine.apply #map10()[%thread_id_x]
            vector.store %14, %11[%15, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %16 = vector.extract_strided_slice %9 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %17 = affine.apply #map11()[%thread_id_x]
            vector.store %16, %11[%17, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %18 = vector.extract_strided_slice %9 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %19 = affine.apply #map12()[%thread_id_x]
            vector.store %18, %11[%19, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %20 = vector.extract_strided_slice %9 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %21 = affine.apply #map13()[%thread_id_x]
            vector.store %20, %11[%21, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %22 = vector.extract_strided_slice %9 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %23 = affine.apply #map14()[%thread_id_x]
            vector.store %22, %11[%23, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %24 = vector.extract_strided_slice %9 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %25 = affine.apply #map15()[%thread_id_x]
            vector.store %24, %11[%25, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %26 = vector.extract_strided_slice %9 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %27 = affine.apply #map16()[%thread_id_x]
            vector.store %26, %11[%27, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %28 = vector.extract_strided_slice %9 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %29 = affine.apply #map17()[%thread_id_x]
            vector.store %28, %11[%29, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %30 = vector.extract_strided_slice %9 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %31 = affine.apply #map18()[%thread_id_x]
            vector.store %30, %11[%31, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %32 = vector.extract_strided_slice %9 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %33 = affine.apply #map19()[%thread_id_x]
            vector.store %32, %11[%33, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %34 = vector.extract_strided_slice %9 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %35 = affine.apply #map20()[%thread_id_x]
            vector.store %34, %11[%35, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %36 = vector.extract_strided_slice %9 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %37 = affine.apply #map21()[%thread_id_x]
            vector.store %36, %11[%37, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %38 = vector.extract_strided_slice %9 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %39 = affine.apply #map22()[%thread_id_x]
            vector.store %38, %11[%39, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %40 = vector.extract_strided_slice %9 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %41 = affine.apply #map23()[%thread_id_x]
            vector.store %40, %11[%41, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %42 = vector.extract_strided_slice %9 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %43 = affine.apply #map24()[%thread_id_x]
            vector.store %42, %11[%43, %13] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            return
        }
        }
    }
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<64x511xbf16>
        %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<128x511xbf16>
        %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<64x128xf32>
        %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<64x511xbf16>, tensor<128x511xbf16>, tensor<64x128xf32>) -> %2
        %4 = hal.tensor.barrier join(%3 : tensor<64x128xf32>) => %arg4 : !hal.fence
        %5 = hal.tensor.export %4 : tensor<64x128xf32> -> !hal.buffer_view
        return %5 : !hal.buffer_view
    }
    }
    """    
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
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        override_mlir=harshversion,
        wave_runtime=True,
        use_scheduling_barriers=enable_scheduling_barriers,
    )
    
    options = set_default_run_config(options)
    return wave_compile(options, gemm)

def get_wave_gemm_pipelined_16x16(shape: tuple[int, int, int],dtype: torch.dtype, dynamic_dims: bool | tuple[bool, bool, bool],mfma_variant: MMAType):

    pipelined_version = """
    #map = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 4) mod 64)>
    #map1 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>
    #map2 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 4 - ((s1 * 32 + s0 floordiv 4) floordiv 64) * 64)>
    #map3 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16)>
    #map4 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
    #map5 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 16)>
    #map6 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16 + 16)>
    #map7 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 32)>
    #map8 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 32 + 16)>
    #map9 = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 8 - (s1 floordiv 4) * 32)>
    #map10 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4)>
    #map11 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 32 - (s0 floordiv 16) * 16)>
    #map12 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 1)>
    #map13 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 2)>
    #map14 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 3)>
    #map15 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 32 - (s0 floordiv 16) * 16 + 16)>
    #map16 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>
    #map17 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>
    #map18 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>
    #map19 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>
    #prefetch_current = affine_map<()[s1] -> (0 * 32 + s1 * 8 - (s1 floordiv 4) * 32)>
    #prefetch_next = affine_map<()[s1] -> (1 * 32 + s1 * 8 - (s1 floordiv 4) * 32)>
    #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>
    module attributes {transform.with_named_sequence} {
    stream.executable private @gemm {
        stream.executable.export public @gemm workgroups() -> (index, index, index) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        stream.return %c1, %c2, %c1 : index, index, index
        }
        builtin.module {
        func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
            %cst = arith.constant dense<0.000000e+00> : vector<8xbf16>
            %cst_0 = arith.constant dense<511> : vector<8xindex>
            %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c16 = arith.constant 16 : index
            %c4608 = arith.constant 4608 : index
            %c0 = arith.constant 0 : index
            %pipeline_depth= arith.constant 2 : index
            %cst_2 = arith.constant dense<0.000000e+00> : vector<4xf32>
            %block_id_y = gpu.block_id  y upper_bound 2
            %thread_id_x = gpu.thread_id  x upper_bound 128
            %thread_id_y = gpu.thread_id  y upper_bound 2
            %alloc = memref.alloc() : memref<9216xi8, #gpu.address_space<workgroup>>
            %view = memref.view %alloc[%c0][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xbf16, #gpu.address_space<workgroup>>
            %view_3 = memref.view %alloc[%c4608][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xbf16, #gpu.address_space<workgroup>>
            %0 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<128x511xbf16, strided<[511, 1], offset: ?>>
            %1 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<64x511xbf16, strided<[511, 1], offset: ?>>
            %2 = affine.apply #map()[%thread_id_x, %thread_id_y]
            %3 = affine.apply #map1()[%thread_id_x]
            %4 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y]
            %5 = affine.apply #map3()[%thread_id_x, %thread_id_y]
            %6 = affine.apply #map4()[%thread_id_x]
            %7 = affine.apply #map5()[%thread_id_x]
            %8 = affine.apply #map6()[%thread_id_x, %thread_id_y]
            %9 = affine.apply #map7()[%thread_id_x]
            %10 = affine.apply #map8()[%thread_id_x]
            // pre-load 
            %next = affine.apply #prefetch_current()[%thread_id_x]
            %current = affine.apply #prefetch_next()[%thread_id_x]
            //normally %39 depends on arg3 loop It K , replace it with 0 and 1 for two first it
            %prefetchA_0 = vector.load %1[%2, %current] : memref<64x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
            %prefetchB_0 = vector.load %0[%4, %current] : memref<128x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
            %prefetchA_1 = vector.load %1[%2, %next] : memref<64x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
            %prefetchB_1 = vector.load %0[%4, %next] : memref<128x511xbf16, strided<[511, 1], offset: ?>>, vector<8xbf16>
            %11:8 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %cst_2, %arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %prefetchAcurrent=%prefetchA_0 , %prefetchBcurrent= %prefetchB_0 , %prefetchAnext=%prefetchA_1 , %prefetchBnext= %prefetchB_1) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,vector<8xbf16>,vector<8xbf16>,vector<8xbf16>,vector<8xbf16>) {
            %K_iter= arith.addi %arg3, %pipeline_depth : index
            %39 = affine.apply #map9()[%K_iter, %thread_id_x]   //+pipeline_depth ,2  in this case
            %40 = vector.broadcast %39 : index to vector<8xindex>
            %41 = arith.addi %40, %cst_1 overflow<nsw, nuw> : vector<8xindex>
            %42 = arith.cmpi slt, %41, %cst_0 : vector<8xindex>

            %43 = vector.maskedload %1[%2, %39], %42, %cst : memref<64x511xbf16, strided<[511, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
            %44 = vector.maskedload %0[%4, %39], %42, %cst : memref<128x511xbf16, strided<[511, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>

            vector.store %prefetchAcurrent, %view_3[%2, %3] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
            vector.store %prefetchBcurrent, %view[%2, %3] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
            
            amdgpu.lds_barrier
            %45 = vector.load %view[%5, %6] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
            %49 = vector.load %view_3[%9, %6] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>

            %46 = vector.load %view[%5, %7] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
            %50 = vector.load %view_3[%9, %7] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>

            %47 = vector.load %view[%8, %6] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
            %48 = vector.load %view[%8, %7] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>

            %51 = vector.load %view_3[%10, %6] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
            %52 = vector.load %view_3[%10, %7] : memref<64x36xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>

            // mfma repeated four times to cover the whole 64x64 tile each mfma covers a 16x16 tile
            %53 = amdgpu.mfma %49 * %45 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
            %54 = amdgpu.mfma %50 * %46 + %53 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
            
            %55 = amdgpu.mfma %49 * %47 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
            %56 = amdgpu.mfma %50 * %48 + %55 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
            
            %57 = amdgpu.mfma %51 * %45 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
            %58 = amdgpu.mfma %52 * %46 + %57 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
            
            %59 = amdgpu.mfma %51 * %47 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
            %60 = amdgpu.mfma %52 * %48 + %59 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
            scf.yield %54, %56, %58, %60,%prefetchAnext, %prefetchBnext, %43 , %44: vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,vector<8xbf16>,vector<8xbf16>,vector<8xbf16>,vector<8xbf16>
            }
            %12 = vector.extract_strided_slice %11#0 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %13 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<64x128xf32, strided<[128, 1], offset: ?>>
            %14 = affine.apply #map10()[%thread_id_x]
            %15 = affine.apply #map11()[%thread_id_x, %block_id_y, %thread_id_y]
            vector.store %12, %13[%14, %15] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %16 = vector.extract_strided_slice %11#0 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %17 = affine.apply #map12()[%thread_id_x]
            vector.store %16, %13[%17, %15] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %18 = vector.extract_strided_slice %11#0 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %19 = affine.apply #map13()[%thread_id_x]
            vector.store %18, %13[%19, %15] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %20 = vector.extract_strided_slice %11#0 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %21 = affine.apply #map14()[%thread_id_x]
            vector.store %20, %13[%21, %15] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %22 = vector.extract_strided_slice %11#1 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %23 = affine.apply #map15()[%thread_id_x, %block_id_y, %thread_id_y]
            vector.store %22, %13[%14, %23] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %24 = vector.extract_strided_slice %11#1 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            vector.store %24, %13[%17, %23] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %25 = vector.extract_strided_slice %11#1 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            vector.store %25, %13[%19, %23] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %26 = vector.extract_strided_slice %11#1 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            vector.store %26, %13[%21, %23] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %27 = vector.extract_strided_slice %11#2 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %28 = affine.apply #map16()[%thread_id_x]
            vector.store %27, %13[%28, %15] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %29 = vector.extract_strided_slice %11#2 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %30 = affine.apply #map17()[%thread_id_x]
            vector.store %29, %13[%30, %15] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %31 = vector.extract_strided_slice %11#2 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %32 = affine.apply #map18()[%thread_id_x]
            vector.store %31, %13[%32, %15] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %33 = vector.extract_strided_slice %11#2 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %34 = affine.apply #map19()[%thread_id_x]
            vector.store %33, %13[%34, %15] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %35 = vector.extract_strided_slice %11#3 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            vector.store %35, %13[%28, %23] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %36 = vector.extract_strided_slice %11#3 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            vector.store %36, %13[%30, %23] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %37 = vector.extract_strided_slice %11#3 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            vector.store %37, %13[%32, %23] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            %38 = vector.extract_strided_slice %11#3 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            vector.store %38, %13[%34, %23] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
            return
        }
        }
    }
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<64x511xbf16>
        %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<128x511xbf16>
        %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<64x128xf32>
        %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<64x511xbf16>, tensor<128x511xbf16>, tensor<64x128xf32>) -> %2
        %4 = hal.tensor.barrier join(%3 : tensor<64x128xf32>) => %arg4 : !hal.fence
        %5 = hal.tensor.export %4 : tensor<64x128xf32> -> !hal.buffer_view
        return %5 : !hal.buffer_view
    }
    }
    """
    
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
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        override_mlir=pipelined_version,
        wave_runtime=True,
        use_scheduling_barriers=enable_scheduling_barriers,
    )
    
    options = set_default_run_config(options)
    return wave_compile(options, gemm)


def testReorderedPingPongGemm(shape: tuple[int, int, int],dtype: torch.dtype, dynamic_dims: bool | tuple[bool, bool, bool],mfma_variant: MMAType):
   
    # Input sizes
    M = shape[0]
    N = shape[1]
    K = shape[2]
    # Workgroup tile sizes
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    # Group size
    GROUP_SIZE_M = 16

    reordered_gemm, hyperparams = get_reordered_matmul(
        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, mfma_variant
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        use_buffer_ops=True,
        print_mlir=True,
    )
    options = set_default_run_config(options)
    return wave_compile(options, reordered_gemm)


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
    DEVICE_M = tkl.sym.DEVICE_M
    DEVICE_N = tkl.sym.DEVICE_N
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    wave_dtype = torch_dtype_to_wave(dtype)
    # Expose user-constraints
    
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.DeviceConstraint(M, DEVICE_M, 0)]
    constraints += [tkw.DeviceConstraint(N, DEVICE_N, 1)]
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
    def test(
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
        DEVICE_M:8192,
        DEVICE_N:16384,
    }
    hyperparams.update(get_default_scheduling_params())


    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=enable_scheduling_barriers,
        wave_runtime=False,
        #print_ir_after=["all"],
    )
    
    options = set_default_run_config(options)
    return wave_compile(options, test)
    

def torch_compile_matmul(a, b):
    # Simple wrapper around matmul
    def fn(x, y):
        return torch.mm(x, y.t())   # same convention as your GEMM test

    compiled_fn = torch.compile(fn)     # JIT-compile it with TorchDynamo+Inductor
    return compiled_fn(a, b)


def calculate_diff_gemm(M, N, K, dtype=torch.float16):
    # Random test matrices
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(N, K, dtype=dtype, device="cuda")  # careful: ABᵀ → shape (M,N)
    C = torch.empty((M, N), dtype=torch.float32, device="cuda")

    # ---- WAVE ----
    wave_kernel = get_wave_gemm((M,N,K), dtype, [False,False,False],MMAType.F32_32x32x16_K8_F16)  # <- your Wave GEMM builder
    wave_kernel(A.clone(), B.clone(),C) 
    
    # ---- WAVE PIPELINED ----
    # wave_gemm = testReorderedPingPongGemm( (M,N,K), dtype, [False,False,False], MMAType.F32_16x16x16_F16)
    # wave_gemm(A,B,C) 

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
# M_vals = [64,128,256]
# N_vals = [128 ,128,256]
# K_vals = [511,511,511 ]
# M_vals = [64]
# N_vals = [128]
# K_vals = [511]

# M_vals = [16384]
# N_vals = [32768]
# K_vals = [6144]

M_vals = [256]
N_vals = [256]
K_vals = [511]

configs = list(itertools.product(M_vals, N_vals, K_vals))

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        # line_vals=["wave", "reordered_gemm","wave_pipelined_16x16","wave_pipelined_32x32", "harsh_version_32x32x8","triton"],
        # line_names=["wave_16x16x16", "reordered_gemm","wave_pipelined_16x16_16","wave_pipelined_32x32_16","harsh_version_32x32x8","Triton"],
        line_vals=[ "triton","reordered_gemm"],
        line_names=[ "triton","reordered_gemm"],
        styles=[("blue","-"), ("red","-"),("green","-"),("orange","-"),("black","-"),("black","-")],
        ylabel="ms",
        plot_name="gemm-abt-performance",
        args={},
    )
)
def bench(M, N, K, provider):
    dtype = torch.float16
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
        
    elif provider == "wave_pipelined_16x16":
        # plug your compiled wave GEMM here; it should compute C in fp32
        wave_gemm = get_wave_gemm_pipelined_16x16( (M,N,K), dtype, [False,False,False],MMAType.F32_32x32x16_K8_F16)
        C = torch.empty((M, N), dtype=torch.float32, device="cuda")
        _ = wave_gemm(A, B, C)   # warmup; expect A(M,K), B(N,K), C(M,N)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: wave_gemm(A, B, C),
            quantiles=quantiles,
        )
    
    elif provider == "harsh_version_32x32x8":
        # plug your compiled wave GEMM here; it should compute C in fp32
        wave_gemm = get_wave_gemm_pipelined_harsh( (M,N,K), dtype, [False,False,False],MMAType.F32_32x32x16_K8_F16)
        C = torch.empty((M, N), dtype=torch.float32, device="cuda")
        _ = wave_gemm(A, B, C)   # warmup; expect A(M,K), B(N,K), C(M,N)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: wave_gemm(A, B, C),
            quantiles=quantiles,
        )
    elif provider == "reordered_gemm":
        # plug your compiled wave GEMM here; it should compute C in fp32
        wave_gemm = testReorderedPingPongGemm( (M,N,K), dtype, [False,False,False], MMAType.F32_16x16x16_F16)
        C = torch.empty((M, N), dtype=torch.float32, device="cuda")
        _ = wave_gemm(A, B, C)   # warmup; expect A(M,K), B(N,K), C(M,N)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: wave_gemm(A, B, C),
            quantiles=quantiles,
        )
    elif provider == "wave_pipelined_32x32":
        # plug your compiled wave GEMM here; it should compute C in fp32
        wave_gemm = get_wave_gemm_pipelined_32x32x16( (M,N,K), dtype, [False,False,False],MMAType.F32_32x32x16_K8_F16)
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
    
    # M_vals = [16384]
    # N_vals = [32768]
    # K_vals = [6144]
    
    # perf sweep
    bench.run(print_data=True, show_plots=False)
    
    #calculate_diff_gemm(256, 256, 511)
    #calculate_diff_gemm(64, 128, 511)
    #calculate_diff_gemm(16384, 32768, 6144)

    







