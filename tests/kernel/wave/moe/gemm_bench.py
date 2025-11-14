import triton
import triton.language as tl
import torch
import itertools

import triton.compiler as tc

from torch.nn import functional as F
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


def triton_matmul_abt(A: torch.Tensor, B: torch.Tensor, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2):
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
        schedule=SchedulingType.PREFETCH,
        use_buffer_ops=True,
        print_mlir=True,
    )
    
    options = set_default_run_config(options)
    return wave_compile(options, gemm)


def testReorderedPingPongGemm(shape: tuple[int, int, int],dtype: torch.dtype, dynamic_dims: bool | tuple[bool, bool, bool],mfma_variant: MMAType):

    asm = """
    #map = affine_map<()[s0, s1] -> ((s0 * 2 + s1) mod 8)>
    #map1 = affine_map<()[s0, s1, s2] -> (((s0 * 132 + s1 * 66 + s2 - ((s0 * 2 + s1) floordiv 8) * 527) floordiv 4256) * -16 + 2)>
    #map2 = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128 + ((s2 * 132 + s3 * 66 + s4 - ((s2 * 2 + s3) floordiv 8) * 527) floordiv 4256) * 2048 + (((s2 * 132 + s3 * 66 + s5 - ((s2 * 2 + s3) floordiv 8) * 527) mod 4256) mod s6) * 128)>
    #map3 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 8) * 64)>
    #map4 = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + ((s2 * 132 + s3 * 66 + s4 - ((s2 * 2 + s3) floordiv 8) * 527) floordiv 4256) * 2048 + (((s2 * 132 + s3 * 66 + s5 - ((s2 * 2 + s3) floordiv 8) * 527) mod 4256) mod s6) * 128 + 64)>
    #map5 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256)>
    #map6 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 64)>
    #map7 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 128)>
    #map8 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 192)>
    #map9 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 128)>
    #map10 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + 64)>
    #map11 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
    #map12 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
    #map13 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
    #map14 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
    #map15 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
    #map16 = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
    #map17 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8)>
    #map18 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 16)>
    #map19 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32)>
    #map20 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 32)>
    #map21 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 64)>
    #map22 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 96)>
    #map23 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 32)>
    #map24 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 48)>
    #map25 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 - (s1 floordiv 8) * 64 + 64)>
    #map26 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256)>
    #map27 = affine_map<()[s0, s1, s2, s3, s4] -> (((s0 * 132 + s1 * 66 + s2 - ((s0 * 2 + s1) floordiv 8) * 527) floordiv 4256) * 2048 + (((s0 * 132 + s1 * 66 + s3 - ((s0 * 2 + s1) floordiv 8) * 527) mod 4256) mod s4) * 128)>
    #map28 = affine_map<()[s0, s1, s2, s3] -> ((((s0 * 66 + s1 * 132 + s2 - ((s0 + s1 * 2) floordiv 8) * 527) mod 4256) floordiv s3) * 256)>
    #map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
    #map30 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
    #map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
    #map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
    #map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
    #map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
    #map35 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
    #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
    #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
    #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
    #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
    #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
    #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
    #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
    #map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
    #map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
    #map45 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 32)>
    #map46 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 64)>
    #map47 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 96)>
    #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
    module attributes {transform.with_named_sequence} {
    stream.executable private @gemm {
        stream.executable.export public @gemm workgroups() -> (index, index, index) {
        %c2 = arith.constant 2 : index
        %c266 = arith.constant 266 : index
        %c1 = arith.constant 1 : index
        stream.return %c2, %c266, %c1 : index, index, index
        }
        builtin.module {
        func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
            %c4_i32 = arith.constant 4 : i32
            %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>
            %cst_0 = arith.constant dense<1073741823> : vector<8xindex>
            %c1024_i14 = arith.constant 1024 : i14
            %c536870911 = arith.constant 536870911 : index
            %c2147483643_i64 = arith.constant 2147483643 : i64
            %c536870910 = arith.constant 536870910 : index
            %c0_i32 = arith.constant 0 : i32
            %c15 = arith.constant 15 : index
            %c68032 = arith.constant 68032 : index
            %c2147483645_i64 = arith.constant 2147483645 : i64
            %c1073741822 = arith.constant 1073741822 : index
            %c1024 = arith.constant 1024 : index
            %c1 = arith.constant 1 : index
            %c4 = arith.constant 4 : index
            %c34816 = arith.constant 34816 : index
            %cst_1 = arith.constant dense<0.000000e+00> : vector<16xf32>
            %c0 = arith.constant 0 : index
            %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<f16>
            %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<f16>
            %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
            %block_id_x = gpu.block_id  x upper_bound 2
            %block_id_y = gpu.block_id  y upper_bound 266
            %thread_id_x = gpu.thread_id  x upper_bound 256
            %thread_id_y = gpu.thread_id  y upper_bound 2
            %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 1024], strides: [1024, 1] : memref<f16> to memref<256x1024xf16, strided<[1024, 1], offset: ?>>
            %reinterpret_cast_2 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [68032, 1024], strides: [1024, 1] : memref<f16> to memref<68032x1024xf16, strided<[1024, 1], offset: ?>>
            %reinterpret_cast_3 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 68032], strides: [68032, 1] : memref<f32> to memref<256x68032xf32, strided<[68032, 1], offset: ?>>
            %alloc = memref.alloc() : memref<52224xi8, #gpu.address_space<workgroup>>
            %view = memref.view %alloc[%c0][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<256x68xf16, #gpu.address_space<workgroup>>
            %view_4 = memref.view %alloc[%c34816][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<128x68xf16, #gpu.address_space<workgroup>>
            %3 = affine.apply #map()[%block_id_y, %block_id_x]
            %4 = arith.minsi %3, %c4 : index
            %5 = affine.apply #map1()[%block_id_y, %block_id_x, %4]
            %6 = arith.maxsi %5, %c1 : index
            %7 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
            %8 = affine.apply #map3()[%thread_id_x]
            %9 = arith.muli %7, %c1024 overflow<nsw> : index
            %10 = arith.addi %9, %8 overflow<nsw> : index
            %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x1024xf16, strided<[1024, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
            %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<f16> to memref<?xf16, strided<[1], offset: ?>>
            %11 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>
            %66 = affine.apply #map9()[%thread_id_x, %thread_id_y]
            //%12 = vector.load %11[%10] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            amdgpu.gather_to_lds %11[%10], %view_4[%66, %8] : vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x68xf16, #gpu.address_space<workgroup>>
            //vector.store %12, %view_4[%66, %8] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>

            %13 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
            %14 = arith.muli %13, %c1024 overflow<nsw> : index
            %15 = arith.addi %14, %8 overflow<nsw> : index
            //%16 = vector.load %11[%15] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %67 = affine.apply #map10()[%thread_id_x, %thread_id_y]
            //vector.store %16, %view_4[%67, %8] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            amdgpu.gather_to_lds %11[%15], %view_4[%67, %8] : vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x68xf16, #gpu.address_space<workgroup>>

            %17 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
            %18 = arith.cmpi slt, %17, %c68032 : index
            %19 = vector.broadcast %18 : i1 to vector<8xi1>
            %20 = arith.muli %17, %c1024 overflow<nsw> : index
            %21 = arith.addi %20, %8 overflow<nsw> : index
            %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_2 : memref<68032x1024xf16, strided<[1024, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
            %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<f16> to memref<?xf16, strided<[1], offset: ?>>
            %22 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>
            %23 = arith.index_cast %21 : index to i32
            %24 = vector.broadcast %23 : i32 to vector<8xi32>
            %25 = arith.addi %24, %cst : vector<8xi32>
            %26 = arith.index_cast %25 : vector<8xi32> to vector<8xindex>
            %27 = arith.select %19, %26, %cst_0 : vector<8xi1>, vector<8xindex>
            %28 = vector.extract %27[0] : index from vector<8xindex>
            //%29 = vector.load %22[%28] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %68 = affine.apply #map11()[%thread_id_x, %thread_id_y]
            //vector.store %29, %view[%68, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            amdgpu.gather_to_lds %22[%28] , %view[%68, %8] : vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x68xf16, #gpu.address_space<workgroup>>
            %30 = affine.apply #map6()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
            %31 = arith.cmpi slt, %30, %c68032 : index
            %32 = vector.broadcast %31 : i1 to vector<8xi1>
            %33 = arith.muli %30, %c1024 overflow<nsw> : index
            %34 = arith.addi %33, %8 overflow<nsw> : index
            %35 = arith.index_cast %34 : index to i32
            %36 = vector.broadcast %35 : i32 to vector<8xi32>
            %37 = arith.addi %36, %cst : vector<8xi32>
            %38 = arith.index_cast %37 : vector<8xi32> to vector<8xindex>
            %39 = arith.select %32, %38, %cst_0 : vector<8xi1>, vector<8xindex>
            %40 = vector.extract %39[0] : index from vector<8xindex>
            //%41 = vector.load %22[%40] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %69 = affine.apply #map12()[%thread_id_x, %thread_id_y]
            //vector.store %41, %view[%69, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            amdgpu.gather_to_lds %22[%40] , %view[%69, %8] : vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x68xf16, #gpu.address_space<workgroup>>
            %42 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
            %43 = arith.cmpi slt, %42, %c68032 : index
            %44 = vector.broadcast %43 : i1 to vector<8xi1>
            %45 = arith.muli %42, %c1024 overflow<nsw> : index
            %46 = arith.addi %45, %8 overflow<nsw> : index
            %47 = arith.index_cast %46 : index to i32
            %48 = vector.broadcast %47 : i32 to vector<8xi32>
            %49 = arith.addi %48, %cst : vector<8xi32>
            %50 = arith.index_cast %49 : vector<8xi32> to vector<8xindex>
            %51 = arith.select %44, %50, %cst_0 : vector<8xi1>, vector<8xindex>
            %52 = vector.extract %51[0] : index from vector<8xindex>
            //%53 = vector.load %22[%52] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %70 = affine.apply #map13()[%thread_id_x, %thread_id_y]
            //vector.store %53, %view[%70, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            amdgpu.gather_to_lds %22[%52] , %view[%70, %8] : vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x68xf16, #gpu.address_space<workgroup>>
            %54 = affine.apply #map8()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
            %55 = arith.cmpi slt, %54, %c68032 : index
            %56 = vector.broadcast %55 : i1 to vector<8xi1>
            %57 = arith.muli %54, %c1024 overflow<nsw> : index
            %58 = arith.addi %57, %8 overflow<nsw> : index
            %59 = arith.index_cast %58 : index to i32
            %60 = vector.broadcast %59 : i32 to vector<8xi32>
            %61 = arith.addi %60, %cst : vector<8xi32>
            %62 = arith.index_cast %61 : vector<8xi32> to vector<8xindex>
            %63 = arith.select %56, %62, %cst_0 : vector<8xi1>, vector<8xindex>
            %64 = vector.extract %63[0] : index from vector<8xindex>
            //%65 = vector.load %22[%64] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %71 = affine.apply #map14()[%thread_id_x, %thread_id_y]
            //vector.store %65, %view[%71, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            amdgpu.gather_to_lds%22[%64] ,  %view[%71, %8]: vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x68xf16, #gpu.address_space<workgroup>>
            amdgpu.lds_barrier
            %72 = affine.apply #map15()[%thread_id_x, %thread_id_y]
            %73 = arith.index_cast %72 : index to i32
            %74 = arith.cmpi sge, %73, %c4_i32 : i32
            %75 = arith.cmpi slt, %73, %c4_i32 : i32
            scf.if %74 {
            rocdl.s.barrier
            }
            %76 = affine.apply #map16()[%thread_id_x]
            %77 = affine.apply #map17()[%thread_id_x]
            %78 = affine.apply #map18()[%thread_id_x]
            %79 = affine.apply #map19()[%thread_id_x, %thread_id_y]
            %80 = affine.apply #map20()[%thread_id_x, %thread_id_y]
            %81 = affine.apply #map21()[%thread_id_x, %thread_id_y]
            %82 = affine.apply #map22()[%thread_id_x, %thread_id_y]
            %83 = affine.apply #map23()[%thread_id_x]
            %84 = affine.apply #map24()[%thread_id_x]
            %85:4 = scf.for %arg3 = %c0 to %c15 step %c1 iter_args(%arg4 = %cst_1, %arg5 = %cst_1, %arg6 = %cst_1, %arg7 = %cst_1) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
            %369 = vector.load %view_4[%76, %77] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %370 = vector.load %view_4[%76, %78] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %371 = vector.load %view[%79, %77] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %372 = vector.load %view[%79, %78] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %373 = vector.load %view[%80, %77] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %374 = vector.load %view[%80, %78] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %375 = vector.load %view[%81, %77] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %376 = vector.load %view[%81, %78] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %377 = vector.load %view[%82, %77] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %378 = vector.load %view[%82, %78] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
            %379 = affine.apply #map25()[%arg3, %thread_id_x]
            %380 = arith.addi %14, %379 overflow<nsw> : index
            %381 = vector.load %11[%380] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %382 = arith.addi %9, %379 overflow<nsw> : index
            %383 = vector.load %11[%382] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
            %384 = vector.load %view_4[%76, %83] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %385 = vector.load %view_4[%76, %84] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %386 = vector.load %view[%79, %83] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %387 = vector.load %view[%79, %84] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %388 = vector.load %view[%80, %83] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %389 = vector.load %view[%80, %84] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %390 = vector.load %view[%81, %83] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %391 = vector.load %view[%81, %84] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %392 = vector.load %view[%82, %83] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %393 = vector.load %view[%82, %84] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
            %394 = arith.addi %33, %379 overflow<nsw> : index
            %395 = arith.index_cast %394 : index to i32
            %396 = vector.broadcast %395 : i32 to vector<8xi32>
            %397 = arith.addi %396, %cst : vector<8xi32>
            %398 = arith.index_cast %397 : vector<8xi32> to vector<8xindex>
            %399 = arith.select %32, %398, %cst_0 : vector<8xi1>, vector<8xindex>
            %400 = vector.extract %399[0] : index from vector<8xindex>
            %401 = vector.load %22[%400] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %402 = arith.addi %45, %379 overflow<nsw> : index
            %403 = arith.index_cast %402 : index to i32
            %404 = vector.broadcast %403 : i32 to vector<8xi32>
            %405 = arith.addi %404, %cst : vector<8xi32>
            %406 = arith.index_cast %405 : vector<8xi32> to vector<8xindex>
            %407 = arith.select %44, %406, %cst_0 : vector<8xi1>, vector<8xindex>
            %408 = vector.extract %407[0] : index from vector<8xindex>
            %409 = vector.load %22[%408] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %410 = arith.addi %20, %379 overflow<nsw> : index
            %411 = arith.index_cast %410 : index to i32
            %412 = vector.broadcast %411 : i32 to vector<8xi32>
            %413 = arith.addi %412, %cst : vector<8xi32>
            %414 = arith.index_cast %413 : vector<8xi32> to vector<8xindex>
            %415 = arith.select %19, %414, %cst_0 : vector<8xi1>, vector<8xindex>
            %416 = vector.extract %415[0] : index from vector<8xindex>
            %417 = vector.load %22[%416] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %418 = arith.addi %57, %379 overflow<nsw> : index
            %419 = arith.index_cast %418 : index to i32
            %420 = vector.broadcast %419 : i32 to vector<8xi32>
            %421 = arith.addi %420, %cst : vector<8xi32>
            %422 = arith.index_cast %421 : vector<8xi32> to vector<8xindex>
            %423 = arith.select %56, %422, %cst_0 : vector<8xi1>, vector<8xindex>
            %424 = vector.extract %423[0] : index from vector<8xindex>
            %425 = vector.load %22[%424] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            rocdl.s.barrier
            llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
            rocdl.s.setprio 1
            %426 = amdgpu.mfma %369 * %371 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %427 = amdgpu.mfma %370 * %372 + %426 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %428 = amdgpu.mfma %369 * %373 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %429 = amdgpu.mfma %370 * %374 + %428 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %430 = amdgpu.mfma %369 * %375 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %431 = amdgpu.mfma %370 * %376 + %430 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %432 = amdgpu.mfma %369 * %377 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %433 = amdgpu.mfma %370 * %378 + %432 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            rocdl.s.setprio 0
            amdgpu.lds_barrier
            llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
            vector.store %383, %view_4[%66, %8] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            vector.store %381, %view_4[%67, %8] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            vector.store %409, %view[%70, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            vector.store %425, %view[%71, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            vector.store %417, %view[%68, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            vector.store %401, %view[%69, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            rocdl.s.barrier
            llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
            rocdl.s.setprio 1
            %434 = amdgpu.mfma %384 * %386 + %427 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %435 = amdgpu.mfma %385 * %387 + %434 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %436 = amdgpu.mfma %384 * %388 + %429 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %437 = amdgpu.mfma %385 * %389 + %436 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %438 = amdgpu.mfma %384 * %390 + %431 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %439 = amdgpu.mfma %385 * %391 + %438 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %440 = amdgpu.mfma %384 * %392 + %433 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %441 = amdgpu.mfma %385 * %393 + %440 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            rocdl.s.setprio 0
            llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
            amdgpu.lds_barrier
            scf.yield %435, %437, %439, %441 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
            }
            scf.if %75 {
            rocdl.s.barrier
            }
            %86 = affine.apply #map19()[%thread_id_x, %thread_id_y]
            %87 = affine.apply #map17()[%thread_id_x]
            %88 = vector.load %view[%86, %87] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %89 = affine.apply #map18()[%thread_id_x]
            %90 = vector.load %view[%86, %89] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %91 = affine.apply #map23()[%thread_id_x]
            %92 = vector.load %view[%86, %91] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %93 = affine.apply #map24()[%thread_id_x]
            %94 = vector.load %view[%86, %93] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %95 = affine.apply #map20()[%thread_id_x, %thread_id_y]
            %96 = vector.load %view[%95, %87] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %97 = vector.load %view[%95, %89] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %98 = vector.load %view[%95, %91] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %99 = vector.load %view[%95, %93] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %100 = affine.apply #map21()[%thread_id_x, %thread_id_y]
            %101 = vector.load %view[%100, %87] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %102 = vector.load %view[%100, %89] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %103 = vector.load %view[%100, %91] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %104 = vector.load %view[%100, %93] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %105 = affine.apply #map22()[%thread_id_x, %thread_id_y]
            %106 = vector.load %view[%105, %87] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %107 = vector.load %view[%105, %89] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %108 = vector.load %view[%105, %91] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %109 = vector.load %view[%105, %93] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %110 = affine.apply #map16()[%thread_id_x]
            %111 = vector.load %view_4[%110, %87] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %112 = vector.load %view_4[%110, %89] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %113 = vector.load %view_4[%110, %91] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %114 = vector.load %view_4[%110, %93] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %115 = amdgpu.mfma %111 * %88 + %85#0 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %116 = amdgpu.mfma %112 * %90 + %115 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %117 = amdgpu.mfma %113 * %92 + %116 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %118 = amdgpu.mfma %114 * %94 + %117 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %119 = amdgpu.mfma %111 * %96 + %85#1 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %120 = amdgpu.mfma %112 * %97 + %119 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %121 = amdgpu.mfma %113 * %98 + %120 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %122 = amdgpu.mfma %114 * %99 + %121 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %123 = amdgpu.mfma %111 * %101 + %85#2 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %124 = amdgpu.mfma %112 * %102 + %123 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %125 = amdgpu.mfma %113 * %103 + %124 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %126 = amdgpu.mfma %114 * %104 + %125 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %127 = amdgpu.mfma %111 * %106 + %85#3 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %128 = amdgpu.mfma %112 * %107 + %127 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %129 = amdgpu.mfma %113 * %108 + %128 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %130 = amdgpu.mfma %114 * %109 + %129 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
            %131 = vector.extract_strided_slice %118 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %132 = affine.apply #map26()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
            %133 = arith.cmpi slt, %132, %c68032 : index
            %134 = affine.apply #map27()[%block_id_y, %block_id_x, %4, %4, %6]
            %135 = affine.apply #map28()[%block_id_x, %block_id_y, %4, %6]
            %136 = affine.apply #map29()[%thread_id_x]
            %137 = arith.muli %134, %c68032 overflow<nsw> : index
            %138 = arith.muli %136, %c68032 overflow<nsw> : index
            %139 = arith.addi %137, %135 overflow<nsw> : index
            %140 = arith.addi %138, %86 overflow<nsw> : index
            %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %reinterpret_cast_3 : memref<256x68032xf32, strided<[68032, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
            %141 = arith.addi %139, %offset_12 overflow<nsw> : index
            %reinterpret_cast_15 = memref.reinterpret_cast %2 to offset: [%141], sizes: [%c536870910], strides: [1] : memref<f32> to memref<?xf32, strided<[1], offset: ?>>
            %142 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_15 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
            %143 = arith.select %133, %140, %c536870911 : index
            vector.store %131, %142[%143] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %144 = vector.extract_strided_slice %118 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %145 = affine.apply #map30()[%thread_id_x]
            %146 = arith.muli %145, %c68032 overflow<nsw> : index
            %147 = arith.addi %146, %86 overflow<nsw> : index
            %148 = arith.select %133, %147, %c536870911 : index
            vector.store %144, %142[%148] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %149 = vector.extract_strided_slice %118 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %150 = affine.apply #map31()[%thread_id_x]
            %151 = arith.muli %150, %c68032 overflow<nsw> : index
            %152 = arith.addi %151, %86 overflow<nsw> : index
            %153 = arith.select %133, %152, %c536870911 : index
            vector.store %149, %142[%153] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %154 = vector.extract_strided_slice %118 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %155 = affine.apply #map32()[%thread_id_x]
            %156 = arith.muli %155, %c68032 overflow<nsw> : index
            %157 = arith.addi %156, %86 overflow<nsw> : index
            %158 = arith.select %133, %157, %c536870911 : index
            vector.store %154, %142[%158] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %159 = vector.extract_strided_slice %118 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %160 = affine.apply #map33()[%thread_id_x]
            %161 = arith.muli %160, %c68032 overflow<nsw> : index
            %162 = arith.addi %161, %86 overflow<nsw> : index
            %163 = arith.select %133, %162, %c536870911 : index
            vector.store %159, %142[%163] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %164 = vector.extract_strided_slice %118 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %165 = affine.apply #map34()[%thread_id_x]
            %166 = arith.muli %165, %c68032 overflow<nsw> : index
            %167 = arith.addi %166, %86 overflow<nsw> : index
            %168 = arith.select %133, %167, %c536870911 : index
            vector.store %164, %142[%168] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %169 = vector.extract_strided_slice %118 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %170 = affine.apply #map35()[%thread_id_x]
            %171 = arith.muli %170, %c68032 overflow<nsw> : index
            %172 = arith.addi %171, %86 overflow<nsw> : index
            %173 = arith.select %133, %172, %c536870911 : index
            vector.store %169, %142[%173] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %174 = vector.extract_strided_slice %118 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %175 = affine.apply #map36()[%thread_id_x]
            %176 = arith.muli %175, %c68032 overflow<nsw> : index
            %177 = arith.addi %176, %86 overflow<nsw> : index
            %178 = arith.select %133, %177, %c536870911 : index
            vector.store %174, %142[%178] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %179 = vector.extract_strided_slice %118 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %180 = affine.apply #map37()[%thread_id_x]
            %181 = arith.muli %180, %c68032 overflow<nsw> : index
            %182 = arith.addi %181, %86 overflow<nsw> : index
            %183 = arith.select %133, %182, %c536870911 : index
            vector.store %179, %142[%183] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %184 = vector.extract_strided_slice %118 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %185 = affine.apply #map38()[%thread_id_x]
            %186 = arith.muli %185, %c68032 overflow<nsw> : index
            %187 = arith.addi %186, %86 overflow<nsw> : index
            %188 = arith.select %133, %187, %c536870911 : index
            vector.store %184, %142[%188] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %189 = vector.extract_strided_slice %118 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %190 = affine.apply #map39()[%thread_id_x]
            %191 = arith.muli %190, %c68032 overflow<nsw> : index
            %192 = arith.addi %191, %86 overflow<nsw> : index
            %193 = arith.select %133, %192, %c536870911 : index
            vector.store %189, %142[%193] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %194 = vector.extract_strided_slice %118 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %195 = affine.apply #map40()[%thread_id_x]
            %196 = arith.muli %195, %c68032 overflow<nsw> : index
            %197 = arith.addi %196, %86 overflow<nsw> : index
            %198 = arith.select %133, %197, %c536870911 : index
            vector.store %194, %142[%198] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %199 = vector.extract_strided_slice %118 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %200 = affine.apply #map41()[%thread_id_x]
            %201 = arith.muli %200, %c68032 overflow<nsw> : index
            %202 = arith.addi %201, %86 overflow<nsw> : index
            %203 = arith.select %133, %202, %c536870911 : index
            vector.store %199, %142[%203] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %204 = vector.extract_strided_slice %118 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %205 = affine.apply #map42()[%thread_id_x]
            %206 = arith.muli %205, %c68032 overflow<nsw> : index
            %207 = arith.addi %206, %86 overflow<nsw> : index
            %208 = arith.select %133, %207, %c536870911 : index
            vector.store %204, %142[%208] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %209 = vector.extract_strided_slice %118 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %210 = affine.apply #map43()[%thread_id_x]
            %211 = arith.muli %210, %c68032 overflow<nsw> : index
            %212 = arith.addi %211, %86 overflow<nsw> : index
            %213 = arith.select %133, %212, %c536870911 : index
            vector.store %209, %142[%213] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %214 = vector.extract_strided_slice %118 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %215 = affine.apply #map44()[%thread_id_x]
            %216 = arith.muli %215, %c68032 overflow<nsw> : index
            %217 = arith.addi %216, %86 overflow<nsw> : index
            %218 = arith.select %133, %217, %c536870911 : index
            vector.store %214, %142[%218] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %219 = vector.extract_strided_slice %122 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %220 = affine.apply #map45()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
            %221 = arith.cmpi slt, %220, %c68032 : index
            %222 = arith.addi %138, %95 overflow<nsw> : index
            %223 = arith.select %221, %222, %c536870911 : index
            vector.store %219, %142[%223] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %224 = vector.extract_strided_slice %122 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %225 = arith.addi %146, %95 overflow<nsw> : index
            %226 = arith.select %221, %225, %c536870911 : index
            vector.store %224, %142[%226] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %227 = vector.extract_strided_slice %122 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %228 = arith.addi %151, %95 overflow<nsw> : index
            %229 = arith.select %221, %228, %c536870911 : index
            vector.store %227, %142[%229] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %230 = vector.extract_strided_slice %122 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %231 = arith.addi %156, %95 overflow<nsw> : index
            %232 = arith.select %221, %231, %c536870911 : index
            vector.store %230, %142[%232] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %233 = vector.extract_strided_slice %122 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %234 = arith.addi %161, %95 overflow<nsw> : index
            %235 = arith.select %221, %234, %c536870911 : index
            vector.store %233, %142[%235] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %236 = vector.extract_strided_slice %122 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %237 = arith.addi %166, %95 overflow<nsw> : index
            %238 = arith.select %221, %237, %c536870911 : index
            vector.store %236, %142[%238] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %239 = vector.extract_strided_slice %122 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %240 = arith.addi %171, %95 overflow<nsw> : index
            %241 = arith.select %221, %240, %c536870911 : index
            vector.store %239, %142[%241] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %242 = vector.extract_strided_slice %122 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %243 = arith.addi %176, %95 overflow<nsw> : index
            %244 = arith.select %221, %243, %c536870911 : index
            vector.store %242, %142[%244] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %245 = vector.extract_strided_slice %122 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %246 = arith.addi %181, %95 overflow<nsw> : index
            %247 = arith.select %221, %246, %c536870911 : index
            vector.store %245, %142[%247] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %248 = vector.extract_strided_slice %122 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %249 = arith.addi %186, %95 overflow<nsw> : index
            %250 = arith.select %221, %249, %c536870911 : index
            vector.store %248, %142[%250] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %251 = vector.extract_strided_slice %122 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %252 = arith.addi %191, %95 overflow<nsw> : index
            %253 = arith.select %221, %252, %c536870911 : index
            vector.store %251, %142[%253] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %254 = vector.extract_strided_slice %122 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %255 = arith.addi %196, %95 overflow<nsw> : index
            %256 = arith.select %221, %255, %c536870911 : index
            vector.store %254, %142[%256] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %257 = vector.extract_strided_slice %122 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %258 = arith.addi %201, %95 overflow<nsw> : index
            %259 = arith.select %221, %258, %c536870911 : index
            vector.store %257, %142[%259] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %260 = vector.extract_strided_slice %122 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %261 = arith.addi %206, %95 overflow<nsw> : index
            %262 = arith.select %221, %261, %c536870911 : index
            vector.store %260, %142[%262] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %263 = vector.extract_strided_slice %122 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %264 = arith.addi %211, %95 overflow<nsw> : index
            %265 = arith.select %221, %264, %c536870911 : index
            vector.store %263, %142[%265] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %266 = vector.extract_strided_slice %122 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %267 = arith.addi %216, %95 overflow<nsw> : index
            %268 = arith.select %221, %267, %c536870911 : index
            vector.store %266, %142[%268] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %269 = vector.extract_strided_slice %126 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %270 = affine.apply #map46()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
            %271 = arith.cmpi slt, %270, %c68032 : index
            %272 = arith.addi %138, %100 overflow<nsw> : index
            %273 = arith.select %271, %272, %c536870911 : index
            vector.store %269, %142[%273] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %274 = vector.extract_strided_slice %126 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %275 = arith.addi %146, %100 overflow<nsw> : index
            %276 = arith.select %271, %275, %c536870911 : index
            vector.store %274, %142[%276] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %277 = vector.extract_strided_slice %126 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %278 = arith.addi %151, %100 overflow<nsw> : index
            %279 = arith.select %271, %278, %c536870911 : index
            vector.store %277, %142[%279] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %280 = vector.extract_strided_slice %126 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %281 = arith.addi %156, %100 overflow<nsw> : index
            %282 = arith.select %271, %281, %c536870911 : index
            vector.store %280, %142[%282] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %283 = vector.extract_strided_slice %126 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %284 = arith.addi %161, %100 overflow<nsw> : index
            %285 = arith.select %271, %284, %c536870911 : index
            vector.store %283, %142[%285] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %286 = vector.extract_strided_slice %126 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %287 = arith.addi %166, %100 overflow<nsw> : index
            %288 = arith.select %271, %287, %c536870911 : index
            vector.store %286, %142[%288] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %289 = vector.extract_strided_slice %126 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %290 = arith.addi %171, %100 overflow<nsw> : index
            %291 = arith.select %271, %290, %c536870911 : index
            vector.store %289, %142[%291] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %292 = vector.extract_strided_slice %126 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %293 = arith.addi %176, %100 overflow<nsw> : index
            %294 = arith.select %271, %293, %c536870911 : index
            vector.store %292, %142[%294] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %295 = vector.extract_strided_slice %126 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %296 = arith.addi %181, %100 overflow<nsw> : index
            %297 = arith.select %271, %296, %c536870911 : index
            vector.store %295, %142[%297] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %298 = vector.extract_strided_slice %126 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %299 = arith.addi %186, %100 overflow<nsw> : index
            %300 = arith.select %271, %299, %c536870911 : index
            vector.store %298, %142[%300] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %301 = vector.extract_strided_slice %126 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %302 = arith.addi %191, %100 overflow<nsw> : index
            %303 = arith.select %271, %302, %c536870911 : index
            vector.store %301, %142[%303] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %304 = vector.extract_strided_slice %126 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %305 = arith.addi %196, %100 overflow<nsw> : index
            %306 = arith.select %271, %305, %c536870911 : index
            vector.store %304, %142[%306] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %307 = vector.extract_strided_slice %126 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %308 = arith.addi %201, %100 overflow<nsw> : index
            %309 = arith.select %271, %308, %c536870911 : index
            vector.store %307, %142[%309] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %310 = vector.extract_strided_slice %126 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %311 = arith.addi %206, %100 overflow<nsw> : index
            %312 = arith.select %271, %311, %c536870911 : index
            vector.store %310, %142[%312] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %313 = vector.extract_strided_slice %126 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %314 = arith.addi %211, %100 overflow<nsw> : index
            %315 = arith.select %271, %314, %c536870911 : index
            vector.store %313, %142[%315] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %316 = vector.extract_strided_slice %126 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %317 = arith.addi %216, %100 overflow<nsw> : index
            %318 = arith.select %271, %317, %c536870911 : index
            vector.store %316, %142[%318] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %319 = vector.extract_strided_slice %130 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %320 = affine.apply #map47()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
            %321 = arith.cmpi slt, %320, %c68032 : index
            %322 = arith.addi %138, %105 overflow<nsw> : index
            %323 = arith.select %321, %322, %c536870911 : index
            vector.store %319, %142[%323] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %324 = vector.extract_strided_slice %130 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %325 = arith.addi %146, %105 overflow<nsw> : index
            %326 = arith.select %321, %325, %c536870911 : index
            vector.store %324, %142[%326] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %327 = vector.extract_strided_slice %130 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %328 = arith.addi %151, %105 overflow<nsw> : index
            %329 = arith.select %321, %328, %c536870911 : index
            vector.store %327, %142[%329] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %330 = vector.extract_strided_slice %130 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %331 = arith.addi %156, %105 overflow<nsw> : index
            %332 = arith.select %321, %331, %c536870911 : index
            vector.store %330, %142[%332] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %333 = vector.extract_strided_slice %130 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %334 = arith.addi %161, %105 overflow<nsw> : index
            %335 = arith.select %321, %334, %c536870911 : index
            vector.store %333, %142[%335] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %336 = vector.extract_strided_slice %130 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %337 = arith.addi %166, %105 overflow<nsw> : index
            %338 = arith.select %321, %337, %c536870911 : index
            vector.store %336, %142[%338] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %339 = vector.extract_strided_slice %130 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %340 = arith.addi %171, %105 overflow<nsw> : index
            %341 = arith.select %321, %340, %c536870911 : index
            vector.store %339, %142[%341] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %342 = vector.extract_strided_slice %130 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %343 = arith.addi %176, %105 overflow<nsw> : index
            %344 = arith.select %321, %343, %c536870911 : index
            vector.store %342, %142[%344] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %345 = vector.extract_strided_slice %130 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %346 = arith.addi %181, %105 overflow<nsw> : index
            %347 = arith.select %321, %346, %c536870911 : index
            vector.store %345, %142[%347] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %348 = vector.extract_strided_slice %130 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %349 = arith.addi %186, %105 overflow<nsw> : index
            %350 = arith.select %321, %349, %c536870911 : index
            vector.store %348, %142[%350] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %351 = vector.extract_strided_slice %130 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %352 = arith.addi %191, %105 overflow<nsw> : index
            %353 = arith.select %321, %352, %c536870911 : index
            vector.store %351, %142[%353] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %354 = vector.extract_strided_slice %130 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %355 = arith.addi %196, %105 overflow<nsw> : index
            %356 = arith.select %321, %355, %c536870911 : index
            vector.store %354, %142[%356] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %357 = vector.extract_strided_slice %130 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %358 = arith.addi %201, %105 overflow<nsw> : index
            %359 = arith.select %321, %358, %c536870911 : index
            vector.store %357, %142[%359] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %360 = vector.extract_strided_slice %130 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %361 = arith.addi %206, %105 overflow<nsw> : index
            %362 = arith.select %321, %361, %c536870911 : index
            vector.store %360, %142[%362] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %363 = vector.extract_strided_slice %130 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %364 = arith.addi %211, %105 overflow<nsw> : index
            %365 = arith.select %321, %364, %c536870911 : index
            vector.store %363, %142[%365] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            %366 = vector.extract_strided_slice %130 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            %367 = arith.addi %216, %105 overflow<nsw> : index
            %368 = arith.select %321, %367, %c536870911 : index
            vector.store %366, %142[%368] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
            return
        }
        }
    }
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x1024xf16>
        %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<68032x1024xf16>
        %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x68032xf32>
        %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<256x1024xf16>, tensor<68032x1024xf16>, tensor<256x68032xf32>) -> %2
        %4 = hal.tensor.barrier join(%3 : tensor<256x68032xf32>) => %arg4 : !hal.fence
        %5 = hal.tensor.export %4 : tensor<256x68032xf32> -> !hal.buffer_view
        return %5 : !hal.buffer_view
    }
    }
    """
   
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
        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, mfma_variant, tkl.bf16
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        use_buffer_ops=True,
        print_mlir=True,
        #override_mlir=asm,

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


def calculate_diff_gemm(M, N, K, dtype=torch.bfloat16):
    # Random test matrices
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(N, K, dtype=dtype, device="cuda")  # careful: ABᵀ → shape (M,N)
    C = torch.empty((M, N), dtype=torch.float32, device="cuda")

    # ---- WAVE ----
    # wave_kernel = get_wave_gemm((M,N,K), dtype, [False,False,False],MMAType.F32_32x32x16_K8_F16)  # <- your Wave GEMM builder
    # wave_kernel(A.clone(), B.clone(),C) 
    
    # ---- WAVE PIPELINED ----
    #warmup
    wave_gemm = testReorderedPingPongGemm((M,N,K), dtype, [False,False,False], MMAType.F32_32x32x16_F16)
    for i in range(209):
        wave_gemm(A,B,C) 

    wave_gemm(A,B,C) 

    # ---- TRITON ----
    output_triton = triton_matmul_abt(A.clone(), B.clone())  

    # ---- TORCH (reference, uses rocBLAS) ----
    # GEMM ABᵀ → (M,K) * (N,K)ᵀ = (M,N)
    #output_torch = torch.matmul(A, B.t())

    # ---- Compare ----
    print(f"Wave output shape:   {C.shape}")
    print(f"Triton output shape: {output_triton.shape}")
   # print(f"Torch output shape:  {output_torch.shape}")
    
    print("wave output:")
    print(C[:4, :16])
    print("ref output")
    print(output_triton[:4, :16])

    if torch.allclose(C, output_triton.to(torch.float32), atol=1e-2, rtol=1e-2):
        print("✅ Wave and Triton implementations match")
    else:
        print("❌ Wave and Triton implementations differ")
        max_diff = (C - output_triton.to(torch.float32)).abs().max().item()
        print(f"Max diff Wave vs Triton: {max_diff}")


# Pick a grid to match what you want to compare with Wave
# M_vals = [64,128,256]
# N_vals = [128 ,128,256]
# K_vals = [511,511,511 ]
# M_vals = [64]
# N_vals = [128]
# K_vals = [512]

# M_vals = [16384]
# N_vals = [32768]
# K_vals = [6144]

M_vals = [256]
N_vals = [68032]
K_vals = [1024]

# M_vals = [256]
# N_vals = [256]
# K_vals = [512]

configs = list(itertools.product(M_vals, N_vals, K_vals))

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        # line_vals=["wave", "reordered_gemm","wave_pipelined_16x16","wave_pipelined_32x32", "harsh_version_32x32x8","triton"],
        # line_names=["wave_16x16x16", "reordered_gemm","wave_pipelined_16x16_16","wave_pipelined_32x32_16","harsh_version_32x32x8","Triton"],
        line_vals=["reordered_gemm"],
        line_names=["reordered_gemm"],
        styles=[("blue","-"), ("red","-"),("green","-"),("orange","-"),("black","-"),("black","-")],
        ylabel="ms",
        plot_name="gemm-abt-performance",
        args={},
    )
)

def bench(M, N, K, provider):
    dtype = torch.float16
    dtype32=torch.float32
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
        C = torch.empty((M, N), dtype=dtype32, device="cuda")
        _ = wave_gemm(A, B, C)   # warmup; expect A(M,K), B(N,K), C(M,N)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: wave_gemm(A, B, C),
            quantiles=quantiles,
        )
        
    elif provider == "wave_pipelined_16x16":
        # plug your compiled wave GEMM here; it should compute C in fp32
        wave_gemm = get_wave_gemm_pipelined_16x16( (M,N,K), dtype, [False,False,False],MMAType.F32_32x32x16_K8_F16)
        C = torch.empty((M, N), dtype=dtype32, device="cuda")
        _ = wave_gemm(A, B, C)   # warmup; expect A(M,K), B(N,K), C(M,N)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: wave_gemm(A, B, C),
            quantiles=quantiles,
        )
    
    elif provider == "harsh_version_32x32x8":
        # plug your compiled wave GEMM here; it should compute C in fp32
        wave_gemm = get_wave_gemm_pipelined_harsh( (M,N,K), dtype, [False,False,False],MMAType.F32_32x32x16_K8_F16)
        C = torch.empty((M, N), dtype=dtype32, device="cuda")
        _ = wave_gemm(A, B, C)   # warmup; expect A(M,K), B(N,K), C(M,N)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: wave_gemm(A, B, C),
            quantiles=quantiles,
        )
    elif provider == "reordered_gemm":
        # plug your compiled wave GEMM here; it should compute C in fp32
        wave_gemm = testReorderedPingPongGemm( (M,N,K), dtype, [False,False,False], MMAType.F32_32x32x16_F16)
        C = torch.empty((M, N), dtype=dtype32, device="cuda")
        _ = wave_gemm(A, B, C)   # warmup; expect A(M,K), B(N,K), C(M,N)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: wave_gemm(A, B, C),
            quantiles=quantiles,
        )
    elif provider == "wave_pipelined_32x32":
        # plug your compiled wave GEMM here; it should compute C in fp32
        wave_gemm = get_wave_gemm_pipelined_32x32x16( (M,N,K), dtype, [False,False,False],MMAType.F32_32x32x16_K8_F16)
        C = torch.empty((M, N), dtype=dtype32, device="cuda")
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
    #bench.run(print_data=True, show_plots=False)
    
    calculate_diff_gemm(256, 68032, 1024, torch.bfloat16)

    #calculate_diff_gemm(256, 256, 512)
    #calculate_diff_gemm(64, 128, 511)
    #calculate_diff_gemm(16384, 32768, 6144)

    







