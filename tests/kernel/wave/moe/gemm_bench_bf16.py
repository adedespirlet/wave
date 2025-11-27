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
    C = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

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
   
    asm_bf16="""
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
        #map_wave_offset = affine_map<()[s0, s1] -> ((s1 * 4 + s0 floordiv 64) * 8)>
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
                %c64 = arith.constant 64 : index
                %c128 = arith.constant 128 : index
                %c192 = arith.constant 192 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 2
                %block_id_y = gpu.block_id  y upper_bound 266
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 1024], strides: [1024, 1] : memref<bf16> to memref<256x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_2 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [68032, 1024], strides: [1024, 1] : memref<bf16> to memref<68032x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_3 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 68032], strides: [68032, 1] : memref<f32> to memref<256x68032xf32, strided<[68032, 1], offset: ?>>
                %alloc = memref.alloc() : memref<52224xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c0][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<256x68xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c34816][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<128x68xbf16, #gpu.address_space<workgroup>>
                %tile4 = memref.subview %view_4[0, 0] [128, 64] [1, 1] : memref<128x68xbf16, #gpu.address_space<workgroup>> to memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>
                %tile = memref.subview %view[0, 0] [256, 64] [1, 1] : memref<256x68xbf16, #gpu.address_space<workgroup>> to memref<256x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>
        
                %3 = affine.apply #map()[%block_id_y, %block_id_x]
                %4 = arith.minsi %3, %c4 : index
                %5 = affine.apply #map1()[%block_id_y, %block_id_x, %4]
                %6 = arith.maxsi %5, %c1 : index
                %7 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
                %8 = affine.apply #map3()[%thread_id_x]
                %9 = arith.muli %7, %c1024 overflow<nsw> : index
                %10 = arith.addi %9, %8 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %11 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %66 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                
                %67 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                //%12 = vector.load %11[%10] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                //vector.store %12, %view_4[%66, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %wave_offset = affine.apply #map_wave_offset()[%thread_id_x, %thread_id_y]
                amdgpu.gather_to_lds %11[%10], %tile4[%wave_offset, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

                %13 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
                %14 = arith.muli %13, %c1024 overflow<nsw> : index
                %15 = arith.addi %14, %8 overflow<nsw> : index
                //%16 = vector.load %11[%15] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                //vector.store %16, %view_4[%67, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %wave_offset64= arith.addi %wave_offset, %c64 overflow<nsw> : index
                amdgpu.gather_to_lds %11[%15], %tile4[%wave_offset64, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

                %17 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
                %18 = arith.cmpi slt, %17, %c68032 : index
                %19 = vector.broadcast %18 : i1 to vector<8xi1>
                %20 = arith.muli %17, %c1024 overflow<nsw> : index
                %21 = arith.addi %20, %8 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_2 : memref<68032x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %22 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %23 = arith.index_cast %21 : index to i32
                %24 = vector.broadcast %23 : i32 to vector<8xi32>
                %25 = arith.addi %24, %cst : vector<8xi32>
                %26 = arith.index_cast %25 : vector<8xi32> to vector<8xindex>
                %27 = arith.select %19, %26, %cst_0 : vector<8xi1>, vector<8xindex>
                %28 = vector.extract %27[0] : index from vector<8xindex>
                %29 = vector.load %22[%28] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %68 = affine.apply #map11()[%thread_id_x, %thread_id_y]
                vector.store %29, %view[%68, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                //amdgpu.gather_to_lds %22[%28], %tile[%wave_offset, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

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
                %41 = vector.load %22[%40] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %69 = affine.apply #map12()[%thread_id_x, %thread_id_y]
                vector.store %41, %view[%69, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                //amdgpu.gather_to_lds %22[%40], %tile[%wave_offset64, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

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
                %53 = vector.load %22[%52] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %70 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                vector.store %53, %view[%70, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %wave_offset128 = arith.addi %wave_offset, %c128 overflow<nsw> : index
                //amdgpu.gather_to_lds %22[%52], %tile[%wave_offset128, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

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
                %65 = vector.load %22[%64] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %71 = affine.apply #map14()[%thread_id_x, %thread_id_y]
                vector.store %65, %view[%71, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %wave_offset192 = arith.addi %wave_offset, %c192 overflow<nsw> : index

                //amdgpu.gather_to_lds %22[%64], %tile[%wave_offset192, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>
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
                %369 = vector.load %view_4[%76, %77] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %370 = vector.load %view_4[%76, %78] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %371 = vector.load %view[%79, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %372 = vector.load %view[%79, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %373 = vector.load %view[%80, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %374 = vector.load %view[%80, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %375 = vector.load %view[%81, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %376 = vector.load %view[%81, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %377 = vector.load %view[%82, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %378 = vector.load %view[%82, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %379 = affine.apply #map25()[%arg3, %thread_id_x]
                %380 = arith.addi %14, %379 overflow<nsw> : index
                %381 = vector.load %11[%380] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %382 = arith.addi %9, %379 overflow<nsw> : index
                %383 = vector.load %11[%382] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %384 = vector.load %view_4[%76, %83] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %385 = vector.load %view_4[%76, %84] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %386 = vector.load %view[%79, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %387 = vector.load %view[%79, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %388 = vector.load %view[%80, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %389 = vector.load %view[%80, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %390 = vector.load %view[%81, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %391 = vector.load %view[%81, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %392 = vector.load %view[%82, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %393 = vector.load %view[%82, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %394 = arith.addi %33, %379 overflow<nsw> : index
                %395 = arith.index_cast %394 : index to i32
                %396 = vector.broadcast %395 : i32 to vector<8xi32>
                %397 = arith.addi %396, %cst : vector<8xi32>
                %398 = arith.index_cast %397 : vector<8xi32> to vector<8xindex>
                %399 = arith.select %32, %398, %cst_0 : vector<8xi1>, vector<8xindex>
                %400 = vector.extract %399[0] : index from vector<8xindex>
                %401 = vector.load %22[%400] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %402 = arith.addi %20, %379 overflow<nsw> : index
                %403 = arith.index_cast %402 : index to i32
                %404 = vector.broadcast %403 : i32 to vector<8xi32>
                %405 = arith.addi %404, %cst : vector<8xi32>
                %406 = arith.index_cast %405 : vector<8xi32> to vector<8xindex>
                %407 = arith.select %19, %406, %cst_0 : vector<8xi1>, vector<8xindex>
                %408 = vector.extract %407[0] : index from vector<8xindex>
                %409 = vector.load %22[%408] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %410 = arith.addi %57, %379 overflow<nsw> : index
                %411 = arith.index_cast %410 : index to i32
                %412 = vector.broadcast %411 : i32 to vector<8xi32>
                %413 = arith.addi %412, %cst : vector<8xi32>
                %414 = arith.index_cast %413 : vector<8xi32> to vector<8xindex>
                %415 = arith.select %56, %414, %cst_0 : vector<8xi1>, vector<8xindex>
                %416 = vector.extract %415[0] : index from vector<8xindex>
                %417 = vector.load %22[%416] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %418 = arith.addi %45, %379 overflow<nsw> : index
                %419 = arith.index_cast %418 : index to i32
                %420 = vector.broadcast %419 : i32 to vector<8xi32>
                %421 = arith.addi %420, %cst : vector<8xi32>
                %422 = arith.index_cast %421 : vector<8xi32> to vector<8xindex>
                %423 = arith.select %44, %422, %cst_0 : vector<8xi1>, vector<8xindex>
                %424 = vector.extract %423[0] : index from vector<8xindex>
                %425 = vector.load %22[%424] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %426 = amdgpu.mfma %369 * %371 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %427 = amdgpu.mfma %370 * %372 + %426 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %428 = amdgpu.mfma %369 * %373 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %429 = amdgpu.mfma %370 * %374 + %428 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %430 = amdgpu.mfma %369 * %375 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %431 = amdgpu.mfma %370 * %376 + %430 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %432 = amdgpu.mfma %369 * %377 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %433 = amdgpu.mfma %370 * %378 + %432 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                rocdl.s.setprio 0
                amdgpu.lds_barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                vector.store %381, %view_4[%67, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %383, %view_4[%66, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %425, %view[%70, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %409, %view[%68, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %417, %view[%71, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %401, %view[%69, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %434 = amdgpu.mfma %384 * %386 + %427 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %435 = amdgpu.mfma %385 * %387 + %434 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %436 = amdgpu.mfma %384 * %388 + %429 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %437 = amdgpu.mfma %385 * %389 + %436 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %438 = amdgpu.mfma %384 * %390 + %431 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %439 = amdgpu.mfma %385 * %391 + %438 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %440 = amdgpu.mfma %384 * %392 + %433 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %441 = amdgpu.mfma %385 * %393 + %440 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
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
                %88 = vector.load %view[%86, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %89 = affine.apply #map18()[%thread_id_x]
                %90 = vector.load %view[%86, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %91 = affine.apply #map23()[%thread_id_x]
                %92 = vector.load %view[%86, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %93 = affine.apply #map24()[%thread_id_x]
                %94 = vector.load %view[%86, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %95 = affine.apply #map20()[%thread_id_x, %thread_id_y]
                %96 = vector.load %view[%95, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %97 = vector.load %view[%95, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %98 = vector.load %view[%95, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %99 = vector.load %view[%95, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %100 = affine.apply #map21()[%thread_id_x, %thread_id_y]
                %101 = vector.load %view[%100, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %102 = vector.load %view[%100, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %103 = vector.load %view[%100, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %104 = vector.load %view[%100, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %105 = affine.apply #map22()[%thread_id_x, %thread_id_y]
                %106 = vector.load %view[%105, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %107 = vector.load %view[%105, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %108 = vector.load %view[%105, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %109 = vector.load %view[%105, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %110 = affine.apply #map16()[%thread_id_x]
                %111 = vector.load %view_4[%110, %87] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %112 = vector.load %view_4[%110, %89] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %113 = vector.load %view_4[%110, %91] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %114 = vector.load %view_4[%110, %93] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %115 = amdgpu.mfma %111 * %88 + %85#0 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %116 = amdgpu.mfma %112 * %90 + %115 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %117 = amdgpu.mfma %113 * %92 + %116 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %118 = amdgpu.mfma %114 * %94 + %117 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %119 = amdgpu.mfma %111 * %96 + %85#1 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %120 = amdgpu.mfma %112 * %97 + %119 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %121 = amdgpu.mfma %113 * %98 + %120 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %122 = amdgpu.mfma %114 * %99 + %121 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %123 = amdgpu.mfma %111 * %101 + %85#2 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %124 = amdgpu.mfma %112 * %102 + %123 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %125 = amdgpu.mfma %113 * %103 + %124 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %126 = amdgpu.mfma %114 * %104 + %125 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %127 = amdgpu.mfma %111 * %106 + %85#3 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %128 = amdgpu.mfma %112 * %107 + %127 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %129 = amdgpu.mfma %113 * %108 + %128 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %130 = amdgpu.mfma %114 * %109 + %129 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
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
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x1024xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<68032x1024xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x68032xf32>
            %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<256x1024xbf16>, tensor<68032x1024xbf16>, tensor<256x68032xf32>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x68032xf32>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x68032xf32> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
        """
    asm_debug= """
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
        #map_wave_offset = affine_map<()[s0, s1] -> ((s1 * 4 + s0 floordiv 64) * 8)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>
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
                %c64 = arith.constant 64 : index
                %c128 = arith.constant 128 : index
                %c192 = arith.constant 192 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 1
                %block_id_y = gpu.block_id  y upper_bound 1
                %thread_id_x = gpu.thread_id  x upper_bound 64
                %thread_id_y = gpu.thread_id  y upper_bound 1
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 1024], strides: [1024, 1] : memref<bf16> to memref<256x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_2 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [68032, 1024], strides: [1024, 1] : memref<bf16> to memref<68032x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_3 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 68032], strides: [68032, 1] : memref<f32> to memref<256x68032xf32, strided<[68032, 1], offset: ?>>
                %alloc = memref.alloc() : memref<52224xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c0][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<256x68xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c34816][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<128x68xbf16, #gpu.address_space<workgroup>>
                %tile4 = memref.subview %view_4[0, 0] [128, 64] [1, 1] : memref<128x68xbf16, #gpu.address_space<workgroup>> to memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>
        
                %3 = affine.apply #map()[%block_id_y, %block_id_x]
                %4 = arith.minsi %3, %c4 : index
                %5 = affine.apply #map1()[%block_id_y, %block_id_x, %4]
                %6 = arith.maxsi %5, %c1 : index
                %7 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
                %8 = affine.apply #map3()[%thread_id_x]
                %9 = arith.muli %7, %c1024 overflow<nsw> : index
                %10 = arith.addi %9, %8 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %11 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %66 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                
                %67 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                //%12 = vector.load %11[%10] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                //vector.store %12, %view_4[%66, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %wave_offset = affine.apply #map_wave_offset()[%thread_id_x, %thread_id_y]
                %c65 = arith.constant 65 : index
                amdgpu.gather_to_lds %11[%10], %tile4[%wave_offset, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

                //%13 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
               // %14 = arith.muli %13, %c1024 overflow<nsw> : index
                //%15 = arith.addi %14, %8 overflow<nsw> : index
                //%16 = vector.load %11[%15] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                //vector.store %16, %view_4[%67, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
               // %wave_offset64= arith.addi %wave_offset, %c64 overflow<nsw> : index
                //amdgpu.gather_to_lds %11[%15], %tile4[%wave_offset64, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

                  // Read back from LDS
                %loaded_from_lds = vector.load %view_4[%thread_id_x, %c0] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<68xbf16>
                
                // Reinterpret output as 2D: [128, 1024] 
                //%reinterpret_cast_out = memref.reinterpret_cast %1 to offset: [%c0], sizes: [128, 1024], strides: [1024, 1] : memref<bf16> to memref<128x1024xbf16, strided<[1024, 1], offset: ?>>
                
                // Write to global output (using wave_offset as row, column based on thread)
                vector.store %loaded_from_lds, %reinterpret_cast_2[%thread_id_x, %c0] : memref<68032x1024xbf16, strided<[1024, 1], offset: ?>>, vector<68xbf16>
                
                
               
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x1024xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<68032x1024xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x68032xf32>
            %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<256x1024xbf16>, tensor<68032x1024xbf16>, tensor<256x68032xf32>) -> %1
            %4 = hal.tensor.barrier join(%3 : tensor<68032x1024xbf16>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<68032x1024xbf16> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
    
    """
   
    asm_vectorstore="""
        #map = affine_map<()[s0, s1] -> (((s0 * 133 + s1 * 266 - ((s0 + s1 * 2) floordiv 8) * 1063) floordiv 8512) * -16 + 2)>
        #map1 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128 + ((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) floordiv 8512) * 2048 + (((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) mod 8512) mod s4) * 128)>
        #map2 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map3 = affine_map<()[s0] -> (s0 mod 8)>
        #map4 = affine_map<()[s0] -> (s0 * 8)>
        #map5 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 16) * 128)>
        #map6 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + ((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) floordiv 8512) * 2048 + (((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) mod 8512) mod s4) * 128 + 64)>
        #map7 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 16) * 128 + 64)>
        #map8 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128 + (((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) mod 8512) floordiv s4) * 128)>
        #map9 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + (((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) mod 8512) floordiv s4) * 128 + 64)>
        #map10 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 32) * 32)>
        #map11 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32)>
        #map12 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 2)>
        #map13 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 4)>
        #map14 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 6)>
        #map15 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 32) * 32 + 32)>
        #map16 = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
        #map17 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 + 64)>
        #map18 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 mod 32 + s4 * 64 + (((s1 * 133 + s2 * 266 - ((s1 + s2 * 2) floordiv 8) * 1063) mod 8512) floordiv s3) * 128)>
        #map19 = affine_map<()[s0, s1, s2] -> (((s0 * 133 + s1 * 266 - ((s0 + s1 * 2) floordiv 8) * 1063) floordiv 8512) * 2048 + (((s0 * 133 + s1 * 266 - ((s0 + s1 * 2) floordiv 8) * 1063) mod 8512) mod s2) * 128)>
        #map20 = affine_map<()[s0, s1, s2] -> ((((s0 * 133 + s1 * 266 - ((s0 + s1 * 2) floordiv 8) * 1063) mod 8512) floordiv s2) * 128)>
        #map21 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
        #map22 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
        #map23 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
        #map24 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
        #map25 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
        #map26 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
        #map27 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
        #map28 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
        #map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
        #map30 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
        #map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
        #map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
        #map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
        #map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
        #map35 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
        #intermediaterow = affine_map<()[s0, s1] -> (s0 floordiv 4 + s1 * 64)>
        #intermediatecol = affine_map<()[s0, s1] -> ((s0 mod 4) * 32)>
        #map37 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 mod 32 + s4 * 64 + (((s1 * 133 + s2 * 266 - ((s1 + s2 * 2) floordiv 8) * 1063) mod 8512) floordiv s3) * 128 + 32)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c2 = arith.constant 2 : index
            %c532 = arith.constant 532 : index
            %c1 = arith.constant 1 : index
            stream.return %c2, %c532, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c1024_i14 = arith.constant 1024 : i14
                %c0_i32 = arith.constant 0 : i32
                %c15 = arith.constant 15 : index
                %c1073741823 = arith.constant 1073741823 : index
                %c68032 = arith.constant 68032 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c1024 = arith.constant 1024 : index
                %c1 = arith.constant 1 : index
                %c73728 = arith.constant 73728 : index
                %c49152 = arith.constant 49152 : index
                %c24576 = arith.constant 24576 : index
                %c98304 = arith.constant 98304 : index
                %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<bf16>
                %block_id_x = gpu.block_id  x upper_bound 2
                %block_id_y = gpu.block_id  y upper_bound 532
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 1024], strides: [1024, 1] : memref<bf16> to memref<256x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [68032, 1024], strides: [1024, 1] : memref<bf16> to memref<68032x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 68032], strides: [68032, 1] : memref<bf16> to memref<256x68032xbf16, strided<[68032, 1], offset: ?>>
                %alloc = memref.alloc() : memref<131584xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c0][] : memref<131584xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %view_2 = memref.view %alloc[%c24576][] : memref<131584xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %view_3 = memref.view %alloc[%c49152][] : memref<131584xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c73728][] : memref<131584xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %shared_output = memref.view %alloc[%c98304][] : memref<131584xi8, #gpu.address_space<workgroup>> to memref<128x130xbf16, #gpu.address_space<workgroup>>

                %3 = affine.apply #map()[%block_id_x, %block_id_y]
                %4 = arith.maxsi %3, %c1 : index
                %5 = affine.apply #map1()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %6 = affine.apply #map2()[%thread_id_x]
                %7 = affine.apply #map3()[%thread_id_x]
                %8 = arith.xori %7, %6 : index
                %9 = affine.apply #map4()[%8]
                %10 = affine.apply #map5()[%thread_id_x, %thread_id_y]
                %11 = arith.index_cast %10 : index to i32
                %12 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%11) : (i32) -> i32
                %13 = arith.index_cast %12 : i32 to index
                %14 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %15 = arith.index_cast %14 : i32 to index
                %16 = arith.muli %5, %c1024 overflow<nsw> : index
                %17 = arith.addi %16, %9 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %18 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %18[%17], %view_4[%13, %15] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %19 = affine.apply #map6()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %20 = affine.apply #map7()[%thread_id_x, %thread_id_y]
                %21 = arith.index_cast %20 : index to i32
                %22 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%21) : (i32) -> i32
                %23 = arith.index_cast %22 : i32 to index
                %24 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %25 = arith.index_cast %24 : i32 to index
                %26 = arith.muli %19, %c1024 overflow<nsw> : index
                %27 = arith.addi %26, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%27], %view_4[%23, %25] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %28 = affine.apply #map8()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %29 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%11) : (i32) -> i32
                %30 = arith.index_cast %29 : i32 to index
                %31 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %32 = arith.index_cast %31 : i32 to index
                %33 = arith.muli %28, %c1024 overflow<nsw> : index
                %34 = arith.addi %33, %9 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_0 : memref<68032x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %35 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %36 = arith.cmpi slt, %28, %c68032 : index
                %37 = arith.select %36, %34, %c1073741823 : index
                amdgpu.gather_to_lds %35[%37], %view_2[%30, %32] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %38 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %39 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%21) : (i32) -> i32
                %40 = arith.index_cast %39 : i32 to index
                %41 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %42 = arith.index_cast %41 : i32 to index
                %43 = arith.muli %38, %c1024 overflow<nsw> : index
                %44 = arith.addi %43, %9 overflow<nsw> : index
                %45 = arith.cmpi slt, %38, %c68032 : index
                %46 = arith.select %45, %44, %c1073741823 : index
                amdgpu.gather_to_lds %35[%46], %view_2[%40, %42] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %47 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%11) : (i32) -> i32
                %48 = arith.index_cast %47 : i32 to index
                %49 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %50 = arith.index_cast %49 : i32 to index
                %51 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%21) : (i32) -> i32
                %52 = arith.index_cast %51 : i32 to index
                %53 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %54 = arith.index_cast %53 : i32 to index
                %55 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%11) : (i32) -> i32
                %56 = arith.index_cast %55 : i32 to index
                %57 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %58 = arith.index_cast %57 : i32 to index
                %59 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%21) : (i32) -> i32
                %60 = arith.index_cast %59 : i32 to index
                %61 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %62 = arith.index_cast %61 : i32 to index
                %63 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %64 = affine.apply #map11()[%thread_id_x]
                %65 = arith.xori %64, %7 : index
                %66 = affine.apply #map4()[%65]
                %67 = affine.apply #map12()[%thread_id_x]
                %68 = arith.xori %67, %7 : index
                %69 = affine.apply #map4()[%68]
                %70 = affine.apply #map13()[%thread_id_x]
                %71 = arith.xori %70, %7 : index
                %72 = affine.apply #map4()[%71]
                %73 = affine.apply #map14()[%thread_id_x]
                %74 = arith.xori %73, %7 : index
                %75 = affine.apply #map4()[%74]
                %76 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %77 = affine.apply #map16()[%thread_id_x]
                %78:6 = scf.for %arg3 = %c0 to %c15 step %c1 iter_args(%arg4 = %cst, %arg5 = %cst, %arg6 = %view_4, %arg7 = %view_3, %arg8 = %view_2, %arg9 = %view) -> (vector<16xf32>, vector<16xf32>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>) {
                rocdl.s.waitcnt 16368
                amdgpu.lds_barrier
                %284 = vector.load %arg8[%63, %66] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %285 = vector.load %arg8[%63, %69] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %286 = vector.load %arg8[%63, %72] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %287 = vector.load %arg8[%63, %75] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %288 = vector.load %arg8[%76, %66] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %289 = vector.load %arg8[%76, %69] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %290 = vector.load %arg8[%76, %72] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %291 = vector.load %arg8[%76, %75] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %292 = vector.load %arg6[%77, %66] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %293 = vector.load %arg6[%77, %69] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %294 = vector.load %arg6[%77, %72] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %295 = vector.load %arg6[%77, %75] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                amdgpu.lds_barrier
                %296 = affine.apply #map17()[%arg3, %8]
                %297 = arith.addi %16, %296 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%297], %arg7[%48, %50] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %298 = arith.addi %26, %296 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%298], %arg7[%52, %54] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %299 = arith.addi %33, %296 overflow<nsw> : index
                %300 = arith.select %36, %299, %c1073741823 : index
                amdgpu.gather_to_lds %35[%300], %arg9[%56, %58] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %301 = arith.addi %43, %296 overflow<nsw> : index
                %302 = arith.select %45, %301, %c1073741823 : index
                amdgpu.gather_to_lds %35[%302], %arg9[%60, %62] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %303 = amdgpu.mfma %292 * %284 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %304 = amdgpu.mfma %293 * %285 + %303 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %305 = amdgpu.mfma %294 * %286 + %304 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %306 = amdgpu.mfma %295 * %287 + %305 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %307 = amdgpu.mfma %292 * %288 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %308 = amdgpu.mfma %293 * %289 + %307 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %309 = amdgpu.mfma %294 * %290 + %308 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %310 = amdgpu.mfma %295 * %291 + %309 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                scf.yield %306, %310, %arg7, %arg6, %arg9, %arg8 : vector<16xf32>, vector<16xf32>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                }
                rocdl.s.waitcnt 16368
                amdgpu.lds_barrier
                %79 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %80 = affine.apply #map11()[%thread_id_x]
                %81 = arith.xori %80, %7 : index
                %82 = affine.apply #map4()[%81]
                %83 = vector.load %78#4[%79, %82] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %84 = affine.apply #map12()[%thread_id_x]
                %85 = arith.xori %84, %7 : index
                %86 = affine.apply #map4()[%85]
                %87 = vector.load %78#4[%79, %86] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %88 = affine.apply #map13()[%thread_id_x]
                %89 = arith.xori %88, %7 : index
                %90 = affine.apply #map4()[%89]
                %91 = vector.load %78#4[%79, %90] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %92 = affine.apply #map14()[%thread_id_x]
                %93 = arith.xori %92, %7 : index
                %94 = affine.apply #map4()[%93]
                %95 = vector.load %78#4[%79, %94] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %96 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %97 = vector.load %78#4[%96, %82] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %98 = vector.load %78#4[%96, %86] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %99 = vector.load %78#4[%96, %90] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %100 = vector.load %78#4[%96, %94] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %101 = affine.apply #map16()[%thread_id_x]
                %102 = vector.load %78#2[%101, %82] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %103 = vector.load %78#2[%101, %86] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %104 = vector.load %78#2[%101, %90] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %105 = vector.load %78#2[%101, %94] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %106 = amdgpu.mfma %102 * %83 + %78#0 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %107 = amdgpu.mfma %103 * %87 + %106 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %108 = amdgpu.mfma %104 * %91 + %107 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %109 = amdgpu.mfma %105 * %95 + %108 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %110 = amdgpu.mfma %102 * %97 + %78#1 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %111 = amdgpu.mfma %103 * %98 + %110 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %112 = amdgpu.mfma %104 * %99 + %111 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %113 = amdgpu.mfma %105 * %100 + %112 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %114 = vector.extract_strided_slice %109 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %115 = arith.truncf %114 : vector<1xf32> to vector<1xbf16>
                %116 = affine.apply #map18()[%thread_id_x, %block_id_x, %block_id_y, %4, %thread_id_y]
                %117 = arith.cmpi slt, %116, %c68032 : index
                %118 = affine.apply #map19()[%block_id_x, %block_id_y, %4]
                %119 = affine.apply #map20()[%block_id_x, %block_id_y, %4]
                %120 = affine.apply #map21()[%thread_id_x]
                %121 = arith.muli %118, %c68032 overflow<nsw> : index
                %122 = arith.muli %120, %c68032 overflow<nsw> : index
                %123 = arith.addi %121, %119 overflow<nsw> : index
                %124 = arith.addi %122, %79 overflow<nsw> : index
                %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %reinterpret_cast_1 : memref<256x68032xbf16, strided<[68032, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %125 = arith.addi %123, %offset_12 overflow<nsw> : index
                %reinterpret_cast_15 = memref.reinterpret_cast %2 to offset: [%125], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %126 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_15 validBytes(%c2147483645_i64) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
             
                vector.store %115, %shared_output[%120, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 1
                %128 = vector.extract_strided_slice %109 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %129 = arith.truncf %128 : vector<1xf32> to vector<1xbf16>
                %130 = affine.apply #map22()[%thread_id_x]
                vector.store %129, %shared_output[%130, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 2
                %134 = vector.extract_strided_slice %109 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %135 = arith.truncf %134 : vector<1xf32> to vector<1xbf16>
                %136 = affine.apply #map23()[%thread_id_x]
                vector.store %135, %shared_output[%136, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 3
                %140 = vector.extract_strided_slice %109 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %141 = arith.truncf %140 : vector<1xf32> to vector<1xbf16>
                %142 = affine.apply #map24()[%thread_id_x]
                vector.store %141, %shared_output[%142, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 4
                %146 = vector.extract_strided_slice %109 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %147 = arith.truncf %146 : vector<1xf32> to vector<1xbf16>
                %148 = affine.apply #map25()[%thread_id_x]
                vector.store %147, %shared_output[%148, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 5
                %152 = vector.extract_strided_slice %109 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %153 = arith.truncf %152 : vector<1xf32> to vector<1xbf16>
                %154 = affine.apply #map26()[%thread_id_x]
                vector.store %153, %shared_output[%154, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 6
                %158 = vector.extract_strided_slice %109 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %159 = arith.truncf %158 : vector<1xf32> to vector<1xbf16>
                %160 = affine.apply #map27()[%thread_id_x]
                vector.store %159, %shared_output[%160, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 7
                %164 = vector.extract_strided_slice %109 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %165 = arith.truncf %164 : vector<1xf32> to vector<1xbf16>
                %166 = affine.apply #map28()[%thread_id_x]
                vector.store %165, %shared_output[%166, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 8
                %170 = vector.extract_strided_slice %109 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %171 = arith.truncf %170 : vector<1xf32> to vector<1xbf16>
                %172 = affine.apply #map29()[%thread_id_x]
                vector.store %171, %shared_output[%172, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 9
                %176 = vector.extract_strided_slice %109 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %177 = arith.truncf %176 : vector<1xf32> to vector<1xbf16>
                %178 = affine.apply #map30()[%thread_id_x]
                vector.store %177, %shared_output[%178, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 10
                %182 = vector.extract_strided_slice %109 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %183 = arith.truncf %182 : vector<1xf32> to vector<1xbf16>
                %184 = affine.apply #map31()[%thread_id_x]
                vector.store %183, %shared_output[%184, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 11
                %188 = vector.extract_strided_slice %109 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %189 = arith.truncf %188 : vector<1xf32> to vector<1xbf16>
                %190 = affine.apply #map32()[%thread_id_x]
                vector.store %189, %shared_output[%190, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 12
                %194 = vector.extract_strided_slice %109 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %195 = arith.truncf %194 : vector<1xf32> to vector<1xbf16>
                %196 = affine.apply #map33()[%thread_id_x]
                vector.store %195, %shared_output[%196, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 13
                %200 = vector.extract_strided_slice %109 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %201 = arith.truncf %200 : vector<1xf32> to vector<1xbf16>
                %202 = affine.apply #map34()[%thread_id_x]
                vector.store %201, %shared_output[%202, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 14
                %206 = vector.extract_strided_slice %109 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %207 = arith.truncf %206 : vector<1xf32> to vector<1xbf16>
                %208 = affine.apply #map35()[%thread_id_x]
                vector.store %207, %shared_output[%208, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 15
                %212 = vector.extract_strided_slice %109 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %213 = arith.truncf %212 : vector<1xf32> to vector<1xbf16>
                %214 = affine.apply #map36()[%thread_id_x]
                vector.store %213, %shared_output[%214, %79] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %218 = vector.extract_strided_slice %113 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %219 = arith.truncf %218 : vector<1xf32> to vector<1xbf16>
                vector.store %219, %shared_output[%120, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %224 = vector.extract_strided_slice %113 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %225 = arith.truncf %224 : vector<1xf32> to vector<1xbf16>
                vector.store %225, %shared_output[%130, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %228 = vector.extract_strided_slice %113 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %229 = arith.truncf %228 : vector<1xf32> to vector<1xbf16>
                vector.store %229, %shared_output[%136, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %232 = vector.extract_strided_slice %113 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %233 = arith.truncf %232 : vector<1xf32> to vector<1xbf16>
                vector.store %233, %shared_output[%142, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %236 = vector.extract_strided_slice %113 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %237 = arith.truncf %236 : vector<1xf32> to vector<1xbf16>
                vector.store %237, %shared_output[%148, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %240 = vector.extract_strided_slice %113 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %241 = arith.truncf %240 : vector<1xf32> to vector<1xbf16>
                vector.store %241, %shared_output[%154, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %244 = vector.extract_strided_slice %113 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %245 = arith.truncf %244 : vector<1xf32> to vector<1xbf16>
                vector.store %245, %shared_output[%160, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %248 = vector.extract_strided_slice %113 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %249 = arith.truncf %248 : vector<1xf32> to vector<1xbf16>
                vector.store %249, %shared_output[%166, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %252 = vector.extract_strided_slice %113 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %253 = arith.truncf %252 : vector<1xf32> to vector<1xbf16>
                vector.store %253, %shared_output[%172, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %256 = vector.extract_strided_slice %113 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %257 = arith.truncf %256 : vector<1xf32> to vector<1xbf16>
                vector.store %257, %shared_output[%178, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %260 = vector.extract_strided_slice %113 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %261 = arith.truncf %260 : vector<1xf32> to vector<1xbf16>
                vector.store %261, %shared_output[%184, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %264 = vector.extract_strided_slice %113 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %265 = arith.truncf %264 : vector<1xf32> to vector<1xbf16>
                vector.store %265, %shared_output[%190, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %268 = vector.extract_strided_slice %113 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %269 = arith.truncf %268 : vector<1xf32> to vector<1xbf16>
                vector.store %269, %shared_output[%196, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %272 = vector.extract_strided_slice %113 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %273 = arith.truncf %272 : vector<1xf32> to vector<1xbf16>
                vector.store %273, %shared_output[%202, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %276 = vector.extract_strided_slice %113 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %277 = arith.truncf %276 : vector<1xf32> to vector<1xbf16>
                vector.store %277, %shared_output[%208, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %280 = vector.extract_strided_slice %113 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %281 = arith.truncf %280 : vector<1xf32> to vector<1xbf16>
                vector.store %281, %shared_output[%214, %96] : memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>
                
               

                %rowint = affine.apply #intermediaterow()[%thread_id_x, %thread_id_y]
                %colint = affine.apply #intermediatecol()[%thread_id_x, %thread_id_y]


                %data = vector.load %shared_output[%rowint, %colint]: memref<128x130xbf16, #gpu.address_space<workgroup>>, vector<32xbf16>
                
                %globalrow = arith.muli %rowint, %c68032 overflow<nsw> : index
                %globalindex = arith.addi %globalrow, %colint overflow<nsw> : index

                %idx_in_view = arith.select %117, %globalindex, %c1073741823 : index

                vector.store %data, %126[%idx_in_view] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<32xbf16>

                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x1024xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<68032x1024xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x68032xbf16>
            %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<256x1024xbf16>, tensor<68032x1024xbf16>, tensor<256x68032xbf16>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x68032xbf16>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x68032xbf16> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
        """
   
    asm_nogloballds= """
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
        #intermediaterow = affine_map<()[s0, s1] -> (s0 floordiv 4 + s1 * 64)>
        #intermediatecol = affine_map<()[s0, s1] -> ((s0 mod 4) * 64)>
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
                %c0_i32 = arith.constant 0 : i32
                %c15 = arith.constant 15 : index
                %c1073741823 = arith.constant 1073741823 : index
                %c68032 = arith.constant 68032 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c1024 = arith.constant 1024 : index
                %c1 = arith.constant 1 : index
                %c4 = arith.constant 4 : index
                %c34816 = arith.constant 34816 : index
                %cst_1 = arith.constant dense<0.000000e+00> : vector<16xf32>
                %c0 = arith.constant 0 : index
                %c52224 = arith.constant 52224 : index

                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<bf16>
                %block_id_x = gpu.block_id  x upper_bound 2
                %block_id_y = gpu.block_id  y upper_bound 266
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 1024], strides: [1024, 1] : memref<bf16> to memref<256x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_2 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [68032, 1024], strides: [1024, 1] : memref<bf16> to memref<68032x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_3 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 68032], strides: [68032, 1] : memref<bf16> to memref<256x68032xbf16, strided<[68032, 1], offset: ?>>
                %alloc = memref.alloc() : memref<118784xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c0][] : memref<118784xi8, #gpu.address_space<workgroup>> to memref<256x68xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c34816][] : memref<118784xi8, #gpu.address_space<workgroup>> to memref<128x68xbf16, #gpu.address_space<workgroup>>
                %shared_output = memref.view %alloc[%c52224][] : memref<118784xi8, #gpu.address_space<workgroup>> to memref<128x260xbf16, #gpu.address_space<workgroup>>

                %3 = affine.apply #map()[%block_id_y, %block_id_x]
                %4 = arith.minsi %3, %c4 : index
                %5 = affine.apply #map1()[%block_id_y, %block_id_x, %4]
                %6 = arith.maxsi %5, %c1 : index
                %7 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
                %8 = affine.apply #map3()[%thread_id_x]
                %9 = arith.muli %7, %c1024 overflow<nsw> : index
                %10 = arith.addi %9, %8 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %11 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %12 = vector.load %11[%10] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %13 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
                %14 = arith.muli %13, %c1024 overflow<nsw> : index
                %15 = arith.addi %14, %8 overflow<nsw> : index
                %16 = vector.load %11[%15] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %17 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
                %18 = arith.cmpi slt, %17, %c68032 : index
                %19 = vector.broadcast %18 : i1 to vector<8xi1>
                %20 = arith.muli %17, %c1024 overflow<nsw> : index
                %21 = arith.addi %20, %8 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_2 : memref<68032x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %22 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %23 = arith.index_cast %21 : index to i32
                %24 = vector.broadcast %23 : i32 to vector<8xi32>
                %25 = arith.addi %24, %cst : vector<8xi32>
                %26 = arith.index_cast %25 : vector<8xi32> to vector<8xindex>
                %27 = arith.select %19, %26, %cst_0 : vector<8xi1>, vector<8xindex>
                %28 = vector.extract %27[0] : index from vector<8xindex>
                %29 = vector.load %22[%28] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
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
                %41 = vector.load %22[%40] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
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
                %53 = vector.load %22[%52] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
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
                %65 = vector.load %22[%64] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %66 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                vector.store %12, %view_4[%66, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %67 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                vector.store %16, %view_4[%67, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %68 = affine.apply #map11()[%thread_id_x, %thread_id_y]
                vector.store %29, %view[%68, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %69 = affine.apply #map12()[%thread_id_x, %thread_id_y]
                vector.store %41, %view[%69, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %70 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                vector.store %53, %view[%70, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %71 = affine.apply #map14()[%thread_id_x, %thread_id_y]
                vector.store %65, %view[%71, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
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
                %433 = vector.load %view_4[%76, %77] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %434 = vector.load %view_4[%76, %78] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %435 = vector.load %view[%79, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %436 = vector.load %view[%79, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %437 = vector.load %view[%80, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %438 = vector.load %view[%80, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %439 = vector.load %view[%81, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %440 = vector.load %view[%81, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %441 = vector.load %view[%82, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %442 = vector.load %view[%82, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %443 = affine.apply #map25()[%arg3, %thread_id_x]
                %444 = arith.addi %14, %443 overflow<nsw> : index
                %445 = vector.load %11[%444] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %446 = arith.addi %9, %443 overflow<nsw> : index
                %447 = vector.load %11[%446] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %448 = vector.load %view_4[%76, %83] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %449 = vector.load %view_4[%76, %84] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %450 = vector.load %view[%79, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %451 = vector.load %view[%79, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %452 = vector.load %view[%80, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %453 = vector.load %view[%80, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %454 = vector.load %view[%81, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %455 = vector.load %view[%81, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %456 = vector.load %view[%82, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %457 = vector.load %view[%82, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %458 = arith.addi %33, %443 overflow<nsw> : index
                %459 = arith.index_cast %458 : index to i32
                %460 = vector.broadcast %459 : i32 to vector<8xi32>
                %461 = arith.addi %460, %cst : vector<8xi32>
                %462 = arith.index_cast %461 : vector<8xi32> to vector<8xindex>
                %463 = arith.select %32, %462, %cst_0 : vector<8xi1>, vector<8xindex>
                %464 = vector.extract %463[0] : index from vector<8xindex>
                %465 = vector.load %22[%464] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %466 = arith.addi %20, %443 overflow<nsw> : index
                %467 = arith.index_cast %466 : index to i32
                %468 = vector.broadcast %467 : i32 to vector<8xi32>
                %469 = arith.addi %468, %cst : vector<8xi32>
                %470 = arith.index_cast %469 : vector<8xi32> to vector<8xindex>
                %471 = arith.select %19, %470, %cst_0 : vector<8xi1>, vector<8xindex>
                %472 = vector.extract %471[0] : index from vector<8xindex>
                %473 = vector.load %22[%472] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %474 = arith.addi %57, %443 overflow<nsw> : index
                %475 = arith.index_cast %474 : index to i32
                %476 = vector.broadcast %475 : i32 to vector<8xi32>
                %477 = arith.addi %476, %cst : vector<8xi32>
                %478 = arith.index_cast %477 : vector<8xi32> to vector<8xindex>
                %479 = arith.select %56, %478, %cst_0 : vector<8xi1>, vector<8xindex>
                %480 = vector.extract %479[0] : index from vector<8xindex>
                %481 = vector.load %22[%480] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %482 = arith.addi %45, %443 overflow<nsw> : index
                %483 = arith.index_cast %482 : index to i32
                %484 = vector.broadcast %483 : i32 to vector<8xi32>
                %485 = arith.addi %484, %cst : vector<8xi32>
                %486 = arith.index_cast %485 : vector<8xi32> to vector<8xindex>
                %487 = arith.select %44, %486, %cst_0 : vector<8xi1>, vector<8xindex>
                %488 = vector.extract %487[0] : index from vector<8xindex>
                %489 = vector.load %22[%488] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %490 = amdgpu.mfma %433 * %435 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %491 = amdgpu.mfma %434 * %436 + %490 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %492 = amdgpu.mfma %433 * %437 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %493 = amdgpu.mfma %434 * %438 + %492 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %494 = amdgpu.mfma %433 * %439 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %495 = amdgpu.mfma %434 * %440 + %494 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %496 = amdgpu.mfma %433 * %441 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %497 = amdgpu.mfma %434 * %442 + %496 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                rocdl.s.setprio 0
                amdgpu.lds_barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                vector.store %445, %view_4[%67, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %447, %view_4[%66, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %481, %view[%71, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %465, %view[%69, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %489, %view[%70, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %473, %view[%68, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %498 = amdgpu.mfma %448 * %450 + %491 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %499 = amdgpu.mfma %449 * %451 + %498 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %500 = amdgpu.mfma %448 * %452 + %493 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %501 = amdgpu.mfma %449 * %453 + %500 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %502 = amdgpu.mfma %448 * %454 + %495 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %503 = amdgpu.mfma %449 * %455 + %502 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %504 = amdgpu.mfma %448 * %456 + %497 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %505 = amdgpu.mfma %449 * %457 + %504 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                rocdl.s.setprio 0
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                amdgpu.lds_barrier
                scf.yield %499, %501, %503, %505 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
                }
                scf.if %75 {
                rocdl.s.barrier
                }
                %86 = affine.apply #map19()[%thread_id_x, %thread_id_y]
                %87 = affine.apply #map17()[%thread_id_x]
                %88 = vector.load %view[%86, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %89 = affine.apply #map18()[%thread_id_x]
                %90 = vector.load %view[%86, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %91 = affine.apply #map23()[%thread_id_x]
                %92 = vector.load %view[%86, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %93 = affine.apply #map24()[%thread_id_x]
                %94 = vector.load %view[%86, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %95 = affine.apply #map20()[%thread_id_x, %thread_id_y]
                %96 = vector.load %view[%95, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %97 = vector.load %view[%95, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %98 = vector.load %view[%95, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %99 = vector.load %view[%95, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %100 = affine.apply #map21()[%thread_id_x, %thread_id_y]
                %101 = vector.load %view[%100, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %102 = vector.load %view[%100, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %103 = vector.load %view[%100, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %104 = vector.load %view[%100, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %105 = affine.apply #map22()[%thread_id_x, %thread_id_y]
                %106 = vector.load %view[%105, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %107 = vector.load %view[%105, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %108 = vector.load %view[%105, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %109 = vector.load %view[%105, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %110 = affine.apply #map16()[%thread_id_x]
                %111 = vector.load %view_4[%110, %87] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %112 = vector.load %view_4[%110, %89] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %113 = vector.load %view_4[%110, %91] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %114 = vector.load %view_4[%110, %93] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %115 = amdgpu.mfma %111 * %88 + %85#0 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %116 = amdgpu.mfma %112 * %90 + %115 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %117 = amdgpu.mfma %113 * %92 + %116 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %118 = amdgpu.mfma %114 * %94 + %117 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %119 = amdgpu.mfma %111 * %96 + %85#1 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %120 = amdgpu.mfma %112 * %97 + %119 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %121 = amdgpu.mfma %113 * %98 + %120 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %122 = amdgpu.mfma %114 * %99 + %121 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %123 = amdgpu.mfma %111 * %101 + %85#2 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %124 = amdgpu.mfma %112 * %102 + %123 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %125 = amdgpu.mfma %113 * %103 + %124 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %126 = amdgpu.mfma %114 * %104 + %125 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %127 = amdgpu.mfma %111 * %106 + %85#3 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %128 = amdgpu.mfma %112 * %107 + %127 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %129 = amdgpu.mfma %113 * %108 + %128 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %130 = amdgpu.mfma %114 * %109 + %129 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %131 = vector.extract_strided_slice %118 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %132 = arith.truncf %131 : vector<1xf32> to vector<1xbf16>
                %133 = affine.apply #map26()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %134 = arith.cmpi slt, %133, %c68032 : index
                %135 = affine.apply #map27()[%block_id_y, %block_id_x, %4, %4, %6]
                %136 = affine.apply #map28()[%block_id_x, %block_id_y, %4, %6]
                %137 = affine.apply #map29()[%thread_id_x]
                %138 = arith.muli %135, %c68032 overflow<nsw> : index
                %139 = arith.muli %137, %c68032 overflow<nsw> : index
                %140 = arith.addi %138, %136 overflow<nsw> : index
                %141 = arith.addi %139, %86 overflow<nsw> : index
                %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %reinterpret_cast_3 : memref<256x68032xbf16, strided<[68032, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %142 = arith.addi %140, %offset_12 overflow<nsw> : index
                %reinterpret_cast_15 = memref.reinterpret_cast %2 to offset: [%142], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %143 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_15 validBytes(%c2147483645_i64) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %144 = arith.select %134, %141, %c1073741823 : index
                
                vector.store %132, %shared_output[%137, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 1
                %145 = vector.extract_strided_slice %118 {offsets = [1], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %146 = arith.truncf %145 : vector<1xf32> to vector<1xbf16>
                %147 = affine.apply #map30()[%thread_id_x]

                // NEW: store slice 1 to LDS
                vector.store %146, %shared_output[%147, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %148 = arith.muli %147, %c68032 overflow<nsw> : index
                %149 = arith.addi %148, %86 overflow<nsw> : index
                %150 = arith.select %134, %149, %c1073741823 : index
                vector.store %146, %143[%150]
                : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>

                // Slice 2
                %151 = vector.extract_strided_slice %118 {offsets = [2], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %152 = arith.truncf %151 : vector<1xf32> to vector<1xbf16>
                %153 = affine.apply #map31()[%thread_id_x]

                // NEW: store slice 2 to LDS
                vector.store %152, %shared_output[%153, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %154 = arith.muli %153, %c68032 overflow<nsw> : index
                %155 = arith.addi %154, %86 overflow<nsw> : index
                %156 = arith.select %134, %155, %c1073741823 : index
                vector.store %152, %143[%156]
                : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>

                // Slice 3
                %157 = vector.extract_strided_slice %118 {offsets = [3], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %158 = arith.truncf %157 : vector<1xf32> to vector<1xbf16>
                %159 = affine.apply #map32()[%thread_id_x]

                // NEW: store slice 3 to LDS
                vector.store %158, %shared_output[%159, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %160 = arith.muli %159, %c68032 overflow<nsw> : index
                %161 = arith.addi %160, %86 overflow<nsw> : index
                %162 = arith.select %134, %161, %c1073741823 : index
                vector.store %158, %143[%162]
                : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>

                // Slice 4
                %163 = vector.extract_strided_slice %118 {offsets = [4], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %164 = arith.truncf %163 : vector<1xf32> to vector<1xbf16>
                %165 = affine.apply #map33()[%thread_id_x]

                // NEW: store slice 4 to LDS
                vector.store %164, %shared_output[%165, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %166 = arith.muli %165, %c68032 overflow<nsw> : index
                %167 = arith.addi %166, %86 overflow<nsw> : index
                %168 = arith.select %134, %167, %c1073741823 : index
                vector.store %164, %143[%168]
                : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>

                // Slice 5
                %169 = vector.extract_strided_slice %118 {offsets = [5], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %170 = arith.truncf %169 : vector<1xf32> to vector<1xbf16>
                %171 = affine.apply #map34()[%thread_id_x]

                // NEW: store slice 5 to LDS
                vector.store %170, %shared_output[%171, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %172 = arith.muli %171, %c68032 overflow<nsw> : index
                %173 = arith.addi %172, %86 overflow<nsw> : index
                %174 = arith.select %134, %173, %c1073741823 : index
                vector.store %170, %143[%174]
                : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>

                // Slice 6
                %175 = vector.extract_strided_slice %118 {offsets = [6], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %176 = arith.truncf %175 : vector<1xf32> to vector<1xbf16>
                %177 = affine.apply #map35()[%thread_id_x]

                // NEW: store slice 6 to LDS
                vector.store %176, %shared_output[%177, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %178 = arith.muli %177, %c68032 overflow<nsw> : index
                %179 = arith.addi %178, %86 overflow<nsw> : index
                %180 = arith.select %134, %179, %c1073741823 : index
                vector.store %176, %143[%180]
                : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>

                // Slice 7
                %181 = vector.extract_strided_slice %118 {offsets = [7], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %182 = arith.truncf %181 : vector<1xf32> to vector<1xbf16>
                %183 = affine.apply #map36()[%thread_id_x]

                // NEW: store slice 7 to LDS
                vector.store %182, %shared_output[%183, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %184 = arith.muli %183, %c68032 overflow<nsw> : index
                %185 = arith.addi %184, %86 overflow<nsw> : index
                %186 = arith.select %134, %185, %c1073741823 : index
                vector.store %182, %143[%186]
                : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>

                // Slice 8
                %187 = vector.extract_strided_slice %118 {offsets = [8], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %188 = arith.truncf %187 : vector<1xf32> to vector<1xbf16>
                %189 = affine.apply #map37()[%thread_id_x]

                // NEW: store slice 8 to LDS
                vector.store %188, %shared_output[%189, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %190 = arith.muli %189, %c68032 overflow<nsw> : index
                %191 = arith.addi %190, %86 overflow<nsw> : index
                %192 = arith.select %134, %191, %c1073741823 : index
                vector.store %188, %143[%192]
                : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>

                // Slice 9
                %193 = vector.extract_strided_slice %118 {offsets = [9], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %194 = arith.truncf %193 : vector<1xf32> to vector<1xbf16>
                %195 = affine.apply #map38()[%thread_id_x]

                // NEW: store slice 9 to LDS
                vector.store %194, %shared_output[%195, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

            

                // Slice 10
                %199 = vector.extract_strided_slice %118 {offsets = [10], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %200 = arith.truncf %199 : vector<1xf32> to vector<1xbf16>
                %201 = affine.apply #map39()[%thread_id_x]

                // NEW: store slice 10 to LDS
                vector.store %200, %shared_output[%201, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

               

                // Slice 11
                %205 = vector.extract_strided_slice %118 {offsets = [11], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %206 = arith.truncf %205 : vector<1xf32> to vector<1xbf16>
                %207 = affine.apply #map40()[%thread_id_x]

                // NEW: store slice 11 to LDS
                vector.store %206, %shared_output[%207, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 12
                %211 = vector.extract_strided_slice %118 {offsets = [12], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %212 = arith.truncf %211 : vector<1xf32> to vector<1xbf16>
                %213 = affine.apply #map41()[%thread_id_x]

                // NEW: store slice 12 to LDS
                vector.store %212, %shared_output[%213, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 13
                %217 = vector.extract_strided_slice %118 {offsets = [13], sizes = [1], strides = [1]}  : vector<16xf32> to vector<1xf32>
                %218 = arith.truncf %217 : vector<1xf32> to vector<1xbf16>
                %219 = affine.apply #map42()[%thread_id_x]

                // NEW: store slice 13 to LDS
                vector.store %218, %shared_output[%219, %86] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 14
                %223 = vector.extract_strided_slice %118 {offsets = [14], sizes = [1], strides = [1]}: vector<16xf32> to vector<1xf32>
                %224 = arith.truncf %223 : vector<1xf32> to vector<1xbf16>
                %225 = affine.apply #map43()[%thread_id_x]

                // NEW: store slice 14 to LDS
                vector.store %224, %shared_output[%225, %86]  : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>
                  // Slice 15
                %229 = vector.extract_strided_slice %118 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %230 = arith.truncf %229 : vector<1xf32> to vector<1xbf16>
                %231 = affine.apply #map44()[%thread_id_x]

                // NEW: store slice 15 to LDS
                vector.store %230, %shared_output[%231, %86]: memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %235 = vector.extract_strided_slice %122 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %236 = arith.truncf %235 : vector<1xf32> to vector<1xbf16>
                %237 = affine.apply #map45()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %238 = arith.cmpi slt, %237, %c68032 : index
                %239 = arith.addi %139, %95 overflow<nsw> : index
                %240 = arith.select %238, %239, %c1073741823 : index


                // NEW: LDS store for slice 0 (row = %137, col = %95)
                vector.store %236, %shared_output[%137, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 1
                %241 = vector.extract_strided_slice %122 {offsets = [1], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %242 = arith.truncf %241 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 1 (row = %147, col = %95)
                vector.store %242, %shared_output[%147, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 2
                %245 = vector.extract_strided_slice %122 {offsets = [2], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %246 = arith.truncf %245 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 2 (row = %153, col = %95)
                vector.store %246, %shared_output[%153, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 3
                %249 = vector.extract_strided_slice %122 {offsets = [3], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %250 = arith.truncf %249 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 3 (row = %159, col = %95)
                vector.store %250, %shared_output[%159, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 4
                %253 = vector.extract_strided_slice %122 {offsets = [4], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %254 = arith.truncf %253 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 4 (row = %165, col = %95)
                vector.store %254, %shared_output[%165, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 5
                %257 = vector.extract_strided_slice %122 {offsets = [5], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %258 = arith.truncf %257 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 5 (row = %171, col = %95)
                vector.store %258, %shared_output[%171, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 6
                %261 = vector.extract_strided_slice %122 {offsets = [6], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %262 = arith.truncf %261 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 6 (row = %177, col = %95)
                vector.store %262, %shared_output[%177, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 7
                %265 = vector.extract_strided_slice %122 {offsets = [7], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %266 = arith.truncf %265 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 7 (row = %183, col = %95)
                vector.store %266, %shared_output[%183, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

            

                // Slice 8
                %269 = vector.extract_strided_slice %122 {offsets = [8], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %270 = arith.truncf %269 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 8 (row = %189, col = %95)
                vector.store %270, %shared_output[%189, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 9
                %273 = vector.extract_strided_slice %122 {offsets = [9], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %274 = arith.truncf %273 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 9 (row = %195, col = %95)
                vector.store %274, %shared_output[%195, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 10
                %277 = vector.extract_strided_slice %122 {offsets = [10], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %278 = arith.truncf %277 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 10 (row = %201, col = %95)
                vector.store %278, %shared_output[%201, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 11
                %281 = vector.extract_strided_slice %122 {offsets = [11], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %282 = arith.truncf %281 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 11 (row = %207, col = %95)
                vector.store %282, %shared_output[%207, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

        

                // Slice 12
                %285 = vector.extract_strided_slice %122 {offsets = [12], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %286 = arith.truncf %285 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 12 (row = %213, col = %95)
                vector.store %286, %shared_output[%213, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 13
                %289 = vector.extract_strided_slice %122 {offsets = [13], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %290 = arith.truncf %289 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 13 (row = %219, col = %95)
                vector.store %290, %shared_output[%219, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 14
                %293 = vector.extract_strided_slice %122 {offsets = [14], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %294 = arith.truncf %293 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 14 (row = %225, col = %95)
                vector.store %294, %shared_output[%225, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 15
                %297 = vector.extract_strided_slice %122 {offsets = [15], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %298 = arith.truncf %297 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store for slice 15 (row = %231, col = %95)
                vector.store %298, %shared_output[%231, %95] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                        // Slice 0 from %126
                %301 = vector.extract_strided_slice %126 {offsets = [0], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %302 = arith.truncf %301 : vector<1xf32> to vector<1xbf16>
                %303 = affine.apply #map46()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %304 = arith.cmpi slt, %303, %c68032 : index

                // NEW: LDS store slice 0 (row = %137, col = %100)
                vector.store %302, %shared_output[%137, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 1
                %307 = vector.extract_strided_slice %126 {offsets = [1], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %308 = arith.truncf %307 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 1 (row = %147, col = %100)
                vector.store %308, %shared_output[%147, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 2
                %311 = vector.extract_strided_slice %126 {offsets = [2], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %312 = arith.truncf %311 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 2 (row = %153, col = %100)
                vector.store %312, %shared_output[%153, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 3
                %315 = vector.extract_strided_slice %126 {offsets = [3], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %316 = arith.truncf %315 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 3 (row = %159, col = %100)
                vector.store %316, %shared_output[%159, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 4
                %319 = vector.extract_strided_slice %126 {offsets = [4], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %320 = arith.truncf %319 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 4 (row = %165, col = %100)
                vector.store %320, %shared_output[%165, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 5
                %323 = vector.extract_strided_slice %126 {offsets = [5], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %324 = arith.truncf %323 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 5 (row = %171, col = %100)
                vector.store %324, %shared_output[%171, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 6
                %327 = vector.extract_strided_slice %126 {offsets = [6], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %328 = arith.truncf %327 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 6 (row = %177, col = %100)
                vector.store %328, %shared_output[%177, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 7
                %331 = vector.extract_strided_slice %126 {offsets = [7], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %332 = arith.truncf %331 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 7 (row = %183, col = %100)
                vector.store %332, %shared_output[%183, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 8
                %335 = vector.extract_strided_slice %126 {offsets = [8], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %336 = arith.truncf %335 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 8 (row = %189, col = %100)
                vector.store %336, %shared_output[%189, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 9
                %339 = vector.extract_strided_slice %126 {offsets = [9], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %340 = arith.truncf %339 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 9 (row = %195, col = %100)
                vector.store %340, %shared_output[%195, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 10
                %343 = vector.extract_strided_slice %126 {offsets = [10], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %344 = arith.truncf %343 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 10 (row = %201, col = %100)
                vector.store %344, %shared_output[%201, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 11
                %347 = vector.extract_strided_slice %126 {offsets = [11], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %348 = arith.truncf %347 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 11 (row = %207, col = %100)
                vector.store %348, %shared_output[%207, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 12
                %351 = vector.extract_strided_slice %126 {offsets = [12], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %352 = arith.truncf %351 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 12 (row = %213, col = %100)
                vector.store %352, %shared_output[%213, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 13
                %355 = vector.extract_strided_slice %126 {offsets = [13], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %356 = arith.truncf %355 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 13 (row = %219, col = %100)
                vector.store %356, %shared_output[%219, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 14
                %359 = vector.extract_strided_slice %126 {offsets = [14], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %360 = arith.truncf %359 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 14 (row = %225, col = %100)
                vector.store %360, %shared_output[%225, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 15
                %363 = vector.extract_strided_slice %126 {offsets = [15], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %364 = arith.truncf %363 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 15 (row = %231, col = %100)
                vector.store %364, %shared_output[%231, %100] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 0 from %130
                %367 = vector.extract_strided_slice %130 {offsets = [0], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %368 = arith.truncf %367 : vector<1xf32> to vector<1xbf16>
                %369 = affine.apply #map47()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %370 = arith.cmpi slt, %369, %c68032 : index

                // NEW: LDS store slice 0 (row = %137, col = %105)
                vector.store %368, %shared_output[%137, %105] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 1
                %373 = vector.extract_strided_slice %130 {offsets = [1], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %374 = arith.truncf %373 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 1 (row = %147, col = %105)
                vector.store %374, %shared_output[%147, %105] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 2
                %377 = vector.extract_strided_slice %130 {offsets = [2], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %378 = arith.truncf %377 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 2 (row = %153, col = %105)
                vector.store %378, %shared_output[%153, %105] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 3
                %381 = vector.extract_strided_slice %130 {offsets = [3], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %382 = arith.truncf %381 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 3 (row = %159, col = %105)
                vector.store %382, %shared_output[%159, %105] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 4
                %385 = vector.extract_strided_slice %130 {offsets = [4], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %386 = arith.truncf %385 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 4 (row = %165, col = %105)
                vector.store %386, %shared_output[%165, %105] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 5
                %389 = vector.extract_strided_slice %130 {offsets = [5], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %390 = arith.truncf %389 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 5 (row = %171, col = %105)
                vector.store %390, %shared_output[%171, %105] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 6
                %393 = vector.extract_strided_slice %130 {offsets = [6], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %394 = arith.truncf %393 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 6 (row = %177, col = %105)
                vector.store %394, %shared_output[%177, %105] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 7
                %397 = vector.extract_strided_slice %130 {offsets = [7], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %398 = arith.truncf %397 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 7 (row = %183, col = %105)
                vector.store %398, %shared_output[%183, %105] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 8
                %401 = vector.extract_strided_slice %130 {offsets = [8], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %402 = arith.truncf %401 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 8 (row = %189, col = %105)
                vector.store %402, %shared_output[%189, %105] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 9
                %405 = vector.extract_strided_slice %130 {offsets = [9], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %406 = arith.truncf %405 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 9 (row = %195, col = %105)
                vector.store %406, %shared_output[%195, %105] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>



                // Slice 10
                %409 = vector.extract_strided_slice %130 {offsets = [10], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %410 = arith.truncf %409 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 10 (row = %201, col = %105)
                vector.store %410, %shared_output[%201, %105]   : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 11
                %413 = vector.extract_strided_slice %130 {offsets = [11], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %414 = arith.truncf %413 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 11 (row = %207, col = %105)
                vector.store %414, %shared_output[%207, %105]      : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 12
                %417 = vector.extract_strided_slice %130 {offsets = [12], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %418 = arith.truncf %417 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 12 (row = %213, col = %105)
                vector.store %418, %shared_output[%213, %105]   : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 13
                %421 = vector.extract_strided_slice %130 {offsets = [13], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %422 = arith.truncf %421 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 13 (row = %219, col = %105)
                vector.store %422, %shared_output[%219, %105]   : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>


                // Slice 14
                %425 = vector.extract_strided_slice %130 {offsets = [14], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %426 = arith.truncf %425 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 14 (row = %225, col = %105)
                vector.store %426, %shared_output[%225, %105] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 15
                %429 = vector.extract_strided_slice %130 {offsets = [15], sizes = [1], strides = [1]}   : vector<16xf32> to vector<1xf32>
                %430 = arith.truncf %429 : vector<1xf32> to vector<1xbf16>

                // NEW: LDS store slice 15 (row = %231, col = %105)
                vector.store %430, %shared_output[%231, %105]  : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %rowint = affine.apply #intermediaterow()[%thread_id_x, %thread_id_y]
                %colint = affine.apply #intermediatecol()[%thread_id_x, %thread_id_y]


                %data = vector.load %shared_output[%rowint, %colint]: memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<64xbf16>
                        
                %globalrow = arith.muli %rowint, %c68032 overflow<nsw> : index
                %globalindex = arith.addi %globalrow, %colint overflow<nsw> : index

                //%idx_in_view = arith.select %117, %globalindex, %c1073741823 : index

                vector.store %data, %143[%globalindex] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<64xbf16>

                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x1024xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<68032x1024xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x68032xbf16>
            %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<256x1024xbf16>, tensor<68032x1024xbf16>, tensor<256x68032xbf16>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x68032xbf16>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x68032xbf16> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
    """
   
    asm_256x4096x1024 = """
        #map = affine_map<()[s0, s1] -> (((s0 * 4 + s1 * 8 - ((s0 + s1 * 2) floordiv 8) * 31) floordiv 256) * -16 + 2)>
        #map1 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128 + ((s2 * 4 + s3 * 8 - ((s2 + s3 * 2) floordiv 8) * 31) floordiv 256) * 2048 + (((s2 * 4 + s3 * 8 - ((s2 + s3 * 2) floordiv 8) * 31) mod 256) mod s4) * 128)>
        #map2 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 8) * 64)>
        #map3 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + ((s2 * 4 + s3 * 8 - ((s2 + s3 * 2) floordiv 8) * 31) floordiv 256) * 2048 + (((s2 * 4 + s3 * 8 - ((s2 + s3 * 2) floordiv 8) * 31) mod 256) mod s4) * 128 + 64)>
        #map4 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256 + (((s2 * 4 + s3 * 8 - ((s2 + s3 * 2) floordiv 8) * 31) mod 256) floordiv s4) * 256)>
        #map5 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + (((s2 * 4 + s3 * 8 - ((s2 + s3 * 2) floordiv 8) * 31) mod 256) floordiv s4) * 256 + 64)>
        #map6 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + (((s2 * 4 + s3 * 8 - ((s2 + s3 * 2) floordiv 8) * 31) mod 256) floordiv s4) * 256 + 128)>
        #map7 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + (((s2 * 4 + s3 * 8 - ((s2 + s3 * 2) floordiv 8) * 31) mod 256) floordiv s4) * 256 + 192)>
        #map8 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 128)>
        #map9 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + 64)>
        #map10 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
        #map11 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
        #map12 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
        #map13 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
        #map14 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map15 = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
        #map16 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8)>
        #map17 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 16)>
        #map18 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32)>
        #map19 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 32)>
        #map20 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 64)>
        #map21 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 96)>
        #map22 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 32)>
        #map23 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 48)>
        #map24 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 - (s1 floordiv 8) * 64 + 64)>
        #map25 = affine_map<()[s0, s1, s2] -> (((s0 * 4 + s1 * 8 - ((s0 + s1 * 2) floordiv 8) * 31) floordiv 256) * 2048 + (((s0 * 4 + s1 * 8 - ((s0 + s1 * 2) floordiv 8) * 31) mod 256) mod s2) * 128)>
        #map26 = affine_map<()[s0, s1, s2] -> ((((s0 * 4 + s1 * 8 - ((s0 + s1 * 2) floordiv 8) * 31) mod 256) floordiv s2) * 256)>
        #map27 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
        #map28 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
        #map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
        #map30 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
        #map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
        #map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
        #map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
        #map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
        #map35 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
        #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
        #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
        #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
        #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
        #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
        #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
        #intermediaterow = affine_map<()[s0, s1] -> (s0 floordiv 4 + s1 * 64)>
        #intermediatecol = affine_map<()[s0, s1] -> ((s0 mod 4) * 64)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c2 = arith.constant 2 : index
            %c16 = arith.constant 16 : index
            %c1 = arith.constant 1 : index
            stream.return %c2, %c16, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c4096_i14 = arith.constant 4096 : i14
                %c4_i32 = arith.constant 4 : i32
                %c1024_i14 = arith.constant 1024 : i14
                %c4096 = arith.constant 4096 : index
                %c0_i32 = arith.constant 0 : i32
                %c15 = arith.constant 15 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c1024 = arith.constant 1024 : index
                %c52224 = arith.constant 52224 : index
                %c1 = arith.constant 1 : index
                %c34816 = arith.constant 34816 : index
                
                %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<bf16>
                %block_id_x = gpu.block_id  x upper_bound 2
                %block_id_y = gpu.block_id  y upper_bound 16
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 1024], strides: [1024, 1] : memref<bf16> to memref<256x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [4096, 1024], strides: [1024, 1] : memref<bf16> to memref<4096x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 4096], strides: [4096, 1] : memref<bf16> to memref<256x4096xbf16, strided<[4096, 1], offset: ?>>
                %alloc = memref.alloc() : memref<118784xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c0][] : memref<118784xi8, #gpu.address_space<workgroup>> to memref<256x68xbf16, #gpu.address_space<workgroup>>
                %view_2 = memref.view %alloc[%c34816][] : memref<118784xi8, #gpu.address_space<workgroup>> to memref<128x68xbf16, #gpu.address_space<workgroup>>
                %shared_output = memref.view %alloc[%c52224][] : memref<118784xi8, #gpu.address_space<workgroup>> to memref<128x260xbf16, #gpu.address_space<workgroup>>

                %3 = affine.apply #map()[%block_id_x, %block_id_y]
                %4 = arith.maxsi %3, %c1 : index
                %5 = affine.apply #map1()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %6 = affine.apply #map2()[%thread_id_x]
                %7 = arith.muli %5, %c1024 overflow<nsw> : index
                %8 = arith.addi %7, %6 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_3 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %9 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_3 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %10 = vector.load %9[%8] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %11 = affine.apply #map3()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %12 = arith.muli %11, %c1024 overflow<nsw> : index
                %13 = arith.addi %12, %6 overflow<nsw> : index
                %14 = vector.load %9[%13] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %15 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %16 = arith.muli %15, %c1024 overflow<nsw> : index
                %17 = arith.addi %16, %6 overflow<nsw> : index
                %base_buffer_4, %offset_5, %sizes_6:2, %strides_7:2 = memref.extract_strided_metadata %reinterpret_cast_0 : memref<4096x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_8 = memref.reinterpret_cast %1 to offset: [%offset_5], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %18 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_8 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %19 = vector.load %18[%17] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %20 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %21 = arith.muli %20, %c1024 overflow<nsw> : index
                %22 = arith.addi %21, %6 overflow<nsw> : index
                %23 = vector.load %18[%22] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %24 = affine.apply #map6()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %25 = arith.muli %24, %c1024 overflow<nsw> : index
                %26 = arith.addi %25, %6 overflow<nsw> : index
                %27 = vector.load %18[%26] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %28 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %29 = arith.muli %28, %c1024 overflow<nsw> : index
                %30 = arith.addi %29, %6 overflow<nsw> : index
                %31 = vector.load %18[%30] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %32 = affine.apply #map8()[%thread_id_x, %thread_id_y]
                vector.store %10, %view_2[%32, %6] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %33 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                vector.store %14, %view_2[%33, %6] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %34 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                vector.store %19, %view[%34, %6] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %35 = affine.apply #map11()[%thread_id_x, %thread_id_y]
                vector.store %23, %view[%35, %6] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %36 = affine.apply #map12()[%thread_id_x, %thread_id_y]
                vector.store %27, %view[%36, %6] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %37 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                vector.store %31, %view[%37, %6] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                amdgpu.lds_barrier
                %38 = affine.apply #map14()[%thread_id_x, %thread_id_y]
                %39 = arith.index_cast %38 : index to i32
                %40 = arith.cmpi sge, %39, %c4_i32 : i32
                %41 = arith.cmpi slt, %39, %c4_i32 : i32
                scf.if %40 {
                rocdl.s.barrier
                }
                %42 = affine.apply #map15()[%thread_id_x]
                %43 = affine.apply #map16()[%thread_id_x]
                %44 = affine.apply #map17()[%thread_id_x]
                %45 = affine.apply #map18()[%thread_id_x, %thread_id_y]
                %46 = affine.apply #map19()[%thread_id_x, %thread_id_y]
                %47 = affine.apply #map20()[%thread_id_x, %thread_id_y]
                %48 = affine.apply #map21()[%thread_id_x, %thread_id_y]
                %49 = affine.apply #map22()[%thread_id_x]
                %50 = affine.apply #map23()[%thread_id_x]
                %51:4 = scf.for %arg3 = %c0 to %c15 step %c1 iter_args(%arg4 = %cst, %arg5 = %cst, %arg6 = %cst, %arg7 = %cst) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
                %327 = vector.load %view_2[%42, %43] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %328 = vector.load %view_2[%42, %44] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %329 = vector.load %view[%45, %43] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %330 = vector.load %view[%45, %44] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %331 = vector.load %view[%46, %43] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %332 = vector.load %view[%46, %44] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %333 = vector.load %view[%47, %43] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %334 = vector.load %view[%47, %44] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %335 = vector.load %view[%48, %43] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %336 = vector.load %view[%48, %44] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %337 = affine.apply #map24()[%arg3, %thread_id_x]
                %338 = arith.addi %12, %337 overflow<nsw> : index
                %339 = vector.load %9[%338] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %340 = arith.addi %7, %337 overflow<nsw> : index
                %341 = vector.load %9[%340] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %342 = vector.load %view_2[%42, %49] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %343 = vector.load %view_2[%42, %50] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %344 = vector.load %view[%45, %49] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %345 = vector.load %view[%45, %50] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %346 = vector.load %view[%46, %49] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %347 = vector.load %view[%46, %50] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %348 = vector.load %view[%47, %49] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %349 = vector.load %view[%47, %50] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %350 = vector.load %view[%48, %49] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %351 = vector.load %view[%48, %50] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %352 = arith.addi %16, %337 overflow<nsw> : index
                %353 = vector.load %18[%352] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %354 = arith.addi %29, %337 overflow<nsw> : index
                %355 = vector.load %18[%354] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %356 = arith.addi %25, %337 overflow<nsw> : index
                %357 = vector.load %18[%356] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %358 = arith.addi %21, %337 overflow<nsw> : index
                %359 = vector.load %18[%358] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %360 = amdgpu.mfma %327 * %329 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %361 = amdgpu.mfma %328 * %330 + %360 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %362 = amdgpu.mfma %327 * %331 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %363 = amdgpu.mfma %328 * %332 + %362 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %364 = amdgpu.mfma %327 * %333 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %365 = amdgpu.mfma %328 * %334 + %364 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %366 = amdgpu.mfma %327 * %335 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %367 = amdgpu.mfma %328 * %336 + %366 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                rocdl.s.setprio 0
                amdgpu.lds_barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                vector.store %339, %view_2[%33, %6] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %341, %view_2[%32, %6] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %355, %view[%37, %6] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %353, %view[%34, %6] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %357, %view[%36, %6] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %359, %view[%35, %6] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %368 = amdgpu.mfma %342 * %344 + %361 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %369 = amdgpu.mfma %343 * %345 + %368 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %370 = amdgpu.mfma %342 * %346 + %363 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %371 = amdgpu.mfma %343 * %347 + %370 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %372 = amdgpu.mfma %342 * %348 + %365 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %373 = amdgpu.mfma %343 * %349 + %372 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %374 = amdgpu.mfma %342 * %350 + %367 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %375 = amdgpu.mfma %343 * %351 + %374 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                rocdl.s.setprio 0
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                amdgpu.lds_barrier
                scf.yield %369, %371, %373, %375 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
                }
                scf.if %41 {
                rocdl.s.barrier
                }
                %52 = affine.apply #map18()[%thread_id_x, %thread_id_y]
                %53 = affine.apply #map16()[%thread_id_x]
                %54 = vector.load %view[%52, %53] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %55 = affine.apply #map17()[%thread_id_x]
                %56 = vector.load %view[%52, %55] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %57 = affine.apply #map22()[%thread_id_x]
                %58 = vector.load %view[%52, %57] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %59 = affine.apply #map23()[%thread_id_x]
                %60 = vector.load %view[%52, %59] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %61 = affine.apply #map19()[%thread_id_x, %thread_id_y]
                %62 = vector.load %view[%61, %53] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %63 = vector.load %view[%61, %55] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %64 = vector.load %view[%61, %57] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %65 = vector.load %view[%61, %59] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %66 = affine.apply #map20()[%thread_id_x, %thread_id_y]
                %67 = vector.load %view[%66, %53] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %68 = vector.load %view[%66, %55] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %69 = vector.load %view[%66, %57] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %70 = vector.load %view[%66, %59] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %71 = affine.apply #map21()[%thread_id_x, %thread_id_y]
                %72 = vector.load %view[%71, %53] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %73 = vector.load %view[%71, %55] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %74 = vector.load %view[%71, %57] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %75 = vector.load %view[%71, %59] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %76 = affine.apply #map15()[%thread_id_x]
                %77 = vector.load %view_2[%76, %53] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %78 = vector.load %view_2[%76, %55] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %79 = vector.load %view_2[%76, %57] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %80 = vector.load %view_2[%76, %59] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %81 = amdgpu.mfma %77 * %54 + %51#0 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %82 = amdgpu.mfma %78 * %56 + %81 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %83 = amdgpu.mfma %79 * %58 + %82 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %84 = amdgpu.mfma %80 * %60 + %83 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %85 = amdgpu.mfma %77 * %62 + %51#1 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %86 = amdgpu.mfma %78 * %63 + %85 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %87 = amdgpu.mfma %79 * %64 + %86 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %88 = amdgpu.mfma %80 * %65 + %87 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %89 = amdgpu.mfma %77 * %67 + %51#2 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %90 = amdgpu.mfma %78 * %68 + %89 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %91 = amdgpu.mfma %79 * %69 + %90 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %92 = amdgpu.mfma %80 * %70 + %91 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %93 = amdgpu.mfma %77 * %72 + %51#3 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %94 = amdgpu.mfma %78 * %73 + %93 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %95 = amdgpu.mfma %79 * %74 + %94 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %96 = amdgpu.mfma %80 * %75 + %95 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %97 = vector.extract_strided_slice %84 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %98 = arith.truncf %97 : vector<1xf32> to vector<1xbf16>
                %99 = affine.apply #map25()[%block_id_x, %block_id_y, %4]
                %100 = affine.apply #map26()[%block_id_x, %block_id_y, %4]
                %101 = affine.apply #map27()[%thread_id_x]
                %102 = arith.muli %99, %c4096 overflow<nsw> : index
                %103 = arith.muli %101, %c4096 overflow<nsw> : index
                %104 = arith.addi %102, %100 overflow<nsw> : index
                %105 = arith.addi %103, %52 overflow<nsw> : index
                %base_buffer_9, %offset_10, %sizes_11:2, %strides_12:2 = memref.extract_strided_metadata %reinterpret_cast_1 : memref<256x4096xbf16, strided<[4096, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %106 = arith.addi %104, %offset_10 overflow<nsw> : index
                %reinterpret_cast_13 = memref.reinterpret_cast %2 to offset: [%106], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %107 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_13 validBytes(%c2147483645_i64) cacheSwizzleStride(%c4096_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                vector.store %98, %shared_output[%101, %52] : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 1 from %84 → LDS
                %108 = vector.extract_strided_slice %84 {offsets = [1], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %109 = arith.truncf %108 : vector<1xf32> to vector<1xbf16>
                %110 = affine.apply #map28()[%thread_id_x]
            
                vector.store %109, %shared_output[%110, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 2 from %84 → LDS
                %113 = vector.extract_strided_slice %84 {offsets = [2], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %114 = arith.truncf %113 : vector<1xf32> to vector<1xbf16>
                %115 = affine.apply #map29()[%thread_id_x]
        
                vector.store %114, %shared_output[%115, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 3 from %84 → LDS
                %118 = vector.extract_strided_slice %84 {offsets = [3], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %119 = arith.truncf %118 : vector<1xf32> to vector<1xbf16>
                %120 = affine.apply #map30()[%thread_id_x]
        
                vector.store %119, %shared_output[%120, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 4 from %84 → LDS
                %123 = vector.extract_strided_slice %84 {offsets = [4], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %124 = arith.truncf %123 : vector<1xf32> to vector<1xbf16>
                %125 = affine.apply #map31()[%thread_id_x]

                vector.store %124, %shared_output[%125, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 5 from %84 → LDS
                %128 = vector.extract_strided_slice %84 {offsets = [5], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %129 = arith.truncf %128 : vector<1xf32> to vector<1xbf16>
                %130 = affine.apply #map32()[%thread_id_x]
        
                vector.store %129, %shared_output[%130, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 6 from %84 → LDS
                %133 = vector.extract_strided_slice %84 {offsets = [6], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %134 = arith.truncf %133 : vector<1xf32> to vector<1xbf16>
                %135 = affine.apply #map33()[%thread_id_x]

                vector.store %134, %shared_output[%135, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 7 from %84 → LDS
                %138 = vector.extract_strided_slice %84 {offsets = [7], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %139 = arith.truncf %138 : vector<1xf32> to vector<1xbf16>
                %140 = affine.apply #map34()[%thread_id_x]
                vector.store %139, %shared_output[%140, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 8 from %84 → LDS
                %143 = vector.extract_strided_slice %84 {offsets = [8], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %144 = arith.truncf %143 : vector<1xf32> to vector<1xbf16>
                %145 = affine.apply #map35()[%thread_id_x]

                vector.store %144, %shared_output[%145, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 9 from %84 → LDS
                %148 = vector.extract_strided_slice %84 {offsets = [9], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %149 = arith.truncf %148 : vector<1xf32> to vector<1xbf16>
                %150 = affine.apply #map36()[%thread_id_x]
                vector.store %149, %shared_output[%150, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 10 from %84 → LDS
                %153 = vector.extract_strided_slice %84 {offsets = [10], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %154 = arith.truncf %153 : vector<1xf32> to vector<1xbf16>
                %155 = affine.apply #map37()[%thread_id_x]
                vector.store %154, %shared_output[%155, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 11 from %84 → LDS
                %158 = vector.extract_strided_slice %84 {offsets = [11], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %159 = arith.truncf %158 : vector<1xf32> to vector<1xbf16>
                %160 = affine.apply #map38()[%thread_id_x]

                vector.store %159, %shared_output[%160, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 12 from %84 → LDS
                %163 = vector.extract_strided_slice %84 {offsets = [12], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %164 = arith.truncf %163 : vector<1xf32> to vector<1xbf16>
                %165 = affine.apply #map39()[%thread_id_x]

                vector.store %164, %shared_output[%165, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 13 from %84 → LDS
                %168 = vector.extract_strided_slice %84 {offsets = [13], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %169 = arith.truncf %168 : vector<1xf32> to vector<1xbf16>
                %170 = affine.apply #map40()[%thread_id_x]

                vector.store %169, %shared_output[%170, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 14 from %84 → LDS
                %173 = vector.extract_strided_slice %84 {offsets = [14], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %174 = arith.truncf %173 : vector<1xf32> to vector<1xbf16>
                %175 = affine.apply #map41()[%thread_id_x]

                vector.store %174, %shared_output[%175, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Slice 15 from %84 → LDS
                %178 = vector.extract_strided_slice %84 {offsets = [15], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %179 = arith.truncf %178 : vector<1xf32> to vector<1xbf16>
                %180 = affine.apply #map42()[%thread_id_x]

                vector.store %179, %shared_output[%180, %52]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // Now the same pattern for %88 → col %61
                %183 = vector.extract_strided_slice %88 {offsets = [0], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %184 = arith.truncf %183 : vector<1xf32> to vector<1xbf16>
                vector.store %184, %shared_output[%101, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %186 = vector.extract_strided_slice %88 {offsets = [1], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %187 = arith.truncf %186 : vector<1xf32> to vector<1xbf16>
                vector.store %187, %shared_output[%110, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %189 = vector.extract_strided_slice %88 {offsets = [2], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %190 = arith.truncf %189 : vector<1xf32> to vector<1xbf16>
                vector.store %190, %shared_output[%115, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %192 = vector.extract_strided_slice %88 {offsets = [3], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %193 = arith.truncf %192 : vector<1xf32> to vector<1xbf16>
                vector.store %193, %shared_output[%120, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %195 = vector.extract_strided_slice %88 {offsets = [4], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %196 = arith.truncf %195 : vector<1xf32> to vector<1xbf16>
                vector.store %196, %shared_output[%125, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %198 = vector.extract_strided_slice %88 {offsets = [5], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %199 = arith.truncf %198 : vector<1xf32> to vector<1xbf16>
                vector.store %199, %shared_output[%130, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %201 = vector.extract_strided_slice %88 {offsets = [6], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %202 = arith.truncf %201 : vector<1xf32> to vector<1xbf16>
                vector.store %202, %shared_output[%135, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %204 = vector.extract_strided_slice %88 {offsets = [7], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %205 = arith.truncf %204 : vector<1xf32> to vector<1xbf16>
                vector.store %205, %shared_output[%140, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %207 = vector.extract_strided_slice %88 {offsets = [8], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %208 = arith.truncf %207 : vector<1xf32> to vector<1xbf16>
                vector.store %208, %shared_output[%145, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %210 = vector.extract_strided_slice %88 {offsets = [9], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %211 = arith.truncf %210 : vector<1xf32> to vector<1xbf16>
                vector.store %211, %shared_output[%150, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %213 = vector.extract_strided_slice %88 {offsets = [10], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %214 = arith.truncf %213 : vector<1xf32> to vector<1xbf16>
                vector.store %214, %shared_output[%155, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %216 = vector.extract_strided_slice %88 {offsets = [11], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %217 = arith.truncf %216 : vector<1xf32> to vector<1xbf16>
                vector.store %217, %shared_output[%160, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %219 = vector.extract_strided_slice %88 {offsets = [12], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %220 = arith.truncf %219 : vector<1xf32> to vector<1xbf16>
                vector.store %220, %shared_output[%165, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %222 = vector.extract_strided_slice %88 {offsets = [13], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %223 = arith.truncf %222 : vector<1xf32> to vector<1xbf16>
                vector.store %223, %shared_output[%170, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %225 = vector.extract_strided_slice %88 {offsets = [14], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %226 = arith.truncf %225 : vector<1xf32> to vector<1xbf16>
                vector.store %226, %shared_output[%175, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %228 = vector.extract_strided_slice %88 {offsets = [15], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %229 = arith.truncf %228 : vector<1xf32> to vector<1xbf16>
                vector.store %229, %shared_output[%180, %61]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>
                %231 = vector.extract_strided_slice %92 {offsets = [0], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %232 = arith.truncf %231 : vector<1xf32> to vector<1xbf16>
                vector.store %232, %shared_output[%101, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %234 = vector.extract_strided_slice %92 {offsets = [1], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %235 = arith.truncf %234 : vector<1xf32> to vector<1xbf16>
                vector.store %235, %shared_output[%110, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %237 = vector.extract_strided_slice %92 {offsets = [2], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %238 = arith.truncf %237 : vector<1xf32> to vector<1xbf16>
                vector.store %238, %shared_output[%115, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %240 = vector.extract_strided_slice %92 {offsets = [3], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %241 = arith.truncf %240 : vector<1xf32> to vector<1xbf16>
                vector.store %241, %shared_output[%120, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %243 = vector.extract_strided_slice %92 {offsets = [4], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %244 = arith.truncf %243 : vector<1xf32> to vector<1xbf16>
                vector.store %244, %shared_output[%125, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %246 = vector.extract_strided_slice %92 {offsets = [5], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %247 = arith.truncf %246 : vector<1xf32> to vector<1xbf16>
                vector.store %247, %shared_output[%130, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %249 = vector.extract_strided_slice %92 {offsets = [6], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %250 = arith.truncf %249 : vector<1xf32> to vector<1xbf16>
                vector.store %250, %shared_output[%135, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %252 = vector.extract_strided_slice %92 {offsets = [7], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %253 = arith.truncf %252 : vector<1xf32> to vector<1xbf16>
                vector.store %253, %shared_output[%140, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %255 = vector.extract_strided_slice %92 {offsets = [8], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %256 = arith.truncf %255 : vector<1xf32> to vector<1xbf16>
                vector.store %256, %shared_output[%145, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %258 = vector.extract_strided_slice %92 {offsets = [9], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %259 = arith.truncf %258 : vector<1xf32> to vector<1xbf16>
                vector.store %259, %shared_output[%150, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %261 = vector.extract_strided_slice %92 {offsets = [10], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %262 = arith.truncf %261 : vector<1xf32> to vector<1xbf16>
                vector.store %262, %shared_output[%155, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %264 = vector.extract_strided_slice %92 {offsets = [11], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %265 = arith.truncf %264 : vector<1xf32> to vector<1xbf16>
                vector.store %265, %shared_output[%160, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %267 = vector.extract_strided_slice %92 {offsets = [12], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %268 = arith.truncf %267 : vector<1xf32> to vector<1xbf16>
                vector.store %268, %shared_output[%165, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %270 = vector.extract_strided_slice %92 {offsets = [13], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %271 = arith.truncf %270 : vector<1xf32> to vector<1xbf16>
                vector.store %271, %shared_output[%170, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %273 = vector.extract_strided_slice %92 {offsets = [14], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %274 = arith.truncf %273 : vector<1xf32> to vector<1xbf16>
                vector.store %274, %shared_output[%175, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %276 = vector.extract_strided_slice %92 {offsets = [15], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %277 = arith.truncf %276 : vector<1xf32> to vector<1xbf16>
                vector.store %277, %shared_output[%180, %66]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                // ===== Tile for %96 → col %71 =====
                %279 = vector.extract_strided_slice %96 {offsets = [0], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %280 = arith.truncf %279 : vector<1xf32> to vector<1xbf16>
                vector.store %280, %shared_output[%101, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %282 = vector.extract_strided_slice %96 {offsets = [1], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %283 = arith.truncf %282 : vector<1xf32> to vector<1xbf16>
                vector.store %283, %shared_output[%110, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %285 = vector.extract_strided_slice %96 {offsets = [2], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %286 = arith.truncf %285 : vector<1xf32> to vector<1xbf16>
                vector.store %286, %shared_output[%115, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %288 = vector.extract_strided_slice %96 {offsets = [3], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %289 = arith.truncf %288 : vector<1xf32> to vector<1xbf16>
                vector.store %289, %shared_output[%120, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %291 = vector.extract_strided_slice %96 {offsets = [4], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %292 = arith.truncf %291 : vector<1xf32> to vector<1xbf16>
                vector.store %292, %shared_output[%125, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %294 = vector.extract_strided_slice %96 {offsets = [5], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %295 = arith.truncf %294 : vector<1xf32> to vector<1xbf16>
                vector.store %295, %shared_output[%130, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %297 = vector.extract_strided_slice %96 {offsets = [6], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %298 = arith.truncf %297 : vector<1xf32> to vector<1xbf16>
                vector.store %298, %shared_output[%135, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %300 = vector.extract_strided_slice %96 {offsets = [7], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %301 = arith.truncf %300 : vector<1xf32> to vector<1xbf16>
                vector.store %301, %shared_output[%140, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %303 = vector.extract_strided_slice %96 {offsets = [8], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %304 = arith.truncf %303 : vector<1xf32> to vector<1xbf16>
                vector.store %304, %shared_output[%145, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %306 = vector.extract_strided_slice %96 {offsets = [9], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %307 = arith.truncf %306 : vector<1xf32> to vector<1xbf16>
                vector.store %307, %shared_output[%150, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %309 = vector.extract_strided_slice %96 {offsets = [10], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %310 = arith.truncf %309 : vector<1xf32> to vector<1xbf16>
                vector.store %310, %shared_output[%155, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %312 = vector.extract_strided_slice %96 {offsets = [11], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %313 = arith.truncf %312 : vector<1xf32> to vector<1xbf16>
                vector.store %313, %shared_output[%160, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %315 = vector.extract_strided_slice %96 {offsets = [12], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %316 = arith.truncf %315 : vector<1xf32> to vector<1xbf16>
                vector.store %316, %shared_output[%165, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %318 = vector.extract_strided_slice %96 {offsets = [13], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %319 = arith.truncf %318 : vector<1xf32> to vector<1xbf16>
                vector.store %319, %shared_output[%170, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %321 = vector.extract_strided_slice %96 {offsets = [14], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %322 = arith.truncf %321 : vector<1xf32> to vector<1xbf16>
                vector.store %322, %shared_output[%175, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %324 = vector.extract_strided_slice %96 {offsets = [15], sizes = [1], strides = [1]}
                    : vector<16xf32> to vector<1xf32>
                %325 = arith.truncf %324 : vector<1xf32> to vector<1xbf16>
                vector.store %325, %shared_output[%180, %71]
                : memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<1xbf16>

                %rowint = affine.apply #intermediaterow()[%thread_id_x, %thread_id_y]
                %colint = affine.apply #intermediatecol()[%thread_id_x, %thread_id_y]


                %data = vector.load %shared_output[%rowint, %colint]: memref<128x260xbf16, #gpu.address_space<workgroup>>, vector<64xbf16>
                        
                %globalrow = arith.muli %rowint, %c4096 overflow<nsw> : index
                %globalindex = arith.addi %globalrow, %colint overflow<nsw> : index

                vector.store %data, %107[%globalindex] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<64xbf16>

                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x1024xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<4096x1024xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x4096xbf16>
            %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<256x1024xbf16>, tensor<4096x1024xbf16>, tensor<256x4096xbf16>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x4096xbf16>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x4096xbf16> -> !hal.buffer_view
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
        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, mfma_variant, tkl.bf16, tkl.bf16,
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        use_buffer_ops=True,
        print_mlir=True,
        use_global_to_shared=True,
        #override_mlir=asm_256x4096x1024,
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
    # A = torch.ones(M, K, dtype=dtype, device="cuda")
    # B = torch.zeros(N, K, dtype=dtype, device="cuda")  # careful: ABᵀ → shape (M,N)

    #A = torch.arange(M * K, dtype=dtype, device="cuda").reshape(M, K)
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(N, K, dtype=dtype, device="cuda")
    C = torch.empty((M, N), dtype=dtype, device="cuda")

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

    torch.set_printoptions(
    threshold=float("inf"),  # no summarization
    linewidth=200,           # wider line before wrapping
    edgeitems=8,             # how many items to show at each edge
    precision=3,
    sci_mode=False,
    )


    print("wave A:")
    print(A[:2, :16])
    # print("wave B:")
    # print(B[:8, :68])
    print("wave output:")
    print(C[:2, :4])
    print("ref output")
    print(output_triton[:2, :4])

    diff = (C - output_triton.to(torch.bfloat16)).abs()
    if torch.allclose(C, output_triton.to(torch.bfloat16), atol=1e-2, rtol=1e-2):
        print("✅ Wave and Triton implementations match")
    else:
        print("❌ Wave and Triton implementations differ")
        max_diff = diff.max().item()
        print(f"Max diff Wave vs Triton: {max_diff}")
        
        # Per-row accuracy analysis
        threshold = 1e-2
        correct_mask = diff <= threshold
        num_rows = C.shape[0]
        num_cols = C.shape[1]
        
        print(f"\nPer-row accuracy (threshold={threshold}):")
        print(f"{'Row':>6} | {'Correct':>8} / {'Total':>8} | {'Accuracy':>8}")
        print("-" * 50)
        
        for row in range(num_rows):
            correct_in_row = correct_mask[row].sum().item()
            accuracy_pct = (correct_in_row / num_cols) * 100
            print(f"{row:6d} | {correct_in_row:8d} / {num_cols:8d} | {accuracy_pct:7.2f}%")
        
        # Overall stats
        total_correct = correct_mask.sum().item()
        total_elements = C.numel()
        overall_accuracy = (total_correct / total_elements) * 100
        print("-" * 50)
        print(f"Overall: {total_correct:8d} / {total_elements:8d} | {overall_accuracy:7.2f}%")


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

# M_vals = [256]
# N_vals = [4096]
# K_vals = [1024]

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

def bench(M, N, K, provider, ):
    dtype = torch.bfloat16
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
    
    #calculate_diff_gemm(256, 68032, 1024, torch.bfloat16)
    #calculate_diff_gemm(256, 512, 40960, torch.bfloat16)
    #(1280, 32, 192) 
    calculate_diff_gemm(1280, 32, 192, torch.bfloat16)

    #calculate_diff_gemm(256, 512, 128, torch.bfloat16)

    #calculate_diff_gemm(256, 256, 512)
    #calculate_diff_gemm(64, 128, 511)
    #calculate_diff_gemm(16384, 32768, 6144)

    







