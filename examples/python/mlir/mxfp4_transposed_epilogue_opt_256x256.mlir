// Handwritten MLIR for a transposed MXFP4 GEMM with an optimized permlane epilogue.
//
// Shape: M=256, N=256, K=4096. Inputs: MXFP4 (f4E2M1FN) with f8E8M0FNU scales.
// Output: bf16, row-major C[M, N].
//
// The kernel computes C^T = B * A^T by swapping operand roles.
// After the MFMAs, C^T lives in registers in transposed order.
//
// The epilogue uses amdgpu.permlane_swap (xor_mask=16) to exchange data between
// lane pairs and reassembles C^T into contiguous C[M, N] rows, enabling wide
// vector<8xbf16> stores (buffer_store_dwordx4) instead of scalar stores.
//
// Layout: 256 threads (4 warps x 64 lanes, + 2 thread_id_y groups = 8 waves).
// Each thread holds 32 vector<4xf32> accumulators (16 pairs).
// The epilogue issues 16 x vector<8xbf16> = 128 bf16 per thread.
// With 512 threads per workgroup: 512 * 128 = 65536 = 256 * 256. ✓
//
// Permlane_swap(xor=16) pairs:
//   - Lanes 0..15  ↔ lanes 16..31  within each 64-lane warp
//   - Lanes 32..47 ↔ lanes 48..63
// Even-group lanes (0..15, 32..47): store [own_4, partner_4] = 8 contiguous bf16
// Odd-group lanes  (16..31, 48..63): store [partner_4, own_4] (select flips order)
//
// N-base formula:  (thread_id_x floordiv 64) * 64 + ((thread_id_x mod 64) floordiv 32) * 8
// M-base formula:  thread_id_y * 128 + (thread_id_x mod 16)
//
#map = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
#map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
#map2 = affine_map<()[s0] -> (s0 mod 8)>
#map3 = affine_map<()[s0, s1, s2, s3] -> (s0 * 1048576 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8) floordiv 256) * 1048576)>
#map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
#map5 = affine_map<()[s0, s1, s2, s3] -> (s0 * 1048576 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 256) * 1048576 + 262144)>
#map6 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
#map7 = affine_map<()[s0, s1, s2, s3] -> (s0 * 1048576 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 128) floordiv 256) * 1048576 + 524288)>
#map8 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
#map9 = affine_map<()[s0, s1, s2, s3] -> (s0 * 1048576 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 192) floordiv 256) * 1048576 + 786432)>
#map10 = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 4 + (s1 floordiv 64) * 16384 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64)>
#map11 = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 4 + (s1 floordiv 64) * 16384 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 8192)>
#map12 = affine_map<()[s0, s1, s2] -> (s0 * 65536 + s1 * 32768 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64)>
#map13 = affine_map<()[s0, s1, s2] -> (s0 * 65536 + s1 * 32768 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 8192)>
#map14 = affine_map<()[s0, s1, s2] -> (s0 * 65536 + s1 * 32768 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 16384)>
#map15 = affine_map<()[s0, s1, s2] -> (s0 * 65536 + s1 * 32768 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 24576)>
#map16 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
#map17 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
#map18 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048)>
#map19 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048 + 2048)>
#map20 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048 + 4096)>
#map21 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048 + 6144)>
#map22 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048)>
#map23 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 2048)>
#map24 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 4096)>
#map25 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 6144)>
#map26 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 8192)>
#map27 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 10240)>
#map28 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 12288)>
#map29 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 14336)>
#map30 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
#map31 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 1048576 + s1 * 131072 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8) floordiv 256) * 1048576 + 128)>
#map32 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 1048576 + s1 * 131072 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 256) * 1048576 + 262272)>
#map33 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 1048576 + s1 * 131072 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 128) floordiv 256) * 1048576 + 524416)>
#map34 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 1048576 + s1 * 131072 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 192) floordiv 256) * 1048576 + 786560)>
#map35 = affine_map<()[s0, s1, s2] -> (s0 * 65536 + s1 * 4 + s2 * 256 + (s1 floordiv 64) * 16384 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 256)>
#map36 = affine_map<()[s0, s1, s2] -> (s0 * 65536 + s1 * 4 + s2 * 256 + (s1 floordiv 64) * 16384 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 8448)>
#map37 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 32768 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 - (s3 floordiv 16) * 64 + 256)>
#map38 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 32768 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 - (s3 floordiv 16) * 64 + 8448)>
#map39 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 32768 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 - (s3 floordiv 16) * 64 + 16640)>
#map40 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 32768 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 - (s3 floordiv 16) * 64 + 24832)>
#map41 = affine_map<()[s0] -> (s0 * 256)>
#map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
#map43 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
#map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map45 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map46 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map47 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
#map48 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
#map49 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
#map50 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
#map51 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
#map52 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
#map53 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
#map54 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map55 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map56 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map57 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#map58 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
#map59 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
#map60 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
#map61 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
#map62 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
#map63 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
#map64 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
#map65 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
#map_blk256 = affine_map<()[s0] -> (s0 * 256)>
#map_Nbase_ep = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 32) * 8)>
#map_Mbase_ep = affine_map<()[s0, s1] -> (s1 * 128 + s0 - (s0 floordiv 16) * 16)>
#map_lane_parity_ep = affine_map<()[s0] -> ((s0 mod 32) floordiv 16)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (index, index, index) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      stream.return %c4, %c4, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index) attributes {translation_info = #translation} {
        %c4_i32 = arith.constant 4 : i32
        %c4096_i14 = arith.constant 4096 : i14
        %c2147483645_i64 = arith.constant 2147483645 : i64
        %c31 = arith.constant 31 : index
        %c4194304_i64 = arith.constant 4194304 : i64
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
        %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<bf16>
        %block_id_x = gpu.block_id x upper_bound 4
        %block_id_y = gpu.block_id y upper_bound 4
        %thread_id_x = gpu.thread_id x upper_bound 256
        %thread_id_y = gpu.thread_id y upper_bound 2
        %reinterpret_cast = memref.reinterpret_cast %4 to offset: [0], sizes: [1024, 1024], strides: [%arg9, 1] : memref<bf16> to memref<1024x1024xbf16, strided<[?, 1]>>
        %alloc = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
        %alloc_1 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
        %alloc_2 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
        %5 = affine.apply #map()[%thread_id_x, %thread_id_y]
        %6 = gpu.subgroup_broadcast %5, first_active_lane : index
        %7 = gpu.subgroup_broadcast %c0, first_active_lane : index
        %reinterpret_cast_3 = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast = memref.cast %reinterpret_cast_3 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %8 = affine.apply #map1()[%thread_id_x]
        %9 = affine.apply #map2()[%thread_id_x]
        %10 = arith.xori %9, %8 : index
        %11 = affine.apply #map3()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        %12 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c4194304_i64) cacheSwizzleStride(%c4096_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %12[%11], %alloc_2[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %13 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %14 = gpu.subgroup_broadcast %13, first_active_lane : index
        %15 = affine.apply #map5()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%15], %alloc_2[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %16 = affine.apply #map6()[%thread_id_x, %thread_id_y]
        %17 = gpu.subgroup_broadcast %16, first_active_lane : index
        %18 = affine.apply #map7()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%18], %alloc_2[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %19 = affine.apply #map8()[%thread_id_x, %thread_id_y]
        %20 = gpu.subgroup_broadcast %19, first_active_lane : index
        %21 = affine.apply #map9()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%21], %alloc_2[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_4 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_5 = memref.cast %reinterpret_cast_4 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %22 = affine.apply #map3()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        %23 = amdgpu.fat_raw_buffer_cast %cast_5 validBytes(%c4194304_i64) cacheSwizzleStride(%c4096_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %23[%22], %alloc_0[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %24 = affine.apply #map5()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%24], %alloc_0[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %25 = affine.apply #map7()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%25], %alloc_0[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %26 = affine.apply #map9()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%26], %alloc_0[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_6 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %27 = affine.apply #map10()[%block_id_x, %thread_id_x]
        %28 = vector.load %reinterpret_cast_6[%27] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
        %29 = affine.apply #map11()[%block_id_x, %thread_id_x]
        %30 = vector.load %reinterpret_cast_6[%29] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
        %reinterpret_cast_7 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %31 = affine.apply #map12()[%block_id_y, %thread_id_y, %thread_id_x]
        %32 = vector.load %reinterpret_cast_7[%31] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
        %33 = affine.apply #map13()[%block_id_y, %thread_id_y, %thread_id_x]
        %34 = vector.load %reinterpret_cast_7[%33] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
        %35 = affine.apply #map14()[%block_id_y, %thread_id_y, %thread_id_x]
        %36 = vector.load %reinterpret_cast_7[%35] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
        %37 = affine.apply #map15()[%block_id_y, %thread_id_y, %thread_id_x]
        %38 = vector.load %reinterpret_cast_7[%37] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
        amdgpu.memory_counter_wait load(6)
        rocdl.s.barrier
        %39 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %40 = arith.index_cast %39 : index to i32
        %41 = arith.cmpi sge, %40, %c4_i32 : i32
        %42 = arith.cmpi slt, %40, %c4_i32 : i32
        scf.if %41 {
          rocdl.s.barrier
        }
        %43 = affine.apply #map17()[%thread_id_x]
        %44 = arith.xori %43, %9 : index
        %45 = affine.apply #map18()[%thread_id_x, %44]
        %46 = affine.apply #map19()[%thread_id_x, %44]
        %47 = affine.apply #map20()[%thread_id_x, %44]
        %48 = affine.apply #map21()[%thread_id_x, %44]
        %49 = affine.apply #map22()[%thread_id_y, %thread_id_x, %44]
        %50 = affine.apply #map23()[%thread_id_y, %thread_id_x, %44]
        %51 = affine.apply #map24()[%thread_id_y, %thread_id_x, %44]
        %52 = affine.apply #map25()[%thread_id_y, %thread_id_x, %44]
        %53 = affine.apply #map26()[%thread_id_y, %thread_id_x, %44]
        %54 = affine.apply #map27()[%thread_id_y, %thread_id_x, %44]
        %55 = affine.apply #map28()[%thread_id_y, %thread_id_x, %44]
        %56 = affine.apply #map29()[%thread_id_y, %thread_id_x, %44]
        %57 = affine.apply #map30()[%thread_id_x]
        %58 = arith.xori %57, %9 : index
        %59 = affine.apply #map18()[%thread_id_x, %58]
        %60 = affine.apply #map19()[%thread_id_x, %58]
        %61 = affine.apply #map20()[%thread_id_x, %58]
        %62 = affine.apply #map21()[%thread_id_x, %58]
        %63 = affine.apply #map22()[%thread_id_y, %thread_id_x, %58]
        %64 = affine.apply #map23()[%thread_id_y, %thread_id_x, %58]
        %65 = affine.apply #map24()[%thread_id_y, %thread_id_x, %58]
        %66 = affine.apply #map25()[%thread_id_y, %thread_id_x, %58]
        %67 = affine.apply #map26()[%thread_id_y, %thread_id_x, %58]
        %68 = affine.apply #map27()[%thread_id_y, %thread_id_x, %58]
        %69 = affine.apply #map28()[%thread_id_y, %thread_id_x, %58]
        %70 = affine.apply #map29()[%thread_id_y, %thread_id_x, %58]
        %71:42 = scf.for %arg10 = %c0 to %c31 step %c1 iter_args(%arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %cst, %arg39 = %cst, %arg40 = %cst, %arg41 = %cst, %arg42 = %cst, %arg43 = %28, %arg44 = %30, %arg45 = %32, %arg46 = %34, %arg47 = %36, %arg48 = %38, %arg49 = %alloc_2, %arg50 = %alloc_1, %arg51 = %alloc_0, %arg52 = %alloc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>) {
          %524 = vector.bitcast %arg44 : vector<4xi8> to vector<4xf8E8M0FNU>
          %525 = vector.bitcast %arg48 : vector<4xi8> to vector<4xf8E8M0FNU>
          %526 = vector.bitcast %arg47 : vector<4xi8> to vector<4xf8E8M0FNU>
          %527 = vector.bitcast %arg46 : vector<4xi8> to vector<4xf8E8M0FNU>
          %528 = vector.bitcast %arg45 : vector<4xi8> to vector<4xf8E8M0FNU>
          %529 = vector.bitcast %arg43 : vector<4xi8> to vector<4xf8E8M0FNU>
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(0)
          rocdl.s.barrier
          %530 = affine.apply #map31()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%530], %arg50[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %531 = affine.apply #map32()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%531], %arg50[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %532 = affine.apply #map33()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%532], %arg50[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %533 = affine.apply #map34()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%533], %arg50[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %534 = affine.apply #map31()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %23[%534], %arg52[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %535 = affine.apply #map32()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %23[%535], %arg52[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %536 = affine.apply #map33()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %23[%536], %arg52[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %537 = affine.apply #map34()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %23[%537], %arg52[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          rocdl.sched.barrier 0
          %reinterpret_cast_12 = memref.reinterpret_cast %arg49 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
          %538 = vector.load %reinterpret_cast_12[%45] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %539 = vector.load %reinterpret_cast_12[%46] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %540 = vector.load %reinterpret_cast_12[%47] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %541 = vector.load %reinterpret_cast_12[%48] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %reinterpret_cast_13 = memref.reinterpret_cast %arg51 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
          %542 = vector.load %reinterpret_cast_13[%49] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %543 = vector.load %reinterpret_cast_13[%50] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %544 = vector.load %reinterpret_cast_13[%51] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %545 = vector.load %reinterpret_cast_13[%52] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %546 = vector.load %reinterpret_cast_13[%53] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %547 = vector.load %reinterpret_cast_13[%54] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %548 = vector.load %reinterpret_cast_13[%55] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %549 = vector.load %reinterpret_cast_13[%56] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %550 = vector.bitcast %538 : vector<16xi8> to vector<32xf4E2M1FN>
          %551 = vector.bitcast %539 : vector<16xi8> to vector<32xf4E2M1FN>
          %552 = vector.bitcast %540 : vector<16xi8> to vector<32xf4E2M1FN>
          %553 = vector.bitcast %541 : vector<16xi8> to vector<32xf4E2M1FN>
          %554 = vector.bitcast %542 : vector<16xi8> to vector<32xf4E2M1FN>
          %555 = vector.bitcast %543 : vector<16xi8> to vector<32xf4E2M1FN>
          %556 = vector.bitcast %544 : vector<16xi8> to vector<32xf4E2M1FN>
          %557 = vector.bitcast %545 : vector<16xi8> to vector<32xf4E2M1FN>
          %558 = vector.bitcast %546 : vector<16xi8> to vector<32xf4E2M1FN>
          %559 = vector.bitcast %547 : vector<16xi8> to vector<32xf4E2M1FN>
          %560 = vector.bitcast %548 : vector<16xi8> to vector<32xf4E2M1FN>
          %561 = vector.bitcast %549 : vector<16xi8> to vector<32xf4E2M1FN>
          %562 = affine.apply #map35()[%block_id_x, %thread_id_x, %arg10]
          %563 = vector.load %reinterpret_cast_6[%562] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
          %564 = affine.apply #map36()[%block_id_x, %thread_id_x, %arg10]
          %565 = vector.load %reinterpret_cast_6[%564] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
          %566 = affine.apply #map37()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %567 = vector.load %reinterpret_cast_7[%566] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
          %568 = affine.apply #map38()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %569 = vector.load %reinterpret_cast_7[%568] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
          %570 = affine.apply #map39()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %571 = vector.load %reinterpret_cast_7[%570] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
          %572 = affine.apply #map40()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %573 = vector.load %reinterpret_cast_7[%572] : memref<2147483646xi8, strided<[1]>>, vector<4xi8>
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %574 = amdgpu.scaled_mfma 16x16x128 (%529[0] * %550) * (%528[0] * %554) + %arg11 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %575 = amdgpu.scaled_mfma 16x16x128 (%529[0] * %550) * (%528[1] * %555) + %arg12 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %576 = amdgpu.scaled_mfma 16x16x128 (%529[0] * %550) * (%527[0] * %556) + %arg13 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %577 = amdgpu.scaled_mfma 16x16x128 (%529[0] * %550) * (%527[1] * %557) + %arg14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %578 = amdgpu.scaled_mfma 16x16x128 (%529[0] * %550) * (%526[0] * %558) + %arg15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %579 = amdgpu.scaled_mfma 16x16x128 (%529[0] * %550) * (%526[1] * %559) + %arg16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %580 = amdgpu.scaled_mfma 16x16x128 (%529[0] * %550) * (%525[0] * %560) + %arg17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %581 = amdgpu.scaled_mfma 16x16x128 (%529[0] * %550) * (%525[1] * %561) + %arg18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %582 = amdgpu.scaled_mfma 16x16x128 (%529[1] * %551) * (%528[0] * %554) + %arg19 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %583 = amdgpu.scaled_mfma 16x16x128 (%529[1] * %551) * (%528[1] * %555) + %arg20 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %584 = amdgpu.scaled_mfma 16x16x128 (%529[1] * %551) * (%527[0] * %556) + %arg21 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %585 = amdgpu.scaled_mfma 16x16x128 (%529[1] * %551) * (%527[1] * %557) + %arg22 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %586 = amdgpu.scaled_mfma 16x16x128 (%529[1] * %551) * (%526[0] * %558) + %arg23 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %587 = amdgpu.scaled_mfma 16x16x128 (%529[1] * %551) * (%526[1] * %559) + %arg24 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %588 = amdgpu.scaled_mfma 16x16x128 (%529[1] * %551) * (%525[0] * %560) + %arg25 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %589 = amdgpu.scaled_mfma 16x16x128 (%529[1] * %551) * (%525[1] * %561) + %arg26 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %590 = amdgpu.scaled_mfma 16x16x128 (%524[0] * %552) * (%528[0] * %554) + %arg27 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %591 = amdgpu.scaled_mfma 16x16x128 (%524[0] * %552) * (%528[1] * %555) + %arg28 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %592 = amdgpu.scaled_mfma 16x16x128 (%524[0] * %552) * (%527[0] * %556) + %arg29 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %593 = amdgpu.scaled_mfma 16x16x128 (%524[0] * %552) * (%527[1] * %557) + %arg30 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %594 = amdgpu.scaled_mfma 16x16x128 (%524[0] * %552) * (%526[0] * %558) + %arg31 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %595 = amdgpu.scaled_mfma 16x16x128 (%524[0] * %552) * (%526[1] * %559) + %arg32 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %596 = amdgpu.scaled_mfma 16x16x128 (%524[0] * %552) * (%525[0] * %560) + %arg33 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %597 = amdgpu.scaled_mfma 16x16x128 (%524[0] * %552) * (%525[1] * %561) + %arg34 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %598 = amdgpu.scaled_mfma 16x16x128 (%524[1] * %553) * (%528[0] * %554) + %arg35 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %599 = amdgpu.scaled_mfma 16x16x128 (%524[1] * %553) * (%528[1] * %555) + %arg36 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %600 = amdgpu.scaled_mfma 16x16x128 (%524[1] * %553) * (%527[0] * %556) + %arg37 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %601 = amdgpu.scaled_mfma 16x16x128 (%524[1] * %553) * (%527[1] * %557) + %arg38 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %602 = amdgpu.scaled_mfma 16x16x128 (%524[1] * %553) * (%526[0] * %558) + %arg39 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %603 = amdgpu.scaled_mfma 16x16x128 (%524[1] * %553) * (%526[1] * %559) + %arg40 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %604 = amdgpu.scaled_mfma 16x16x128 (%524[1] * %553) * (%525[0] * %560) + %arg41 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %605 = amdgpu.scaled_mfma 16x16x128 (%524[1] * %553) * (%525[1] * %561) + %arg42 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.sched.barrier 0
          %606 = vector.load %reinterpret_cast_12[%59] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %607 = vector.load %reinterpret_cast_12[%60] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %608 = vector.load %reinterpret_cast_12[%61] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %609 = vector.load %reinterpret_cast_12[%62] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %610 = vector.load %reinterpret_cast_13[%63] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %611 = vector.load %reinterpret_cast_13[%64] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %612 = vector.load %reinterpret_cast_13[%65] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %613 = vector.load %reinterpret_cast_13[%66] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %614 = vector.load %reinterpret_cast_13[%67] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %615 = vector.load %reinterpret_cast_13[%68] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %616 = vector.load %reinterpret_cast_13[%69] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %617 = vector.load %reinterpret_cast_13[%70] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %618 = vector.bitcast %606 : vector<16xi8> to vector<32xf4E2M1FN>
          %619 = vector.bitcast %607 : vector<16xi8> to vector<32xf4E2M1FN>
          %620 = vector.bitcast %608 : vector<16xi8> to vector<32xf4E2M1FN>
          %621 = vector.bitcast %609 : vector<16xi8> to vector<32xf4E2M1FN>
          %622 = vector.bitcast %610 : vector<16xi8> to vector<32xf4E2M1FN>
          %623 = vector.bitcast %611 : vector<16xi8> to vector<32xf4E2M1FN>
          %624 = vector.bitcast %612 : vector<16xi8> to vector<32xf4E2M1FN>
          %625 = vector.bitcast %613 : vector<16xi8> to vector<32xf4E2M1FN>
          %626 = vector.bitcast %614 : vector<16xi8> to vector<32xf4E2M1FN>
          %627 = vector.bitcast %615 : vector<16xi8> to vector<32xf4E2M1FN>
          %628 = vector.bitcast %616 : vector<16xi8> to vector<32xf4E2M1FN>
          %629 = vector.bitcast %617 : vector<16xi8> to vector<32xf4E2M1FN>
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(6)
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %630 = amdgpu.scaled_mfma 16x16x128 (%529[2] * %618) * (%528[2] * %622) + %574 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %631 = amdgpu.scaled_mfma 16x16x128 (%529[2] * %618) * (%528[3] * %623) + %575 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %632 = amdgpu.scaled_mfma 16x16x128 (%529[2] * %618) * (%527[2] * %624) + %576 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %633 = amdgpu.scaled_mfma 16x16x128 (%529[2] * %618) * (%527[3] * %625) + %577 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %634 = amdgpu.scaled_mfma 16x16x128 (%529[2] * %618) * (%526[2] * %626) + %578 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %635 = amdgpu.scaled_mfma 16x16x128 (%529[2] * %618) * (%526[3] * %627) + %579 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %636 = amdgpu.scaled_mfma 16x16x128 (%529[2] * %618) * (%525[2] * %628) + %580 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %637 = amdgpu.scaled_mfma 16x16x128 (%529[2] * %618) * (%525[3] * %629) + %581 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %638 = amdgpu.scaled_mfma 16x16x128 (%529[3] * %619) * (%528[2] * %622) + %582 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %639 = amdgpu.scaled_mfma 16x16x128 (%529[3] * %619) * (%528[3] * %623) + %583 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %640 = amdgpu.scaled_mfma 16x16x128 (%529[3] * %619) * (%527[2] * %624) + %584 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %641 = amdgpu.scaled_mfma 16x16x128 (%529[3] * %619) * (%527[3] * %625) + %585 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %642 = amdgpu.scaled_mfma 16x16x128 (%529[3] * %619) * (%526[2] * %626) + %586 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %643 = amdgpu.scaled_mfma 16x16x128 (%529[3] * %619) * (%526[3] * %627) + %587 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %644 = amdgpu.scaled_mfma 16x16x128 (%529[3] * %619) * (%525[2] * %628) + %588 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %645 = amdgpu.scaled_mfma 16x16x128 (%529[3] * %619) * (%525[3] * %629) + %589 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %646 = amdgpu.scaled_mfma 16x16x128 (%524[2] * %620) * (%528[2] * %622) + %590 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %647 = amdgpu.scaled_mfma 16x16x128 (%524[2] * %620) * (%528[3] * %623) + %591 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %648 = amdgpu.scaled_mfma 16x16x128 (%524[2] * %620) * (%527[2] * %624) + %592 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %649 = amdgpu.scaled_mfma 16x16x128 (%524[2] * %620) * (%527[3] * %625) + %593 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %650 = amdgpu.scaled_mfma 16x16x128 (%524[2] * %620) * (%526[2] * %626) + %594 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %651 = amdgpu.scaled_mfma 16x16x128 (%524[2] * %620) * (%526[3] * %627) + %595 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %652 = amdgpu.scaled_mfma 16x16x128 (%524[2] * %620) * (%525[2] * %628) + %596 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %653 = amdgpu.scaled_mfma 16x16x128 (%524[2] * %620) * (%525[3] * %629) + %597 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %654 = amdgpu.scaled_mfma 16x16x128 (%524[3] * %621) * (%528[2] * %622) + %598 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %655 = amdgpu.scaled_mfma 16x16x128 (%524[3] * %621) * (%528[3] * %623) + %599 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %656 = amdgpu.scaled_mfma 16x16x128 (%524[3] * %621) * (%527[2] * %624) + %600 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %657 = amdgpu.scaled_mfma 16x16x128 (%524[3] * %621) * (%527[3] * %625) + %601 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %658 = amdgpu.scaled_mfma 16x16x128 (%524[3] * %621) * (%526[2] * %626) + %602 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %659 = amdgpu.scaled_mfma 16x16x128 (%524[3] * %621) * (%526[3] * %627) + %603 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %660 = amdgpu.scaled_mfma 16x16x128 (%524[3] * %621) * (%525[2] * %628) + %604 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %661 = amdgpu.scaled_mfma 16x16x128 (%524[3] * %621) * (%525[3] * %629) + %605 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          scf.yield %630, %631, %632, %633, %634, %635, %636, %637, %638, %639, %640, %641, %642, %643, %644, %645, %646, %647, %648, %649, %650, %651, %652, %653, %654, %655, %656, %657, %658, %659, %660, %661, %563, %565, %567, %569, %571, %573, %arg50, %arg49, %arg52, %arg51 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        }
        %72 = vector.bitcast %71#33 : vector<4xi8> to vector<4xf8E8M0FNU>
        %73 = vector.bitcast %71#37 : vector<4xi8> to vector<4xf8E8M0FNU>
        %74 = vector.bitcast %71#36 : vector<4xi8> to vector<4xf8E8M0FNU>
        %75 = vector.bitcast %71#35 : vector<4xi8> to vector<4xf8E8M0FNU>
        %76 = vector.bitcast %71#34 : vector<4xi8> to vector<4xf8E8M0FNU>
        %77 = vector.bitcast %71#32 : vector<4xi8> to vector<4xf8E8M0FNU>
        scf.if %42 {
          rocdl.s.barrier
        }
        amdgpu.lds_barrier
        %reinterpret_cast_8 = memref.reinterpret_cast %71#40 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
        %78 = vector.load %reinterpret_cast_8[%49] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %79 = vector.load %reinterpret_cast_8[%63] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %80 = vector.load %reinterpret_cast_8[%50] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %81 = vector.load %reinterpret_cast_8[%64] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %82 = vector.load %reinterpret_cast_8[%51] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %83 = vector.load %reinterpret_cast_8[%65] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %84 = vector.load %reinterpret_cast_8[%52] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %85 = vector.load %reinterpret_cast_8[%66] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %86 = vector.load %reinterpret_cast_8[%53] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %87 = vector.load %reinterpret_cast_8[%67] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %88 = vector.load %reinterpret_cast_8[%54] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %89 = vector.load %reinterpret_cast_8[%68] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %90 = vector.load %reinterpret_cast_8[%55] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %91 = vector.load %reinterpret_cast_8[%69] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %92 = vector.load %reinterpret_cast_8[%56] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %93 = vector.load %reinterpret_cast_8[%70] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %reinterpret_cast_9 = memref.reinterpret_cast %71#38 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
        %94 = vector.load %reinterpret_cast_9[%45] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %95 = vector.load %reinterpret_cast_9[%59] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %96 = vector.load %reinterpret_cast_9[%46] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %97 = vector.load %reinterpret_cast_9[%60] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %98 = vector.load %reinterpret_cast_9[%47] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %99 = vector.load %reinterpret_cast_9[%61] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %100 = vector.load %reinterpret_cast_9[%48] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %101 = vector.load %reinterpret_cast_9[%62] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %102 = vector.bitcast %94 : vector<16xi8> to vector<32xf4E2M1FN>
        %103 = vector.bitcast %95 : vector<16xi8> to vector<32xf4E2M1FN>
        %104 = vector.bitcast %96 : vector<16xi8> to vector<32xf4E2M1FN>
        %105 = vector.bitcast %97 : vector<16xi8> to vector<32xf4E2M1FN>
        %106 = vector.bitcast %98 : vector<16xi8> to vector<32xf4E2M1FN>
        %107 = vector.bitcast %99 : vector<16xi8> to vector<32xf4E2M1FN>
        %108 = vector.bitcast %100 : vector<16xi8> to vector<32xf4E2M1FN>
        %109 = vector.bitcast %101 : vector<16xi8> to vector<32xf4E2M1FN>
        %110 = vector.bitcast %78 : vector<16xi8> to vector<32xf4E2M1FN>
        %111 = vector.bitcast %79 : vector<16xi8> to vector<32xf4E2M1FN>
        %112 = vector.bitcast %80 : vector<16xi8> to vector<32xf4E2M1FN>
        %113 = vector.bitcast %81 : vector<16xi8> to vector<32xf4E2M1FN>
        %114 = vector.bitcast %82 : vector<16xi8> to vector<32xf4E2M1FN>
        %115 = vector.bitcast %83 : vector<16xi8> to vector<32xf4E2M1FN>
        %116 = vector.bitcast %84 : vector<16xi8> to vector<32xf4E2M1FN>
        %117 = vector.bitcast %85 : vector<16xi8> to vector<32xf4E2M1FN>
        %118 = vector.bitcast %86 : vector<16xi8> to vector<32xf4E2M1FN>
        %119 = vector.bitcast %87 : vector<16xi8> to vector<32xf4E2M1FN>
        %120 = vector.bitcast %88 : vector<16xi8> to vector<32xf4E2M1FN>
        %121 = vector.bitcast %89 : vector<16xi8> to vector<32xf4E2M1FN>
        %122 = vector.bitcast %90 : vector<16xi8> to vector<32xf4E2M1FN>
        %123 = vector.bitcast %91 : vector<16xi8> to vector<32xf4E2M1FN>
        %124 = vector.bitcast %92 : vector<16xi8> to vector<32xf4E2M1FN>
        %125 = vector.bitcast %93 : vector<16xi8> to vector<32xf4E2M1FN>
        %126 = amdgpu.scaled_mfma 16x16x128 (%77[0] * %102) * (%76[0] * %110) + %71#0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %127 = amdgpu.scaled_mfma 16x16x128 (%77[2] * %103) * (%76[2] * %111) + %126 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %128 = amdgpu.scaled_mfma 16x16x128 (%77[0] * %102) * (%76[1] * %112) + %71#1 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %129 = amdgpu.scaled_mfma 16x16x128 (%77[2] * %103) * (%76[3] * %113) + %128 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %130 = amdgpu.scaled_mfma 16x16x128 (%77[0] * %102) * (%75[0] * %114) + %71#2 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %131 = amdgpu.scaled_mfma 16x16x128 (%77[2] * %103) * (%75[2] * %115) + %130 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %132 = amdgpu.scaled_mfma 16x16x128 (%77[0] * %102) * (%75[1] * %116) + %71#3 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %133 = amdgpu.scaled_mfma 16x16x128 (%77[2] * %103) * (%75[3] * %117) + %132 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %134 = amdgpu.scaled_mfma 16x16x128 (%77[0] * %102) * (%74[0] * %118) + %71#4 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %135 = amdgpu.scaled_mfma 16x16x128 (%77[2] * %103) * (%74[2] * %119) + %134 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %136 = amdgpu.scaled_mfma 16x16x128 (%77[0] * %102) * (%74[1] * %120) + %71#5 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %137 = amdgpu.scaled_mfma 16x16x128 (%77[2] * %103) * (%74[3] * %121) + %136 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %138 = amdgpu.scaled_mfma 16x16x128 (%77[0] * %102) * (%73[0] * %122) + %71#6 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %139 = amdgpu.scaled_mfma 16x16x128 (%77[2] * %103) * (%73[2] * %123) + %138 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %140 = amdgpu.scaled_mfma 16x16x128 (%77[0] * %102) * (%73[1] * %124) + %71#7 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %141 = amdgpu.scaled_mfma 16x16x128 (%77[2] * %103) * (%73[3] * %125) + %140 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %142 = amdgpu.scaled_mfma 16x16x128 (%77[1] * %104) * (%76[0] * %110) + %71#8 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %143 = amdgpu.scaled_mfma 16x16x128 (%77[3] * %105) * (%76[2] * %111) + %142 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %144 = amdgpu.scaled_mfma 16x16x128 (%77[1] * %104) * (%76[1] * %112) + %71#9 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %145 = amdgpu.scaled_mfma 16x16x128 (%77[3] * %105) * (%76[3] * %113) + %144 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %146 = amdgpu.scaled_mfma 16x16x128 (%77[1] * %104) * (%75[0] * %114) + %71#10 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %147 = amdgpu.scaled_mfma 16x16x128 (%77[3] * %105) * (%75[2] * %115) + %146 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %148 = amdgpu.scaled_mfma 16x16x128 (%77[1] * %104) * (%75[1] * %116) + %71#11 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %149 = amdgpu.scaled_mfma 16x16x128 (%77[3] * %105) * (%75[3] * %117) + %148 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %150 = amdgpu.scaled_mfma 16x16x128 (%77[1] * %104) * (%74[0] * %118) + %71#12 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %151 = amdgpu.scaled_mfma 16x16x128 (%77[3] * %105) * (%74[2] * %119) + %150 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %152 = amdgpu.scaled_mfma 16x16x128 (%77[1] * %104) * (%74[1] * %120) + %71#13 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %153 = amdgpu.scaled_mfma 16x16x128 (%77[3] * %105) * (%74[3] * %121) + %152 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %154 = amdgpu.scaled_mfma 16x16x128 (%77[1] * %104) * (%73[0] * %122) + %71#14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %155 = amdgpu.scaled_mfma 16x16x128 (%77[3] * %105) * (%73[2] * %123) + %154 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %156 = amdgpu.scaled_mfma 16x16x128 (%77[1] * %104) * (%73[1] * %124) + %71#15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %157 = amdgpu.scaled_mfma 16x16x128 (%77[3] * %105) * (%73[3] * %125) + %156 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %158 = amdgpu.scaled_mfma 16x16x128 (%72[0] * %106) * (%76[0] * %110) + %71#16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %159 = amdgpu.scaled_mfma 16x16x128 (%72[2] * %107) * (%76[2] * %111) + %158 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %160 = amdgpu.scaled_mfma 16x16x128 (%72[0] * %106) * (%76[1] * %112) + %71#17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %161 = amdgpu.scaled_mfma 16x16x128 (%72[2] * %107) * (%76[3] * %113) + %160 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %162 = amdgpu.scaled_mfma 16x16x128 (%72[0] * %106) * (%75[0] * %114) + %71#18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %163 = amdgpu.scaled_mfma 16x16x128 (%72[2] * %107) * (%75[2] * %115) + %162 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %164 = amdgpu.scaled_mfma 16x16x128 (%72[0] * %106) * (%75[1] * %116) + %71#19 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %165 = amdgpu.scaled_mfma 16x16x128 (%72[2] * %107) * (%75[3] * %117) + %164 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %166 = amdgpu.scaled_mfma 16x16x128 (%72[0] * %106) * (%74[0] * %118) + %71#20 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %167 = amdgpu.scaled_mfma 16x16x128 (%72[2] * %107) * (%74[2] * %119) + %166 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %168 = amdgpu.scaled_mfma 16x16x128 (%72[0] * %106) * (%74[1] * %120) + %71#21 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %169 = amdgpu.scaled_mfma 16x16x128 (%72[2] * %107) * (%74[3] * %121) + %168 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %170 = amdgpu.scaled_mfma 16x16x128 (%72[0] * %106) * (%73[0] * %122) + %71#22 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %171 = amdgpu.scaled_mfma 16x16x128 (%72[2] * %107) * (%73[2] * %123) + %170 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %172 = amdgpu.scaled_mfma 16x16x128 (%72[0] * %106) * (%73[1] * %124) + %71#23 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %173 = amdgpu.scaled_mfma 16x16x128 (%72[2] * %107) * (%73[3] * %125) + %172 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %174 = amdgpu.scaled_mfma 16x16x128 (%72[1] * %108) * (%76[0] * %110) + %71#24 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %175 = amdgpu.scaled_mfma 16x16x128 (%72[3] * %109) * (%76[2] * %111) + %174 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %176 = amdgpu.scaled_mfma 16x16x128 (%72[1] * %108) * (%76[1] * %112) + %71#25 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %177 = amdgpu.scaled_mfma 16x16x128 (%72[3] * %109) * (%76[3] * %113) + %176 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %178 = amdgpu.scaled_mfma 16x16x128 (%72[1] * %108) * (%75[0] * %114) + %71#26 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %179 = amdgpu.scaled_mfma 16x16x128 (%72[3] * %109) * (%75[2] * %115) + %178 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %180 = amdgpu.scaled_mfma 16x16x128 (%72[1] * %108) * (%75[1] * %116) + %71#27 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %181 = amdgpu.scaled_mfma 16x16x128 (%72[3] * %109) * (%75[3] * %117) + %180 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %182 = amdgpu.scaled_mfma 16x16x128 (%72[1] * %108) * (%74[0] * %118) + %71#28 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %183 = amdgpu.scaled_mfma 16x16x128 (%72[3] * %109) * (%74[2] * %119) + %182 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %184 = amdgpu.scaled_mfma 16x16x128 (%72[1] * %108) * (%74[1] * %120) + %71#29 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %185 = amdgpu.scaled_mfma 16x16x128 (%72[3] * %109) * (%74[3] * %121) + %184 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %186 = amdgpu.scaled_mfma 16x16x128 (%72[1] * %108) * (%73[0] * %122) + %71#30 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %187 = amdgpu.scaled_mfma 16x16x128 (%72[3] * %109) * (%73[2] * %123) + %186 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %188 = amdgpu.scaled_mfma 16x16x128 (%72[1] * %108) * (%73[1] * %124) + %71#31 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %189 = amdgpu.scaled_mfma 16x16x128 (%72[3] * %109) * (%73[3] * %125) + %188 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %190 = arith.truncf %127 : vector<4xf32> to vector<4xbf16>
        %191 = arith.truncf %129 : vector<4xf32> to vector<4xbf16>
        %192 = arith.truncf %131 : vector<4xf32> to vector<4xbf16>
        %193 = arith.truncf %133 : vector<4xf32> to vector<4xbf16>
        %194 = arith.truncf %135 : vector<4xf32> to vector<4xbf16>
        %195 = arith.truncf %137 : vector<4xf32> to vector<4xbf16>
        %196 = arith.truncf %139 : vector<4xf32> to vector<4xbf16>
        %197 = arith.truncf %141 : vector<4xf32> to vector<4xbf16>
        %198 = arith.truncf %143 : vector<4xf32> to vector<4xbf16>
        %199 = arith.truncf %145 : vector<4xf32> to vector<4xbf16>
        %200 = arith.truncf %147 : vector<4xf32> to vector<4xbf16>
        %201 = arith.truncf %149 : vector<4xf32> to vector<4xbf16>
        %202 = arith.truncf %151 : vector<4xf32> to vector<4xbf16>
        %203 = arith.truncf %153 : vector<4xf32> to vector<4xbf16>
        %204 = arith.truncf %155 : vector<4xf32> to vector<4xbf16>
        %205 = arith.truncf %157 : vector<4xf32> to vector<4xbf16>
        %206 = arith.truncf %159 : vector<4xf32> to vector<4xbf16>
        %207 = arith.truncf %161 : vector<4xf32> to vector<4xbf16>
        %208 = arith.truncf %163 : vector<4xf32> to vector<4xbf16>
        %209 = arith.truncf %165 : vector<4xf32> to vector<4xbf16>
        %210 = arith.truncf %167 : vector<4xf32> to vector<4xbf16>
        %211 = arith.truncf %169 : vector<4xf32> to vector<4xbf16>
        %212 = arith.truncf %171 : vector<4xf32> to vector<4xbf16>
        %213 = arith.truncf %173 : vector<4xf32> to vector<4xbf16>
        %214 = arith.truncf %175 : vector<4xf32> to vector<4xbf16>
        %215 = arith.truncf %177 : vector<4xf32> to vector<4xbf16>
        %216 = arith.truncf %179 : vector<4xf32> to vector<4xbf16>
        %217 = arith.truncf %181 : vector<4xf32> to vector<4xbf16>
        %218 = arith.truncf %183 : vector<4xf32> to vector<4xbf16>
        %219 = arith.truncf %185 : vector<4xf32> to vector<4xbf16>
        %220 = arith.truncf %187 : vector<4xf32> to vector<4xbf16>
        %221 = arith.truncf %189 : vector<4xf32> to vector<4xbf16>
        // --- Optimized permlane epilogue: 16 x vector<8xbf16> stores ---
        // Setup block base address for the fat_raw_buffer.
        // Writing C[M, N]: row = M-block (block_id_y), col = N-block (block_id_x).
        // (The kernel is set up as C^T so block_id_x=N-block, block_id_y=M-block.)
        %ep_blk_row = affine.apply #map_blk256()[%block_id_y]
        %ep_blk_col = affine.apply #map_blk256()[%block_id_x]
        %ep_base_buf, %ep_base_off, %ep_sizes:2, %ep_strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<1024x1024xbf16, strided<[?, 1]>> -> memref<bf16>, index, index, index, index, index
        %ep_blk_row_off = arith.muli %ep_blk_row, %ep_strides#0 overflow<nsw> : index
        %ep_blk_base = arith.addi %ep_blk_row_off, %ep_blk_col overflow<nsw> : index
        %ep_recast = memref.reinterpret_cast %4 to offset: [%ep_blk_base], sizes: [1073741822], strides: [1] : memref<bf16> to memref<1073741822xbf16, strided<[1], offset: ?>>
        %ep_cast = memref.cast %ep_recast : memref<1073741822xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
        %ep_stride_i14 = arith.index_cast %ep_strides#0 : index to i14
        %ep_buf = amdgpu.fat_raw_buffer_cast %ep_cast validBytes(%c2147483645_i64) cacheSwizzleStride(%ep_stride_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
        // Even/odd lane parity: (thread_id_x mod 32) floordiv 16 == 0
        %ep_lane_par = affine.apply #map_lane_parity_ep()[%thread_id_x]
        %ep_is_even = arith.cmpi eq, %ep_lane_par, %c0 : index
        // Mbase: row in the 256-row output tile owned by this thread
        %ep_mbase = affine.apply #map_Mbase_ep()[%thread_id_x, %thread_id_y]
        // Nbase: column base of the 8-wide store group (aligned to 8)
        %ep_nbase = affine.apply #map_Nbase_ep()[%thread_id_x]
        %ep_mrow = arith.muli %ep_mbase, %ep_strides#0 overflow<nsw> : index
        %ep_bn0 = arith.addi %ep_mrow, %ep_nbase overflow<nsw> : index
        // N-column group offsets (4 groups, stride 16 in N)
        %ep_c16 = arith.constant 16 : index
        %ep_c32 = arith.constant 32 : index
        %ep_c48 = arith.constant 48 : index
        %ep_c64 = arith.constant 64 : index
        %ep_c80 = arith.constant 80 : index
        %ep_c96 = arith.constant 96 : index
        %ep_c112 = arith.constant 112 : index
        %ep_bn1 = arith.addi %ep_bn0, %ep_c16 overflow<nsw> : index
        %ep_bn2 = arith.addi %ep_bn0, %ep_c32 overflow<nsw> : index
        %ep_bn3 = arith.addi %ep_bn0, %ep_c48 overflow<nsw> : index
        // M-row stride offsets: even lane → mt0/mt2/mt4/mt6; odd lane → mt1/mt3/mt5/mt7
        %ep_mt0 = arith.constant 0 : index
        %ep_mt1 = arith.muli %ep_c16, %ep_strides#0 overflow<nsw> : index
        %ep_mt2 = arith.muli %ep_c32, %ep_strides#0 overflow<nsw> : index
        %ep_mt3 = arith.muli %ep_c48, %ep_strides#0 overflow<nsw> : index
        %ep_mt4 = arith.muli %ep_c64, %ep_strides#0 overflow<nsw> : index
        %ep_mt5 = arith.muli %ep_c80, %ep_strides#0 overflow<nsw> : index
        %ep_mt6 = arith.muli %ep_c96, %ep_strides#0 overflow<nsw> : index
        %ep_mt7 = arith.muli %ep_c112, %ep_strides#0 overflow<nsw> : index
        // --- Precompute all store addresses (4 row-selects hoisted, then 16 addr adds) ---
        // 4 unique row-offset patterns (even/odd selects computed once each)
        %ep_rsel0 = arith.select %ep_is_even, %ep_mt0, %ep_mt1 : index
        %ep_rsel1 = arith.select %ep_is_even, %ep_mt2, %ep_mt3 : index
        %ep_rsel2 = arith.select %ep_is_even, %ep_mt4, %ep_mt5 : index
        %ep_rsel3 = arith.select %ep_is_even, %ep_mt6, %ep_mt7 : index
        // 16 store addresses: bn{0..3} x rsel{0..3}
        %ep_addr_0_0 = arith.addi %ep_bn0, %ep_rsel0 overflow<nsw> : index
        %ep_addr_0_1 = arith.addi %ep_bn0, %ep_rsel1 overflow<nsw> : index
        %ep_addr_0_2 = arith.addi %ep_bn0, %ep_rsel2 overflow<nsw> : index
        %ep_addr_0_3 = arith.addi %ep_bn0, %ep_rsel3 overflow<nsw> : index
        %ep_addr_1_0 = arith.addi %ep_bn1, %ep_rsel0 overflow<nsw> : index
        %ep_addr_1_1 = arith.addi %ep_bn1, %ep_rsel1 overflow<nsw> : index
        %ep_addr_1_2 = arith.addi %ep_bn1, %ep_rsel2 overflow<nsw> : index
        %ep_addr_1_3 = arith.addi %ep_bn1, %ep_rsel3 overflow<nsw> : index
        %ep_addr_2_0 = arith.addi %ep_bn2, %ep_rsel0 overflow<nsw> : index
        %ep_addr_2_1 = arith.addi %ep_bn2, %ep_rsel1 overflow<nsw> : index
        %ep_addr_2_2 = arith.addi %ep_bn2, %ep_rsel2 overflow<nsw> : index
        %ep_addr_2_3 = arith.addi %ep_bn2, %ep_rsel3 overflow<nsw> : index
        %ep_addr_3_0 = arith.addi %ep_bn3, %ep_rsel0 overflow<nsw> : index
        %ep_addr_3_1 = arith.addi %ep_bn3, %ep_rsel1 overflow<nsw> : index
        %ep_addr_3_2 = arith.addi %ep_bn3, %ep_rsel2 overflow<nsw> : index
        %ep_addr_3_3 = arith.addi %ep_bn3, %ep_rsel3 overflow<nsw> : index
        // === N-group 0 (bn0): bf16 pairs %190/%191 .. %196/%197 ===
        // Pair 0: accumulators %190, %191 → bn0, M-row-pair 0
        %ep_p0_swa = amdgpu.permlane_swap %190 16 : vector<4xbf16>
        %ep_p0_swb = amdgpu.permlane_swap %191 16 : vector<4xbf16>
        %ep_p0_hi = vector.shuffle %190, %ep_p0_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p0_lo = vector.shuffle %ep_p0_swb, %191 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p0_vec = arith.select %ep_is_even, %ep_p0_hi, %ep_p0_lo : vector<8xbf16>
        vector.store %ep_p0_vec, %ep_buf[%ep_addr_0_0] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 1: accumulators %192, %193 → bn0, M-row-pair 1
        %ep_p1_swa = amdgpu.permlane_swap %192 16 : vector<4xbf16>
        %ep_p1_swb = amdgpu.permlane_swap %193 16 : vector<4xbf16>
        %ep_p1_hi = vector.shuffle %192, %ep_p1_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p1_lo = vector.shuffle %ep_p1_swb, %193 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p1_vec = arith.select %ep_is_even, %ep_p1_hi, %ep_p1_lo : vector<8xbf16>
        vector.store %ep_p1_vec, %ep_buf[%ep_addr_0_1] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 2: accumulators %194, %195 → bn0, M-row-pair 2
        %ep_p2_swa = amdgpu.permlane_swap %194 16 : vector<4xbf16>
        %ep_p2_swb = amdgpu.permlane_swap %195 16 : vector<4xbf16>
        %ep_p2_hi = vector.shuffle %194, %ep_p2_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p2_lo = vector.shuffle %ep_p2_swb, %195 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p2_vec = arith.select %ep_is_even, %ep_p2_hi, %ep_p2_lo : vector<8xbf16>
        vector.store %ep_p2_vec, %ep_buf[%ep_addr_0_2] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 3: accumulators %196, %197 → bn0, M-row-pair 3
        %ep_p3_swa = amdgpu.permlane_swap %196 16 : vector<4xbf16>
        %ep_p3_swb = amdgpu.permlane_swap %197 16 : vector<4xbf16>
        %ep_p3_hi = vector.shuffle %196, %ep_p3_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p3_lo = vector.shuffle %ep_p3_swb, %197 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p3_vec = arith.select %ep_is_even, %ep_p3_hi, %ep_p3_lo : vector<8xbf16>
        vector.store %ep_p3_vec, %ep_buf[%ep_addr_0_3] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // === N-group 1 (bn1): bf16 pairs %198/%199 .. %204/%205 ===
        // Pair 4: accumulators %198, %199 → bn1, M-row-pair 0
        %ep_p4_swa = amdgpu.permlane_swap %198 16 : vector<4xbf16>
        %ep_p4_swb = amdgpu.permlane_swap %199 16 : vector<4xbf16>
        %ep_p4_hi = vector.shuffle %198, %ep_p4_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p4_lo = vector.shuffle %ep_p4_swb, %199 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p4_vec = arith.select %ep_is_even, %ep_p4_hi, %ep_p4_lo : vector<8xbf16>
        vector.store %ep_p4_vec, %ep_buf[%ep_addr_1_0] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 5: accumulators %200, %201 → bn1, M-row-pair 1
        %ep_p5_swa = amdgpu.permlane_swap %200 16 : vector<4xbf16>
        %ep_p5_swb = amdgpu.permlane_swap %201 16 : vector<4xbf16>
        %ep_p5_hi = vector.shuffle %200, %ep_p5_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p5_lo = vector.shuffle %ep_p5_swb, %201 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p5_vec = arith.select %ep_is_even, %ep_p5_hi, %ep_p5_lo : vector<8xbf16>
        vector.store %ep_p5_vec, %ep_buf[%ep_addr_1_1] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 6: accumulators %202, %203 → bn1, M-row-pair 2
        %ep_p6_swa = amdgpu.permlane_swap %202 16 : vector<4xbf16>
        %ep_p6_swb = amdgpu.permlane_swap %203 16 : vector<4xbf16>
        %ep_p6_hi = vector.shuffle %202, %ep_p6_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p6_lo = vector.shuffle %ep_p6_swb, %203 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p6_vec = arith.select %ep_is_even, %ep_p6_hi, %ep_p6_lo : vector<8xbf16>
        vector.store %ep_p6_vec, %ep_buf[%ep_addr_1_2] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 7: accumulators %204, %205 → bn1, M-row-pair 3
        %ep_p7_swa = amdgpu.permlane_swap %204 16 : vector<4xbf16>
        %ep_p7_swb = amdgpu.permlane_swap %205 16 : vector<4xbf16>
        %ep_p7_hi = vector.shuffle %204, %ep_p7_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p7_lo = vector.shuffle %ep_p7_swb, %205 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p7_vec = arith.select %ep_is_even, %ep_p7_hi, %ep_p7_lo : vector<8xbf16>
        vector.store %ep_p7_vec, %ep_buf[%ep_addr_1_3] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // === N-group 2 (bn2): bf16 pairs %206/%207 .. %212/%213 ===
        // Pair 8: accumulators %206, %207 → bn2, M-row-pair 0
        %ep_p8_swa = amdgpu.permlane_swap %206 16 : vector<4xbf16>
        %ep_p8_swb = amdgpu.permlane_swap %207 16 : vector<4xbf16>
        %ep_p8_hi = vector.shuffle %206, %ep_p8_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p8_lo = vector.shuffle %ep_p8_swb, %207 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p8_vec = arith.select %ep_is_even, %ep_p8_hi, %ep_p8_lo : vector<8xbf16>
        vector.store %ep_p8_vec, %ep_buf[%ep_addr_2_0] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 9: accumulators %208, %209 → bn2, M-row-pair 1
        %ep_p9_swa = amdgpu.permlane_swap %208 16 : vector<4xbf16>
        %ep_p9_swb = amdgpu.permlane_swap %209 16 : vector<4xbf16>
        %ep_p9_hi = vector.shuffle %208, %ep_p9_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p9_lo = vector.shuffle %ep_p9_swb, %209 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p9_vec = arith.select %ep_is_even, %ep_p9_hi, %ep_p9_lo : vector<8xbf16>
        vector.store %ep_p9_vec, %ep_buf[%ep_addr_2_1] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 10: accumulators %210, %211 → bn2, M-row-pair 2
        %ep_p10_swa = amdgpu.permlane_swap %210 16 : vector<4xbf16>
        %ep_p10_swb = amdgpu.permlane_swap %211 16 : vector<4xbf16>
        %ep_p10_hi = vector.shuffle %210, %ep_p10_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p10_lo = vector.shuffle %ep_p10_swb, %211 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p10_vec = arith.select %ep_is_even, %ep_p10_hi, %ep_p10_lo : vector<8xbf16>
        vector.store %ep_p10_vec, %ep_buf[%ep_addr_2_2] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 11: accumulators %212, %213 → bn2, M-row-pair 3
        %ep_p11_swa = amdgpu.permlane_swap %212 16 : vector<4xbf16>
        %ep_p11_swb = amdgpu.permlane_swap %213 16 : vector<4xbf16>
        %ep_p11_hi = vector.shuffle %212, %ep_p11_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p11_lo = vector.shuffle %ep_p11_swb, %213 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p11_vec = arith.select %ep_is_even, %ep_p11_hi, %ep_p11_lo : vector<8xbf16>
        vector.store %ep_p11_vec, %ep_buf[%ep_addr_2_3] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // === N-group 3 (bn3): bf16 pairs %214/%215 .. %220/%221 [256x256 extension] ===
        // Pair 12: accumulators %214, %215 → bn3, M-row-pair 0
        %ep_p12_swa = amdgpu.permlane_swap %214 16 : vector<4xbf16>
        %ep_p12_swb = amdgpu.permlane_swap %215 16 : vector<4xbf16>
        %ep_p12_hi = vector.shuffle %214, %ep_p12_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p12_lo = vector.shuffle %ep_p12_swb, %215 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p12_vec = arith.select %ep_is_even, %ep_p12_hi, %ep_p12_lo : vector<8xbf16>
        vector.store %ep_p12_vec, %ep_buf[%ep_addr_3_0] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 13: accumulators %216, %217 → bn3, M-row-pair 1
        %ep_p13_swa = amdgpu.permlane_swap %216 16 : vector<4xbf16>
        %ep_p13_swb = amdgpu.permlane_swap %217 16 : vector<4xbf16>
        %ep_p13_hi = vector.shuffle %216, %ep_p13_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p13_lo = vector.shuffle %ep_p13_swb, %217 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p13_vec = arith.select %ep_is_even, %ep_p13_hi, %ep_p13_lo : vector<8xbf16>
        vector.store %ep_p13_vec, %ep_buf[%ep_addr_3_1] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 14: accumulators %218, %219 → bn3, M-row-pair 2
        %ep_p14_swa = amdgpu.permlane_swap %218 16 : vector<4xbf16>
        %ep_p14_swb = amdgpu.permlane_swap %219 16 : vector<4xbf16>
        %ep_p14_hi = vector.shuffle %218, %ep_p14_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p14_lo = vector.shuffle %ep_p14_swb, %219 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p14_vec = arith.select %ep_is_even, %ep_p14_hi, %ep_p14_lo : vector<8xbf16>
        vector.store %ep_p14_vec, %ep_buf[%ep_addr_3_2] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 15: accumulators %220, %221 → bn3, M-row-pair 3
        %ep_p15_swa = amdgpu.permlane_swap %220 16 : vector<4xbf16>
        %ep_p15_swb = amdgpu.permlane_swap %221 16 : vector<4xbf16>
        %ep_p15_hi = vector.shuffle %220, %ep_p15_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p15_lo = vector.shuffle %ep_p15_swb, %221 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p15_vec = arith.select %ep_is_even, %ep_p15_hi, %ep_p15_lo : vector<8xbf16>
        vector.store %ep_p15_vec, %ep_buf[%ep_addr_3_3] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: !hal.fence, %arg11: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg10) => %arg0 : !hal.buffer_view -> tensor<1024x4096xi8>
    %1 = hal.tensor.import wait(%arg10) => %arg1 : !hal.buffer_view -> tensor<1024x256xi8>
    %2 = hal.tensor.import wait(%arg10) => %arg2 : !hal.buffer_view -> tensor<1024x4096xi8>
    %3 = hal.tensor.import wait(%arg10) => %arg3 : !hal.buffer_view -> tensor<1024x256xi8>
    %4 = hal.tensor.import wait(%arg10) => %arg4 : !hal.buffer_view -> tensor<1024x1024xbf16>
    %5 = flow.dispatch @gemm::@gemm[%arg5, %arg6, %arg7, %arg8, %arg9](%0, %1, %2, %3, %4, %arg5, %arg6, %arg7, %arg8, %arg9) : (tensor<1024x4096xi8>, tensor<1024x256xi8>, tensor<1024x4096xi8>, tensor<1024x256xi8>, tensor<1024x1024xbf16>, index, index, index, index, index) -> %4
    %6 = hal.tensor.barrier join(%5 : tensor<1024x1024xbf16>) => %arg11 : !hal.fence
    %7 = hal.tensor.export %6 : tensor<1024x1024xbf16> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
