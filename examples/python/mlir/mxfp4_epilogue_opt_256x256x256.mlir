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
#map41 = affine_map<()[s0, s1, s2, s3] -> (s0 * 1048576 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8) floordiv 256) * 1048576 + 3968)>
#map42 = affine_map<()[s0, s1, s2, s3] -> (s0 * 1048576 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 256) * 1048576 + 266112)>
#map43 = affine_map<()[s0, s1, s2, s3] -> (s0 * 1048576 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 128) floordiv 256) * 1048576 + 528256)>
#map44 = affine_map<()[s0, s1, s2, s3] -> (s0 * 1048576 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 192) floordiv 256) * 1048576 + 790400)>
#map45 = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 4 + (s1 floordiv 64) * 16384 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 7936)>
#map46 = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 4 + (s1 floordiv 64) * 16384 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 16128)>
#map47 = affine_map<()[s0, s1, s2] -> (s0 * 65536 + s1 * 32768 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 7936)>
#map48 = affine_map<()[s0, s1, s2] -> (s0 * 65536 + s1 * 32768 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 16128)>
#map49 = affine_map<()[s0, s1, s2] -> (s0 * 65536 + s1 * 32768 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 24320)>
#map50 = affine_map<()[s0, s1, s2] -> (s0 * 65536 + s1 * 32768 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 32512)>
#map51 = affine_map<()[s0] -> (s0 * 256)>
#map52 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
#map53 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
#map54 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map55 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map56 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map57 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
#map58 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
#map59 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
#map60 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
#map61 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
#map62 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
#map63 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
#map64 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map65 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map66 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map67 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#map68 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
#map69 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
#map70 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
#map71 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
#map72 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
#map73 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
#map74 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
#map75 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
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
        %c256_i14 = arith.constant 256 : i14
        %c4096_i14 = arith.constant 4096 : i14
        %c2147483645_i64 = arith.constant 2147483645 : i64
        %c30 = arith.constant 30 : index
        %c262144_i64 = arith.constant 262144 : i64
        %c2 = arith.constant 2 : index
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
        %cast_7 = memref.cast %reinterpret_cast_6 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %27 = amdgpu.fat_raw_buffer_cast %cast_7 validBytes(%c262144_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %28 = affine.apply #map10()[%block_id_x, %thread_id_x]
        %29 = vector.load %27[%28] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %30 = affine.apply #map11()[%block_id_x, %thread_id_x]
        %31 = vector.load %27[%30] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %reinterpret_cast_8 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_9 = memref.cast %reinterpret_cast_8 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %32 = amdgpu.fat_raw_buffer_cast %cast_9 validBytes(%c262144_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %33 = affine.apply #map12()[%block_id_y, %thread_id_y, %thread_id_x]
        %34 = vector.load %32[%33] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %35 = affine.apply #map13()[%block_id_y, %thread_id_y, %thread_id_x]
        %36 = vector.load %32[%35] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %37 = affine.apply #map14()[%block_id_y, %thread_id_y, %thread_id_x]
        %38 = vector.load %32[%37] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %39 = affine.apply #map15()[%block_id_y, %thread_id_y, %thread_id_x]
        %40 = vector.load %32[%39] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        amdgpu.memory_counter_wait load(6)
        rocdl.s.barrier
        %41 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %42 = arith.index_cast %41 : index to i32
        %43 = arith.cmpi sge, %42, %c4_i32 : i32
        %44 = arith.cmpi slt, %42, %c4_i32 : i32
        scf.if %43 {
          rocdl.s.barrier
        }
        %reinterpret_cast_10 = memref.reinterpret_cast %alloc_2 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
        %45 = affine.apply #map17()[%thread_id_x]
        %46 = arith.xori %45, %9 : index
        %47 = affine.apply #map18()[%thread_id_x, %46]
        %48 = affine.apply #map19()[%thread_id_x, %46]
        %49 = affine.apply #map20()[%thread_id_x, %46]
        %50 = affine.apply #map21()[%thread_id_x, %46]
        %reinterpret_cast_11 = memref.reinterpret_cast %alloc_0 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
        %51 = affine.apply #map22()[%thread_id_y, %thread_id_x, %46]
        %52 = affine.apply #map23()[%thread_id_y, %thread_id_x, %46]
        %53 = affine.apply #map24()[%thread_id_y, %thread_id_x, %46]
        %54 = affine.apply #map25()[%thread_id_y, %thread_id_x, %46]
        %55 = affine.apply #map26()[%thread_id_y, %thread_id_x, %46]
        %56 = affine.apply #map27()[%thread_id_y, %thread_id_x, %46]
        %57 = affine.apply #map28()[%thread_id_y, %thread_id_x, %46]
        %58 = affine.apply #map29()[%thread_id_y, %thread_id_x, %46]
        %59 = affine.apply #map30()[%thread_id_x]
        %60 = arith.xori %59, %9 : index
        %61 = affine.apply #map18()[%thread_id_x, %60]
        %62 = affine.apply #map19()[%thread_id_x, %60]
        %63 = affine.apply #map20()[%thread_id_x, %60]
        %64 = affine.apply #map21()[%thread_id_x, %60]
        %65 = affine.apply #map22()[%thread_id_y, %thread_id_x, %60]
        %66 = affine.apply #map23()[%thread_id_y, %thread_id_x, %60]
        %67 = affine.apply #map24()[%thread_id_y, %thread_id_x, %60]
        %68 = affine.apply #map25()[%thread_id_y, %thread_id_x, %60]
        %69 = affine.apply #map26()[%thread_id_y, %thread_id_x, %60]
        %70 = affine.apply #map27()[%thread_id_y, %thread_id_x, %60]
        %71 = affine.apply #map28()[%thread_id_y, %thread_id_x, %60]
        %72 = affine.apply #map29()[%thread_id_y, %thread_id_x, %60]
        %reinterpret_cast_12 = memref.reinterpret_cast %alloc_1 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_13 = memref.reinterpret_cast %alloc to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
        %73:38 = scf.for %arg10 = %c0 to %c30 step %c2 iter_args(%arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %cst, %arg39 = %cst, %arg40 = %cst, %arg41 = %cst, %arg42 = %cst, %arg43 = %29, %arg44 = %31, %arg45 = %34, %arg46 = %36, %arg47 = %38, %arg48 = %40) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>) {
          %664 = vector.bitcast %arg44 : vector<4xi8> to vector<4xf8E8M0FNU>
          %665 = vector.bitcast %arg48 : vector<4xi8> to vector<4xf8E8M0FNU>
          %666 = vector.bitcast %arg47 : vector<4xi8> to vector<4xf8E8M0FNU>
          %667 = vector.bitcast %arg46 : vector<4xi8> to vector<4xf8E8M0FNU>
          %668 = vector.bitcast %arg45 : vector<4xi8> to vector<4xf8E8M0FNU>
          %669 = vector.bitcast %arg43 : vector<4xi8> to vector<4xf8E8M0FNU>
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(0)
          rocdl.s.barrier
          %670 = affine.apply #map31()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%670], %alloc_1[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %671 = affine.apply #map32()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%671], %alloc_1[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %672 = affine.apply #map33()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%672], %alloc_1[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %673 = affine.apply #map34()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%673], %alloc_1[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %674 = affine.apply #map31()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %23[%674], %alloc[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %675 = affine.apply #map32()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %23[%675], %alloc[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %676 = affine.apply #map33()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %23[%676], %alloc[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %677 = affine.apply #map34()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %23[%677], %alloc[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          rocdl.sched.barrier 0
          %678 = vector.load %reinterpret_cast_10[%47] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %679 = vector.load %reinterpret_cast_10[%48] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %680 = vector.load %reinterpret_cast_10[%49] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %681 = vector.load %reinterpret_cast_10[%50] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %682 = vector.load %reinterpret_cast_11[%51] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %683 = vector.load %reinterpret_cast_11[%52] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %684 = vector.load %reinterpret_cast_11[%53] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %685 = vector.load %reinterpret_cast_11[%54] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %686 = vector.load %reinterpret_cast_11[%55] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %687 = vector.load %reinterpret_cast_11[%56] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %688 = vector.load %reinterpret_cast_11[%57] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %689 = vector.load %reinterpret_cast_11[%58] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %690 = vector.bitcast %678 : vector<16xi8> to vector<32xf4E2M1FN>
          %691 = vector.bitcast %679 : vector<16xi8> to vector<32xf4E2M1FN>
          %692 = vector.bitcast %680 : vector<16xi8> to vector<32xf4E2M1FN>
          %693 = vector.bitcast %681 : vector<16xi8> to vector<32xf4E2M1FN>
          %694 = vector.bitcast %682 : vector<16xi8> to vector<32xf4E2M1FN>
          %695 = vector.bitcast %683 : vector<16xi8> to vector<32xf4E2M1FN>
          %696 = vector.bitcast %684 : vector<16xi8> to vector<32xf4E2M1FN>
          %697 = vector.bitcast %685 : vector<16xi8> to vector<32xf4E2M1FN>
          %698 = vector.bitcast %686 : vector<16xi8> to vector<32xf4E2M1FN>
          %699 = vector.bitcast %687 : vector<16xi8> to vector<32xf4E2M1FN>
          %700 = vector.bitcast %688 : vector<16xi8> to vector<32xf4E2M1FN>
          %701 = vector.bitcast %689 : vector<16xi8> to vector<32xf4E2M1FN>
          %702 = affine.apply #map35()[%block_id_x, %thread_id_x, %arg10]
          %703 = vector.load %27[%702] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %704 = vector.bitcast %703 : vector<4xi8> to vector<4xf8E8M0FNU>
          %705 = affine.apply #map36()[%block_id_x, %thread_id_x, %arg10]
          %706 = vector.load %27[%705] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %707 = vector.bitcast %706 : vector<4xi8> to vector<4xf8E8M0FNU>
          %708 = affine.apply #map37()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %709 = vector.load %32[%708] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %710 = vector.bitcast %709 : vector<4xi8> to vector<4xf8E8M0FNU>
          %711 = affine.apply #map38()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %712 = vector.load %32[%711] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %713 = vector.bitcast %712 : vector<4xi8> to vector<4xf8E8M0FNU>
          %714 = affine.apply #map39()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %715 = vector.load %32[%714] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %716 = vector.bitcast %715 : vector<4xi8> to vector<4xf8E8M0FNU>
          %717 = affine.apply #map40()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %718 = vector.load %32[%717] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %719 = vector.bitcast %718 : vector<4xi8> to vector<4xf8E8M0FNU>
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %720 = amdgpu.scaled_mfma 16x16x128 (%669[0] * %690) * (%668[0] * %694) + %arg11 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %721 = amdgpu.scaled_mfma 16x16x128 (%669[0] * %690) * (%668[1] * %695) + %arg12 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %722 = amdgpu.scaled_mfma 16x16x128 (%669[0] * %690) * (%667[0] * %696) + %arg13 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %723 = amdgpu.scaled_mfma 16x16x128 (%669[0] * %690) * (%667[1] * %697) + %arg14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %724 = amdgpu.scaled_mfma 16x16x128 (%669[0] * %690) * (%666[0] * %698) + %arg15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %725 = amdgpu.scaled_mfma 16x16x128 (%669[0] * %690) * (%666[1] * %699) + %arg16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %726 = amdgpu.scaled_mfma 16x16x128 (%669[0] * %690) * (%665[0] * %700) + %arg17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %727 = amdgpu.scaled_mfma 16x16x128 (%669[0] * %690) * (%665[1] * %701) + %arg18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %728 = amdgpu.scaled_mfma 16x16x128 (%669[1] * %691) * (%668[0] * %694) + %arg19 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %729 = amdgpu.scaled_mfma 16x16x128 (%669[1] * %691) * (%668[1] * %695) + %arg20 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %730 = amdgpu.scaled_mfma 16x16x128 (%669[1] * %691) * (%667[0] * %696) + %arg21 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %731 = amdgpu.scaled_mfma 16x16x128 (%669[1] * %691) * (%667[1] * %697) + %arg22 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %732 = amdgpu.scaled_mfma 16x16x128 (%669[1] * %691) * (%666[0] * %698) + %arg23 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %733 = amdgpu.scaled_mfma 16x16x128 (%669[1] * %691) * (%666[1] * %699) + %arg24 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %734 = amdgpu.scaled_mfma 16x16x128 (%669[1] * %691) * (%665[0] * %700) + %arg25 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %735 = amdgpu.scaled_mfma 16x16x128 (%669[1] * %691) * (%665[1] * %701) + %arg26 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %736 = amdgpu.scaled_mfma 16x16x128 (%664[0] * %692) * (%668[0] * %694) + %arg27 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %737 = amdgpu.scaled_mfma 16x16x128 (%664[0] * %692) * (%668[1] * %695) + %arg28 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %738 = amdgpu.scaled_mfma 16x16x128 (%664[0] * %692) * (%667[0] * %696) + %arg29 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %739 = amdgpu.scaled_mfma 16x16x128 (%664[0] * %692) * (%667[1] * %697) + %arg30 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %740 = amdgpu.scaled_mfma 16x16x128 (%664[0] * %692) * (%666[0] * %698) + %arg31 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %741 = amdgpu.scaled_mfma 16x16x128 (%664[0] * %692) * (%666[1] * %699) + %arg32 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %742 = amdgpu.scaled_mfma 16x16x128 (%664[0] * %692) * (%665[0] * %700) + %arg33 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %743 = amdgpu.scaled_mfma 16x16x128 (%664[0] * %692) * (%665[1] * %701) + %arg34 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %744 = amdgpu.scaled_mfma 16x16x128 (%664[1] * %693) * (%668[0] * %694) + %arg35 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %745 = amdgpu.scaled_mfma 16x16x128 (%664[1] * %693) * (%668[1] * %695) + %arg36 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %746 = amdgpu.scaled_mfma 16x16x128 (%664[1] * %693) * (%667[0] * %696) + %arg37 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %747 = amdgpu.scaled_mfma 16x16x128 (%664[1] * %693) * (%667[1] * %697) + %arg38 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %748 = amdgpu.scaled_mfma 16x16x128 (%664[1] * %693) * (%666[0] * %698) + %arg39 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %749 = amdgpu.scaled_mfma 16x16x128 (%664[1] * %693) * (%666[1] * %699) + %arg40 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %750 = amdgpu.scaled_mfma 16x16x128 (%664[1] * %693) * (%665[0] * %700) + %arg41 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %751 = amdgpu.scaled_mfma 16x16x128 (%664[1] * %693) * (%665[1] * %701) + %arg42 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.sched.barrier 0
          %752 = vector.load %reinterpret_cast_10[%61] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %753 = vector.load %reinterpret_cast_10[%62] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %754 = vector.load %reinterpret_cast_10[%63] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %755 = vector.load %reinterpret_cast_10[%64] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %756 = vector.load %reinterpret_cast_11[%65] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %757 = vector.load %reinterpret_cast_11[%66] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %758 = vector.load %reinterpret_cast_11[%67] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %759 = vector.load %reinterpret_cast_11[%68] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %760 = vector.load %reinterpret_cast_11[%69] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %761 = vector.load %reinterpret_cast_11[%70] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %762 = vector.load %reinterpret_cast_11[%71] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %763 = vector.load %reinterpret_cast_11[%72] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %764 = vector.bitcast %752 : vector<16xi8> to vector<32xf4E2M1FN>
          %765 = vector.bitcast %753 : vector<16xi8> to vector<32xf4E2M1FN>
          %766 = vector.bitcast %754 : vector<16xi8> to vector<32xf4E2M1FN>
          %767 = vector.bitcast %755 : vector<16xi8> to vector<32xf4E2M1FN>
          %768 = vector.bitcast %756 : vector<16xi8> to vector<32xf4E2M1FN>
          %769 = vector.bitcast %757 : vector<16xi8> to vector<32xf4E2M1FN>
          %770 = vector.bitcast %758 : vector<16xi8> to vector<32xf4E2M1FN>
          %771 = vector.bitcast %759 : vector<16xi8> to vector<32xf4E2M1FN>
          %772 = vector.bitcast %760 : vector<16xi8> to vector<32xf4E2M1FN>
          %773 = vector.bitcast %761 : vector<16xi8> to vector<32xf4E2M1FN>
          %774 = vector.bitcast %762 : vector<16xi8> to vector<32xf4E2M1FN>
          %775 = vector.bitcast %763 : vector<16xi8> to vector<32xf4E2M1FN>
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(6)
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %776 = amdgpu.scaled_mfma 16x16x128 (%669[2] * %764) * (%668[2] * %768) + %720 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %777 = amdgpu.scaled_mfma 16x16x128 (%669[2] * %764) * (%668[3] * %769) + %721 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %778 = amdgpu.scaled_mfma 16x16x128 (%669[2] * %764) * (%667[2] * %770) + %722 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %779 = amdgpu.scaled_mfma 16x16x128 (%669[2] * %764) * (%667[3] * %771) + %723 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %780 = amdgpu.scaled_mfma 16x16x128 (%669[2] * %764) * (%666[2] * %772) + %724 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %781 = amdgpu.scaled_mfma 16x16x128 (%669[2] * %764) * (%666[3] * %773) + %725 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %782 = amdgpu.scaled_mfma 16x16x128 (%669[2] * %764) * (%665[2] * %774) + %726 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %783 = amdgpu.scaled_mfma 16x16x128 (%669[2] * %764) * (%665[3] * %775) + %727 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %784 = amdgpu.scaled_mfma 16x16x128 (%669[3] * %765) * (%668[2] * %768) + %728 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %785 = amdgpu.scaled_mfma 16x16x128 (%669[3] * %765) * (%668[3] * %769) + %729 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %786 = amdgpu.scaled_mfma 16x16x128 (%669[3] * %765) * (%667[2] * %770) + %730 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %787 = amdgpu.scaled_mfma 16x16x128 (%669[3] * %765) * (%667[3] * %771) + %731 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %788 = amdgpu.scaled_mfma 16x16x128 (%669[3] * %765) * (%666[2] * %772) + %732 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %789 = amdgpu.scaled_mfma 16x16x128 (%669[3] * %765) * (%666[3] * %773) + %733 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %790 = amdgpu.scaled_mfma 16x16x128 (%669[3] * %765) * (%665[2] * %774) + %734 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %791 = amdgpu.scaled_mfma 16x16x128 (%669[3] * %765) * (%665[3] * %775) + %735 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %792 = amdgpu.scaled_mfma 16x16x128 (%664[2] * %766) * (%668[2] * %768) + %736 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %793 = amdgpu.scaled_mfma 16x16x128 (%664[2] * %766) * (%668[3] * %769) + %737 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %794 = amdgpu.scaled_mfma 16x16x128 (%664[2] * %766) * (%667[2] * %770) + %738 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %795 = amdgpu.scaled_mfma 16x16x128 (%664[2] * %766) * (%667[3] * %771) + %739 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %796 = amdgpu.scaled_mfma 16x16x128 (%664[2] * %766) * (%666[2] * %772) + %740 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %797 = amdgpu.scaled_mfma 16x16x128 (%664[2] * %766) * (%666[3] * %773) + %741 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %798 = amdgpu.scaled_mfma 16x16x128 (%664[2] * %766) * (%665[2] * %774) + %742 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %799 = amdgpu.scaled_mfma 16x16x128 (%664[2] * %766) * (%665[3] * %775) + %743 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %800 = amdgpu.scaled_mfma 16x16x128 (%664[3] * %767) * (%668[2] * %768) + %744 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %801 = amdgpu.scaled_mfma 16x16x128 (%664[3] * %767) * (%668[3] * %769) + %745 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %802 = amdgpu.scaled_mfma 16x16x128 (%664[3] * %767) * (%667[2] * %770) + %746 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %803 = amdgpu.scaled_mfma 16x16x128 (%664[3] * %767) * (%667[3] * %771) + %747 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %804 = amdgpu.scaled_mfma 16x16x128 (%664[3] * %767) * (%666[2] * %772) + %748 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %805 = amdgpu.scaled_mfma 16x16x128 (%664[3] * %767) * (%666[3] * %773) + %749 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %806 = amdgpu.scaled_mfma 16x16x128 (%664[3] * %767) * (%665[2] * %774) + %750 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %807 = amdgpu.scaled_mfma 16x16x128 (%664[3] * %767) * (%665[3] * %775) + %751 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          %808 = arith.addi %arg10, %c1 : index
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(0)
          rocdl.s.barrier
          %809 = affine.apply #map31()[%block_id_x, %thread_id_y, %thread_id_x, %808, %10]
          amdgpu.gather_to_lds %12[%809], %alloc_2[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %810 = affine.apply #map32()[%block_id_x, %thread_id_y, %thread_id_x, %808, %10]
          amdgpu.gather_to_lds %12[%810], %alloc_2[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %811 = affine.apply #map33()[%block_id_x, %thread_id_y, %thread_id_x, %808, %10]
          amdgpu.gather_to_lds %12[%811], %alloc_2[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %812 = affine.apply #map34()[%block_id_x, %thread_id_y, %thread_id_x, %808, %10]
          amdgpu.gather_to_lds %12[%812], %alloc_2[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %813 = affine.apply #map31()[%block_id_y, %thread_id_y, %thread_id_x, %808, %10]
          amdgpu.gather_to_lds %23[%813], %alloc_0[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %814 = affine.apply #map32()[%block_id_y, %thread_id_y, %thread_id_x, %808, %10]
          amdgpu.gather_to_lds %23[%814], %alloc_0[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %815 = affine.apply #map33()[%block_id_y, %thread_id_y, %thread_id_x, %808, %10]
          amdgpu.gather_to_lds %23[%815], %alloc_0[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          %816 = affine.apply #map34()[%block_id_y, %thread_id_y, %thread_id_x, %808, %10]
          amdgpu.gather_to_lds %23[%816], %alloc_0[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          rocdl.sched.barrier 0
          %817 = vector.load %reinterpret_cast_12[%47] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %818 = vector.load %reinterpret_cast_12[%48] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %819 = vector.load %reinterpret_cast_12[%49] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %820 = vector.load %reinterpret_cast_12[%50] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %821 = vector.load %reinterpret_cast_13[%51] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %822 = vector.load %reinterpret_cast_13[%52] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %823 = vector.load %reinterpret_cast_13[%53] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %824 = vector.load %reinterpret_cast_13[%54] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %825 = vector.load %reinterpret_cast_13[%55] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %826 = vector.load %reinterpret_cast_13[%56] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %827 = vector.load %reinterpret_cast_13[%57] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %828 = vector.load %reinterpret_cast_13[%58] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %829 = vector.bitcast %817 : vector<16xi8> to vector<32xf4E2M1FN>
          %830 = vector.bitcast %818 : vector<16xi8> to vector<32xf4E2M1FN>
          %831 = vector.bitcast %819 : vector<16xi8> to vector<32xf4E2M1FN>
          %832 = vector.bitcast %820 : vector<16xi8> to vector<32xf4E2M1FN>
          %833 = vector.bitcast %821 : vector<16xi8> to vector<32xf4E2M1FN>
          %834 = vector.bitcast %822 : vector<16xi8> to vector<32xf4E2M1FN>
          %835 = vector.bitcast %823 : vector<16xi8> to vector<32xf4E2M1FN>
          %836 = vector.bitcast %824 : vector<16xi8> to vector<32xf4E2M1FN>
          %837 = vector.bitcast %825 : vector<16xi8> to vector<32xf4E2M1FN>
          %838 = vector.bitcast %826 : vector<16xi8> to vector<32xf4E2M1FN>
          %839 = vector.bitcast %827 : vector<16xi8> to vector<32xf4E2M1FN>
          %840 = vector.bitcast %828 : vector<16xi8> to vector<32xf4E2M1FN>
          %841 = affine.apply #map35()[%block_id_x, %thread_id_x, %808]
          %842 = vector.load %27[%841] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %843 = affine.apply #map36()[%block_id_x, %thread_id_x, %808]
          %844 = vector.load %27[%843] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %845 = affine.apply #map37()[%block_id_y, %thread_id_y, %808, %thread_id_x]
          %846 = vector.load %32[%845] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %847 = affine.apply #map38()[%block_id_y, %thread_id_y, %808, %thread_id_x]
          %848 = vector.load %32[%847] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %849 = affine.apply #map39()[%block_id_y, %thread_id_y, %808, %thread_id_x]
          %850 = vector.load %32[%849] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %851 = affine.apply #map40()[%block_id_y, %thread_id_y, %808, %thread_id_x]
          %852 = vector.load %32[%851] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %853 = amdgpu.scaled_mfma 16x16x128 (%704[0] * %829) * (%710[0] * %833) + %776 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %854 = amdgpu.scaled_mfma 16x16x128 (%704[0] * %829) * (%710[1] * %834) + %777 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %855 = amdgpu.scaled_mfma 16x16x128 (%704[0] * %829) * (%713[0] * %835) + %778 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %856 = amdgpu.scaled_mfma 16x16x128 (%704[0] * %829) * (%713[1] * %836) + %779 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %857 = amdgpu.scaled_mfma 16x16x128 (%704[0] * %829) * (%716[0] * %837) + %780 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %858 = amdgpu.scaled_mfma 16x16x128 (%704[0] * %829) * (%716[1] * %838) + %781 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %859 = amdgpu.scaled_mfma 16x16x128 (%704[0] * %829) * (%719[0] * %839) + %782 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %860 = amdgpu.scaled_mfma 16x16x128 (%704[0] * %829) * (%719[1] * %840) + %783 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %861 = amdgpu.scaled_mfma 16x16x128 (%704[1] * %830) * (%710[0] * %833) + %784 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %862 = amdgpu.scaled_mfma 16x16x128 (%704[1] * %830) * (%710[1] * %834) + %785 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %863 = amdgpu.scaled_mfma 16x16x128 (%704[1] * %830) * (%713[0] * %835) + %786 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %864 = amdgpu.scaled_mfma 16x16x128 (%704[1] * %830) * (%713[1] * %836) + %787 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %865 = amdgpu.scaled_mfma 16x16x128 (%704[1] * %830) * (%716[0] * %837) + %788 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %866 = amdgpu.scaled_mfma 16x16x128 (%704[1] * %830) * (%716[1] * %838) + %789 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %867 = amdgpu.scaled_mfma 16x16x128 (%704[1] * %830) * (%719[0] * %839) + %790 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %868 = amdgpu.scaled_mfma 16x16x128 (%704[1] * %830) * (%719[1] * %840) + %791 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %869 = amdgpu.scaled_mfma 16x16x128 (%707[0] * %831) * (%710[0] * %833) + %792 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %870 = amdgpu.scaled_mfma 16x16x128 (%707[0] * %831) * (%710[1] * %834) + %793 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %871 = amdgpu.scaled_mfma 16x16x128 (%707[0] * %831) * (%713[0] * %835) + %794 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %872 = amdgpu.scaled_mfma 16x16x128 (%707[0] * %831) * (%713[1] * %836) + %795 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %873 = amdgpu.scaled_mfma 16x16x128 (%707[0] * %831) * (%716[0] * %837) + %796 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %874 = amdgpu.scaled_mfma 16x16x128 (%707[0] * %831) * (%716[1] * %838) + %797 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %875 = amdgpu.scaled_mfma 16x16x128 (%707[0] * %831) * (%719[0] * %839) + %798 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %876 = amdgpu.scaled_mfma 16x16x128 (%707[0] * %831) * (%719[1] * %840) + %799 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %877 = amdgpu.scaled_mfma 16x16x128 (%707[1] * %832) * (%710[0] * %833) + %800 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %878 = amdgpu.scaled_mfma 16x16x128 (%707[1] * %832) * (%710[1] * %834) + %801 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %879 = amdgpu.scaled_mfma 16x16x128 (%707[1] * %832) * (%713[0] * %835) + %802 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %880 = amdgpu.scaled_mfma 16x16x128 (%707[1] * %832) * (%713[1] * %836) + %803 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %881 = amdgpu.scaled_mfma 16x16x128 (%707[1] * %832) * (%716[0] * %837) + %804 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %882 = amdgpu.scaled_mfma 16x16x128 (%707[1] * %832) * (%716[1] * %838) + %805 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %883 = amdgpu.scaled_mfma 16x16x128 (%707[1] * %832) * (%719[0] * %839) + %806 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %884 = amdgpu.scaled_mfma 16x16x128 (%707[1] * %832) * (%719[1] * %840) + %807 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.sched.barrier 0
          %885 = vector.load %reinterpret_cast_12[%61] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %886 = vector.load %reinterpret_cast_12[%62] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %887 = vector.load %reinterpret_cast_12[%63] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %888 = vector.load %reinterpret_cast_12[%64] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %889 = vector.load %reinterpret_cast_13[%65] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %890 = vector.load %reinterpret_cast_13[%66] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %891 = vector.load %reinterpret_cast_13[%67] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %892 = vector.load %reinterpret_cast_13[%68] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %893 = vector.load %reinterpret_cast_13[%69] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %894 = vector.load %reinterpret_cast_13[%70] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %895 = vector.load %reinterpret_cast_13[%71] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %896 = vector.load %reinterpret_cast_13[%72] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %897 = vector.bitcast %885 : vector<16xi8> to vector<32xf4E2M1FN>
          %898 = vector.bitcast %886 : vector<16xi8> to vector<32xf4E2M1FN>
          %899 = vector.bitcast %887 : vector<16xi8> to vector<32xf4E2M1FN>
          %900 = vector.bitcast %888 : vector<16xi8> to vector<32xf4E2M1FN>
          %901 = vector.bitcast %889 : vector<16xi8> to vector<32xf4E2M1FN>
          %902 = vector.bitcast %890 : vector<16xi8> to vector<32xf4E2M1FN>
          %903 = vector.bitcast %891 : vector<16xi8> to vector<32xf4E2M1FN>
          %904 = vector.bitcast %892 : vector<16xi8> to vector<32xf4E2M1FN>
          %905 = vector.bitcast %893 : vector<16xi8> to vector<32xf4E2M1FN>
          %906 = vector.bitcast %894 : vector<16xi8> to vector<32xf4E2M1FN>
          %907 = vector.bitcast %895 : vector<16xi8> to vector<32xf4E2M1FN>
          %908 = vector.bitcast %896 : vector<16xi8> to vector<32xf4E2M1FN>
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(6)
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %909 = amdgpu.scaled_mfma 16x16x128 (%704[2] * %897) * (%710[2] * %901) + %853 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %910 = amdgpu.scaled_mfma 16x16x128 (%704[2] * %897) * (%710[3] * %902) + %854 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %911 = amdgpu.scaled_mfma 16x16x128 (%704[2] * %897) * (%713[2] * %903) + %855 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %912 = amdgpu.scaled_mfma 16x16x128 (%704[2] * %897) * (%713[3] * %904) + %856 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %913 = amdgpu.scaled_mfma 16x16x128 (%704[2] * %897) * (%716[2] * %905) + %857 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %914 = amdgpu.scaled_mfma 16x16x128 (%704[2] * %897) * (%716[3] * %906) + %858 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %915 = amdgpu.scaled_mfma 16x16x128 (%704[2] * %897) * (%719[2] * %907) + %859 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %916 = amdgpu.scaled_mfma 16x16x128 (%704[2] * %897) * (%719[3] * %908) + %860 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %917 = amdgpu.scaled_mfma 16x16x128 (%704[3] * %898) * (%710[2] * %901) + %861 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %918 = amdgpu.scaled_mfma 16x16x128 (%704[3] * %898) * (%710[3] * %902) + %862 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %919 = amdgpu.scaled_mfma 16x16x128 (%704[3] * %898) * (%713[2] * %903) + %863 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %920 = amdgpu.scaled_mfma 16x16x128 (%704[3] * %898) * (%713[3] * %904) + %864 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %921 = amdgpu.scaled_mfma 16x16x128 (%704[3] * %898) * (%716[2] * %905) + %865 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %922 = amdgpu.scaled_mfma 16x16x128 (%704[3] * %898) * (%716[3] * %906) + %866 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %923 = amdgpu.scaled_mfma 16x16x128 (%704[3] * %898) * (%719[2] * %907) + %867 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %924 = amdgpu.scaled_mfma 16x16x128 (%704[3] * %898) * (%719[3] * %908) + %868 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %925 = amdgpu.scaled_mfma 16x16x128 (%707[2] * %899) * (%710[2] * %901) + %869 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %926 = amdgpu.scaled_mfma 16x16x128 (%707[2] * %899) * (%710[3] * %902) + %870 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %927 = amdgpu.scaled_mfma 16x16x128 (%707[2] * %899) * (%713[2] * %903) + %871 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %928 = amdgpu.scaled_mfma 16x16x128 (%707[2] * %899) * (%713[3] * %904) + %872 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %929 = amdgpu.scaled_mfma 16x16x128 (%707[2] * %899) * (%716[2] * %905) + %873 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %930 = amdgpu.scaled_mfma 16x16x128 (%707[2] * %899) * (%716[3] * %906) + %874 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %931 = amdgpu.scaled_mfma 16x16x128 (%707[2] * %899) * (%719[2] * %907) + %875 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %932 = amdgpu.scaled_mfma 16x16x128 (%707[2] * %899) * (%719[3] * %908) + %876 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %933 = amdgpu.scaled_mfma 16x16x128 (%707[3] * %900) * (%710[2] * %901) + %877 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %934 = amdgpu.scaled_mfma 16x16x128 (%707[3] * %900) * (%710[3] * %902) + %878 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %935 = amdgpu.scaled_mfma 16x16x128 (%707[3] * %900) * (%713[2] * %903) + %879 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %936 = amdgpu.scaled_mfma 16x16x128 (%707[3] * %900) * (%713[3] * %904) + %880 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %937 = amdgpu.scaled_mfma 16x16x128 (%707[3] * %900) * (%716[2] * %905) + %881 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %938 = amdgpu.scaled_mfma 16x16x128 (%707[3] * %900) * (%716[3] * %906) + %882 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %939 = amdgpu.scaled_mfma 16x16x128 (%707[3] * %900) * (%719[2] * %907) + %883 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %940 = amdgpu.scaled_mfma 16x16x128 (%707[3] * %900) * (%719[3] * %908) + %884 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          scf.yield %909, %910, %911, %912, %913, %914, %915, %916, %917, %918, %919, %920, %921, %922, %923, %924, %925, %926, %927, %928, %929, %930, %931, %932, %933, %934, %935, %936, %937, %938, %939, %940, %842, %844, %846, %848, %850, %852 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>
        }
        %74 = vector.bitcast %73#33 : vector<4xi8> to vector<4xf8E8M0FNU>
        %75 = vector.bitcast %73#37 : vector<4xi8> to vector<4xf8E8M0FNU>
        %76 = vector.bitcast %73#36 : vector<4xi8> to vector<4xf8E8M0FNU>
        %77 = vector.bitcast %73#35 : vector<4xi8> to vector<4xf8E8M0FNU>
        %78 = vector.bitcast %73#34 : vector<4xi8> to vector<4xf8E8M0FNU>
        %79 = vector.bitcast %73#32 : vector<4xi8> to vector<4xf8E8M0FNU>
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(0)
        rocdl.s.barrier
        %80 = affine.apply #map41()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%80], %alloc_1[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %81 = affine.apply #map42()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%81], %alloc_1[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %82 = affine.apply #map43()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%82], %alloc_1[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %83 = affine.apply #map44()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%83], %alloc_1[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %84 = affine.apply #map41()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%84], %alloc[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %85 = affine.apply #map42()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%85], %alloc[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %86 = affine.apply #map43()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%86], %alloc[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %87 = affine.apply #map44()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%87], %alloc[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        rocdl.sched.barrier 0
        %88 = vector.load %reinterpret_cast_10[%47] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %89 = vector.load %reinterpret_cast_10[%48] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %90 = vector.load %reinterpret_cast_10[%49] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %91 = vector.load %reinterpret_cast_10[%50] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %92 = vector.load %reinterpret_cast_11[%51] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %93 = vector.load %reinterpret_cast_11[%52] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %94 = vector.load %reinterpret_cast_11[%53] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %95 = vector.load %reinterpret_cast_11[%54] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %96 = vector.load %reinterpret_cast_11[%55] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %97 = vector.load %reinterpret_cast_11[%56] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %98 = vector.load %reinterpret_cast_11[%57] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %99 = vector.load %reinterpret_cast_11[%58] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %100 = vector.bitcast %88 : vector<16xi8> to vector<32xf4E2M1FN>
        %101 = vector.bitcast %89 : vector<16xi8> to vector<32xf4E2M1FN>
        %102 = vector.bitcast %90 : vector<16xi8> to vector<32xf4E2M1FN>
        %103 = vector.bitcast %91 : vector<16xi8> to vector<32xf4E2M1FN>
        %104 = vector.bitcast %92 : vector<16xi8> to vector<32xf4E2M1FN>
        %105 = vector.bitcast %93 : vector<16xi8> to vector<32xf4E2M1FN>
        %106 = vector.bitcast %94 : vector<16xi8> to vector<32xf4E2M1FN>
        %107 = vector.bitcast %95 : vector<16xi8> to vector<32xf4E2M1FN>
        %108 = vector.bitcast %96 : vector<16xi8> to vector<32xf4E2M1FN>
        %109 = vector.bitcast %97 : vector<16xi8> to vector<32xf4E2M1FN>
        %110 = vector.bitcast %98 : vector<16xi8> to vector<32xf4E2M1FN>
        %111 = vector.bitcast %99 : vector<16xi8> to vector<32xf4E2M1FN>
        %112 = affine.apply #map45()[%block_id_x, %thread_id_x]
        %113 = vector.load %27[%112] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %114 = vector.bitcast %113 : vector<4xi8> to vector<4xf8E8M0FNU>
        %115 = affine.apply #map46()[%block_id_x, %thread_id_x]
        %116 = vector.load %27[%115] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %117 = vector.bitcast %116 : vector<4xi8> to vector<4xf8E8M0FNU>
        %118 = affine.apply #map47()[%block_id_y, %thread_id_y, %thread_id_x]
        %119 = vector.load %32[%118] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %120 = vector.bitcast %119 : vector<4xi8> to vector<4xf8E8M0FNU>
        %121 = affine.apply #map48()[%block_id_y, %thread_id_y, %thread_id_x]
        %122 = vector.load %32[%121] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %123 = vector.bitcast %122 : vector<4xi8> to vector<4xf8E8M0FNU>
        %124 = affine.apply #map49()[%block_id_y, %thread_id_y, %thread_id_x]
        %125 = vector.load %32[%124] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %126 = vector.bitcast %125 : vector<4xi8> to vector<4xf8E8M0FNU>
        %127 = affine.apply #map50()[%block_id_y, %thread_id_y, %thread_id_x]
        %128 = vector.load %32[%127] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %129 = vector.bitcast %128 : vector<4xi8> to vector<4xf8E8M0FNU>
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %130 = amdgpu.scaled_mfma 16x16x128 (%79[0] * %100) * (%78[0] * %104) + %73#0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %131 = amdgpu.scaled_mfma 16x16x128 (%79[0] * %100) * (%78[1] * %105) + %73#1 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %132 = amdgpu.scaled_mfma 16x16x128 (%79[0] * %100) * (%77[0] * %106) + %73#2 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %133 = amdgpu.scaled_mfma 16x16x128 (%79[0] * %100) * (%77[1] * %107) + %73#3 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %134 = amdgpu.scaled_mfma 16x16x128 (%79[0] * %100) * (%76[0] * %108) + %73#4 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %135 = amdgpu.scaled_mfma 16x16x128 (%79[0] * %100) * (%76[1] * %109) + %73#5 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %136 = amdgpu.scaled_mfma 16x16x128 (%79[0] * %100) * (%75[0] * %110) + %73#6 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %137 = amdgpu.scaled_mfma 16x16x128 (%79[0] * %100) * (%75[1] * %111) + %73#7 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %138 = amdgpu.scaled_mfma 16x16x128 (%79[1] * %101) * (%78[0] * %104) + %73#8 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %139 = amdgpu.scaled_mfma 16x16x128 (%79[1] * %101) * (%78[1] * %105) + %73#9 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %140 = amdgpu.scaled_mfma 16x16x128 (%79[1] * %101) * (%77[0] * %106) + %73#10 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %141 = amdgpu.scaled_mfma 16x16x128 (%79[1] * %101) * (%77[1] * %107) + %73#11 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %142 = amdgpu.scaled_mfma 16x16x128 (%79[1] * %101) * (%76[0] * %108) + %73#12 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %143 = amdgpu.scaled_mfma 16x16x128 (%79[1] * %101) * (%76[1] * %109) + %73#13 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %144 = amdgpu.scaled_mfma 16x16x128 (%79[1] * %101) * (%75[0] * %110) + %73#14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %145 = amdgpu.scaled_mfma 16x16x128 (%79[1] * %101) * (%75[1] * %111) + %73#15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %146 = amdgpu.scaled_mfma 16x16x128 (%74[0] * %102) * (%78[0] * %104) + %73#16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %147 = amdgpu.scaled_mfma 16x16x128 (%74[0] * %102) * (%78[1] * %105) + %73#17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %148 = amdgpu.scaled_mfma 16x16x128 (%74[0] * %102) * (%77[0] * %106) + %73#18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %149 = amdgpu.scaled_mfma 16x16x128 (%74[0] * %102) * (%77[1] * %107) + %73#19 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %150 = amdgpu.scaled_mfma 16x16x128 (%74[0] * %102) * (%76[0] * %108) + %73#20 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %151 = amdgpu.scaled_mfma 16x16x128 (%74[0] * %102) * (%76[1] * %109) + %73#21 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %152 = amdgpu.scaled_mfma 16x16x128 (%74[0] * %102) * (%75[0] * %110) + %73#22 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %153 = amdgpu.scaled_mfma 16x16x128 (%74[0] * %102) * (%75[1] * %111) + %73#23 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %154 = amdgpu.scaled_mfma 16x16x128 (%74[1] * %103) * (%78[0] * %104) + %73#24 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %155 = amdgpu.scaled_mfma 16x16x128 (%74[1] * %103) * (%78[1] * %105) + %73#25 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %156 = amdgpu.scaled_mfma 16x16x128 (%74[1] * %103) * (%77[0] * %106) + %73#26 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %157 = amdgpu.scaled_mfma 16x16x128 (%74[1] * %103) * (%77[1] * %107) + %73#27 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %158 = amdgpu.scaled_mfma 16x16x128 (%74[1] * %103) * (%76[0] * %108) + %73#28 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %159 = amdgpu.scaled_mfma 16x16x128 (%74[1] * %103) * (%76[1] * %109) + %73#29 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %160 = amdgpu.scaled_mfma 16x16x128 (%74[1] * %103) * (%75[0] * %110) + %73#30 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %161 = amdgpu.scaled_mfma 16x16x128 (%74[1] * %103) * (%75[1] * %111) + %73#31 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        %162 = vector.load %reinterpret_cast_10[%61] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %163 = vector.load %reinterpret_cast_10[%62] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %164 = vector.load %reinterpret_cast_10[%63] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %165 = vector.load %reinterpret_cast_10[%64] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %166 = vector.load %reinterpret_cast_11[%65] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %167 = vector.load %reinterpret_cast_11[%66] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %168 = vector.load %reinterpret_cast_11[%67] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %169 = vector.load %reinterpret_cast_11[%68] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %170 = vector.load %reinterpret_cast_11[%69] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %171 = vector.load %reinterpret_cast_11[%70] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %172 = vector.load %reinterpret_cast_11[%71] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %173 = vector.load %reinterpret_cast_11[%72] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %174 = vector.bitcast %162 : vector<16xi8> to vector<32xf4E2M1FN>
        %175 = vector.bitcast %163 : vector<16xi8> to vector<32xf4E2M1FN>
        %176 = vector.bitcast %164 : vector<16xi8> to vector<32xf4E2M1FN>
        %177 = vector.bitcast %165 : vector<16xi8> to vector<32xf4E2M1FN>
        %178 = vector.bitcast %166 : vector<16xi8> to vector<32xf4E2M1FN>
        %179 = vector.bitcast %167 : vector<16xi8> to vector<32xf4E2M1FN>
        %180 = vector.bitcast %168 : vector<16xi8> to vector<32xf4E2M1FN>
        %181 = vector.bitcast %169 : vector<16xi8> to vector<32xf4E2M1FN>
        %182 = vector.bitcast %170 : vector<16xi8> to vector<32xf4E2M1FN>
        %183 = vector.bitcast %171 : vector<16xi8> to vector<32xf4E2M1FN>
        %184 = vector.bitcast %172 : vector<16xi8> to vector<32xf4E2M1FN>
        %185 = vector.bitcast %173 : vector<16xi8> to vector<32xf4E2M1FN>
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(6)
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %186 = amdgpu.scaled_mfma 16x16x128 (%79[2] * %174) * (%78[2] * %178) + %130 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %187 = amdgpu.scaled_mfma 16x16x128 (%79[2] * %174) * (%78[3] * %179) + %131 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %188 = amdgpu.scaled_mfma 16x16x128 (%79[2] * %174) * (%77[2] * %180) + %132 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %189 = amdgpu.scaled_mfma 16x16x128 (%79[2] * %174) * (%77[3] * %181) + %133 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %190 = amdgpu.scaled_mfma 16x16x128 (%79[2] * %174) * (%76[2] * %182) + %134 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %191 = amdgpu.scaled_mfma 16x16x128 (%79[2] * %174) * (%76[3] * %183) + %135 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %192 = amdgpu.scaled_mfma 16x16x128 (%79[2] * %174) * (%75[2] * %184) + %136 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %193 = amdgpu.scaled_mfma 16x16x128 (%79[2] * %174) * (%75[3] * %185) + %137 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %194 = amdgpu.scaled_mfma 16x16x128 (%79[3] * %175) * (%78[2] * %178) + %138 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %195 = amdgpu.scaled_mfma 16x16x128 (%79[3] * %175) * (%78[3] * %179) + %139 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %196 = amdgpu.scaled_mfma 16x16x128 (%79[3] * %175) * (%77[2] * %180) + %140 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %197 = amdgpu.scaled_mfma 16x16x128 (%79[3] * %175) * (%77[3] * %181) + %141 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %198 = amdgpu.scaled_mfma 16x16x128 (%79[3] * %175) * (%76[2] * %182) + %142 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %199 = amdgpu.scaled_mfma 16x16x128 (%79[3] * %175) * (%76[3] * %183) + %143 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %200 = amdgpu.scaled_mfma 16x16x128 (%79[3] * %175) * (%75[2] * %184) + %144 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %201 = amdgpu.scaled_mfma 16x16x128 (%79[3] * %175) * (%75[3] * %185) + %145 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %202 = amdgpu.scaled_mfma 16x16x128 (%74[2] * %176) * (%78[2] * %178) + %146 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %203 = amdgpu.scaled_mfma 16x16x128 (%74[2] * %176) * (%78[3] * %179) + %147 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %204 = amdgpu.scaled_mfma 16x16x128 (%74[2] * %176) * (%77[2] * %180) + %148 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %205 = amdgpu.scaled_mfma 16x16x128 (%74[2] * %176) * (%77[3] * %181) + %149 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %206 = amdgpu.scaled_mfma 16x16x128 (%74[2] * %176) * (%76[2] * %182) + %150 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %207 = amdgpu.scaled_mfma 16x16x128 (%74[2] * %176) * (%76[3] * %183) + %151 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %208 = amdgpu.scaled_mfma 16x16x128 (%74[2] * %176) * (%75[2] * %184) + %152 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %209 = amdgpu.scaled_mfma 16x16x128 (%74[2] * %176) * (%75[3] * %185) + %153 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %210 = amdgpu.scaled_mfma 16x16x128 (%74[3] * %177) * (%78[2] * %178) + %154 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %211 = amdgpu.scaled_mfma 16x16x128 (%74[3] * %177) * (%78[3] * %179) + %155 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %212 = amdgpu.scaled_mfma 16x16x128 (%74[3] * %177) * (%77[2] * %180) + %156 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %213 = amdgpu.scaled_mfma 16x16x128 (%74[3] * %177) * (%77[3] * %181) + %157 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %214 = amdgpu.scaled_mfma 16x16x128 (%74[3] * %177) * (%76[2] * %182) + %158 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %215 = amdgpu.scaled_mfma 16x16x128 (%74[3] * %177) * (%76[3] * %183) + %159 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %216 = amdgpu.scaled_mfma 16x16x128 (%74[3] * %177) * (%75[2] * %184) + %160 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %217 = amdgpu.scaled_mfma 16x16x128 (%74[3] * %177) * (%75[3] * %185) + %161 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        scf.if %44 {
          rocdl.s.barrier
        }
        amdgpu.lds_barrier
        %218 = vector.load %reinterpret_cast_13[%51] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %219 = vector.load %reinterpret_cast_13[%65] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %220 = vector.load %reinterpret_cast_13[%52] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %221 = vector.load %reinterpret_cast_13[%66] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %222 = vector.load %reinterpret_cast_13[%53] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %223 = vector.load %reinterpret_cast_13[%67] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %224 = vector.load %reinterpret_cast_13[%54] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %225 = vector.load %reinterpret_cast_13[%68] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %226 = vector.load %reinterpret_cast_13[%55] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %227 = vector.load %reinterpret_cast_13[%69] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %228 = vector.load %reinterpret_cast_13[%56] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %229 = vector.load %reinterpret_cast_13[%70] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %230 = vector.load %reinterpret_cast_13[%57] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %231 = vector.load %reinterpret_cast_13[%71] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %232 = vector.load %reinterpret_cast_13[%58] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %233 = vector.load %reinterpret_cast_13[%72] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %234 = vector.load %reinterpret_cast_12[%47] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %235 = vector.load %reinterpret_cast_12[%61] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %236 = vector.load %reinterpret_cast_12[%48] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %237 = vector.load %reinterpret_cast_12[%62] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %238 = vector.load %reinterpret_cast_12[%49] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %239 = vector.load %reinterpret_cast_12[%63] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %240 = vector.load %reinterpret_cast_12[%50] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %241 = vector.load %reinterpret_cast_12[%64] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %242 = vector.bitcast %234 : vector<16xi8> to vector<32xf4E2M1FN>
        %243 = vector.bitcast %235 : vector<16xi8> to vector<32xf4E2M1FN>
        %244 = vector.bitcast %236 : vector<16xi8> to vector<32xf4E2M1FN>
        %245 = vector.bitcast %237 : vector<16xi8> to vector<32xf4E2M1FN>
        %246 = vector.bitcast %238 : vector<16xi8> to vector<32xf4E2M1FN>
        %247 = vector.bitcast %239 : vector<16xi8> to vector<32xf4E2M1FN>
        %248 = vector.bitcast %240 : vector<16xi8> to vector<32xf4E2M1FN>
        %249 = vector.bitcast %241 : vector<16xi8> to vector<32xf4E2M1FN>
        %250 = vector.bitcast %218 : vector<16xi8> to vector<32xf4E2M1FN>
        %251 = vector.bitcast %219 : vector<16xi8> to vector<32xf4E2M1FN>
        %252 = vector.bitcast %220 : vector<16xi8> to vector<32xf4E2M1FN>
        %253 = vector.bitcast %221 : vector<16xi8> to vector<32xf4E2M1FN>
        %254 = vector.bitcast %222 : vector<16xi8> to vector<32xf4E2M1FN>
        %255 = vector.bitcast %223 : vector<16xi8> to vector<32xf4E2M1FN>
        %256 = vector.bitcast %224 : vector<16xi8> to vector<32xf4E2M1FN>
        %257 = vector.bitcast %225 : vector<16xi8> to vector<32xf4E2M1FN>
        %258 = vector.bitcast %226 : vector<16xi8> to vector<32xf4E2M1FN>
        %259 = vector.bitcast %227 : vector<16xi8> to vector<32xf4E2M1FN>
        %260 = vector.bitcast %228 : vector<16xi8> to vector<32xf4E2M1FN>
        %261 = vector.bitcast %229 : vector<16xi8> to vector<32xf4E2M1FN>
        %262 = vector.bitcast %230 : vector<16xi8> to vector<32xf4E2M1FN>
        %263 = vector.bitcast %231 : vector<16xi8> to vector<32xf4E2M1FN>
        %264 = vector.bitcast %232 : vector<16xi8> to vector<32xf4E2M1FN>
        %265 = vector.bitcast %233 : vector<16xi8> to vector<32xf4E2M1FN>
        %266 = amdgpu.scaled_mfma 16x16x128 (%114[0] * %242) * (%120[0] * %250) + %186 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %267 = amdgpu.scaled_mfma 16x16x128 (%114[2] * %243) * (%120[2] * %251) + %266 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %268 = amdgpu.scaled_mfma 16x16x128 (%114[0] * %242) * (%120[1] * %252) + %187 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %269 = amdgpu.scaled_mfma 16x16x128 (%114[2] * %243) * (%120[3] * %253) + %268 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %270 = amdgpu.scaled_mfma 16x16x128 (%114[0] * %242) * (%123[0] * %254) + %188 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %271 = amdgpu.scaled_mfma 16x16x128 (%114[2] * %243) * (%123[2] * %255) + %270 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %272 = amdgpu.scaled_mfma 16x16x128 (%114[0] * %242) * (%123[1] * %256) + %189 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %273 = amdgpu.scaled_mfma 16x16x128 (%114[2] * %243) * (%123[3] * %257) + %272 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %274 = amdgpu.scaled_mfma 16x16x128 (%114[0] * %242) * (%126[0] * %258) + %190 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %275 = amdgpu.scaled_mfma 16x16x128 (%114[2] * %243) * (%126[2] * %259) + %274 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %276 = amdgpu.scaled_mfma 16x16x128 (%114[0] * %242) * (%126[1] * %260) + %191 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %277 = amdgpu.scaled_mfma 16x16x128 (%114[2] * %243) * (%126[3] * %261) + %276 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %278 = amdgpu.scaled_mfma 16x16x128 (%114[0] * %242) * (%129[0] * %262) + %192 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %279 = amdgpu.scaled_mfma 16x16x128 (%114[2] * %243) * (%129[2] * %263) + %278 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %280 = amdgpu.scaled_mfma 16x16x128 (%114[0] * %242) * (%129[1] * %264) + %193 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %281 = amdgpu.scaled_mfma 16x16x128 (%114[2] * %243) * (%129[3] * %265) + %280 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %282 = amdgpu.scaled_mfma 16x16x128 (%114[1] * %244) * (%120[0] * %250) + %194 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %283 = amdgpu.scaled_mfma 16x16x128 (%114[3] * %245) * (%120[2] * %251) + %282 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %284 = amdgpu.scaled_mfma 16x16x128 (%114[1] * %244) * (%120[1] * %252) + %195 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %285 = amdgpu.scaled_mfma 16x16x128 (%114[3] * %245) * (%120[3] * %253) + %284 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %286 = amdgpu.scaled_mfma 16x16x128 (%114[1] * %244) * (%123[0] * %254) + %196 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %287 = amdgpu.scaled_mfma 16x16x128 (%114[3] * %245) * (%123[2] * %255) + %286 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %288 = amdgpu.scaled_mfma 16x16x128 (%114[1] * %244) * (%123[1] * %256) + %197 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %289 = amdgpu.scaled_mfma 16x16x128 (%114[3] * %245) * (%123[3] * %257) + %288 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %290 = amdgpu.scaled_mfma 16x16x128 (%114[1] * %244) * (%126[0] * %258) + %198 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %291 = amdgpu.scaled_mfma 16x16x128 (%114[3] * %245) * (%126[2] * %259) + %290 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %292 = amdgpu.scaled_mfma 16x16x128 (%114[1] * %244) * (%126[1] * %260) + %199 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %293 = amdgpu.scaled_mfma 16x16x128 (%114[3] * %245) * (%126[3] * %261) + %292 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %294 = amdgpu.scaled_mfma 16x16x128 (%114[1] * %244) * (%129[0] * %262) + %200 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %295 = amdgpu.scaled_mfma 16x16x128 (%114[3] * %245) * (%129[2] * %263) + %294 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %296 = amdgpu.scaled_mfma 16x16x128 (%114[1] * %244) * (%129[1] * %264) + %201 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %297 = amdgpu.scaled_mfma 16x16x128 (%114[3] * %245) * (%129[3] * %265) + %296 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %298 = amdgpu.scaled_mfma 16x16x128 (%117[0] * %246) * (%120[0] * %250) + %202 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %299 = amdgpu.scaled_mfma 16x16x128 (%117[2] * %247) * (%120[2] * %251) + %298 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %300 = amdgpu.scaled_mfma 16x16x128 (%117[0] * %246) * (%120[1] * %252) + %203 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %301 = amdgpu.scaled_mfma 16x16x128 (%117[2] * %247) * (%120[3] * %253) + %300 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %302 = amdgpu.scaled_mfma 16x16x128 (%117[0] * %246) * (%123[0] * %254) + %204 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %303 = amdgpu.scaled_mfma 16x16x128 (%117[2] * %247) * (%123[2] * %255) + %302 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %304 = amdgpu.scaled_mfma 16x16x128 (%117[0] * %246) * (%123[1] * %256) + %205 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %305 = amdgpu.scaled_mfma 16x16x128 (%117[2] * %247) * (%123[3] * %257) + %304 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %306 = amdgpu.scaled_mfma 16x16x128 (%117[0] * %246) * (%126[0] * %258) + %206 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %307 = amdgpu.scaled_mfma 16x16x128 (%117[2] * %247) * (%126[2] * %259) + %306 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %308 = amdgpu.scaled_mfma 16x16x128 (%117[0] * %246) * (%126[1] * %260) + %207 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %309 = amdgpu.scaled_mfma 16x16x128 (%117[2] * %247) * (%126[3] * %261) + %308 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %310 = amdgpu.scaled_mfma 16x16x128 (%117[0] * %246) * (%129[0] * %262) + %208 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %311 = amdgpu.scaled_mfma 16x16x128 (%117[2] * %247) * (%129[2] * %263) + %310 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %312 = amdgpu.scaled_mfma 16x16x128 (%117[0] * %246) * (%129[1] * %264) + %209 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %313 = amdgpu.scaled_mfma 16x16x128 (%117[2] * %247) * (%129[3] * %265) + %312 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %314 = amdgpu.scaled_mfma 16x16x128 (%117[1] * %248) * (%120[0] * %250) + %210 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %315 = amdgpu.scaled_mfma 16x16x128 (%117[3] * %249) * (%120[2] * %251) + %314 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %316 = amdgpu.scaled_mfma 16x16x128 (%117[1] * %248) * (%120[1] * %252) + %211 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %317 = amdgpu.scaled_mfma 16x16x128 (%117[3] * %249) * (%120[3] * %253) + %316 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %318 = amdgpu.scaled_mfma 16x16x128 (%117[1] * %248) * (%123[0] * %254) + %212 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %319 = amdgpu.scaled_mfma 16x16x128 (%117[3] * %249) * (%123[2] * %255) + %318 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %320 = amdgpu.scaled_mfma 16x16x128 (%117[1] * %248) * (%123[1] * %256) + %213 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %321 = amdgpu.scaled_mfma 16x16x128 (%117[3] * %249) * (%123[3] * %257) + %320 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %322 = amdgpu.scaled_mfma 16x16x128 (%117[1] * %248) * (%126[0] * %258) + %214 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %323 = amdgpu.scaled_mfma 16x16x128 (%117[3] * %249) * (%126[2] * %259) + %322 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %324 = amdgpu.scaled_mfma 16x16x128 (%117[1] * %248) * (%126[1] * %260) + %215 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %325 = amdgpu.scaled_mfma 16x16x128 (%117[3] * %249) * (%126[3] * %261) + %324 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %326 = amdgpu.scaled_mfma 16x16x128 (%117[1] * %248) * (%129[0] * %262) + %216 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %327 = amdgpu.scaled_mfma 16x16x128 (%117[3] * %249) * (%129[2] * %263) + %326 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %328 = amdgpu.scaled_mfma 16x16x128 (%117[1] * %248) * (%129[1] * %264) + %217 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %329 = amdgpu.scaled_mfma 16x16x128 (%117[3] * %249) * (%129[3] * %265) + %328 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        // ── Optimized epilogue: XOR-1 shuffle → vector<2xbf16> stores ─────
        %ep_block_x = affine.apply #map51()[%block_id_x]
        %ep_block_y = affine.apply #map51()[%block_id_y]
        %ep_base_buf, %ep_offset, %ep_sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<1024x1024xbf16, strided<[?, 1]>> -> memref<bf16>, index, index, index, index, index
        %ep_block_row_off = arith.muli %ep_block_x, %strides#0 overflow<nsw> : index
        %ep_base_off = arith.addi %ep_block_row_off, %ep_block_y overflow<nsw> : index
        %ep_rcast = memref.reinterpret_cast %4 to offset: [%ep_base_off], sizes: [1073741822], strides: [1] : memref<bf16> to memref<1073741822xbf16, strided<[1], offset: ?>>
        %ep_cast  = memref.cast %ep_rcast : memref<1073741822xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
        %ep_stride_i14 = arith.index_cast %strides#0 : index to i14
        %ep_buffer = amdgpu.fat_raw_buffer_cast %ep_cast validBytes(%c2147483645_i64) cacheSwizzleStride(%ep_stride_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>

        %ep_shuffle_offset = arith.constant 1 : i32
        %ep_shuffle_width  = arith.constant 64 : i32
        %ep_one  = arith.constant 1 : index
        %ep_zero = arith.constant 0 : index
        %ep_lane_parity = arith.andi %thread_id_x, %ep_one : index
        %ep_is_even = arith.cmpi eq, %ep_lane_parity, %ep_zero : index

        %ep_rb0_row0 = affine.apply #map52()[%thread_id_x]
        %ep_rb0_row1 = affine.apply #map54()[%thread_id_x]
        %ep_rb0_row2 = affine.apply #map55()[%thread_id_x]
        %ep_rb0_row3 = affine.apply #map56()[%thread_id_x]
        %ep_rb0_store_row_a = arith.select %ep_is_even, %ep_rb0_row0, %ep_rb0_row2 : index
        %ep_rb0_store_row_b = arith.select %ep_is_even, %ep_rb0_row1, %ep_rb0_row3 : index
        %ep_rb0_store_off_a = arith.muli %ep_rb0_store_row_a, %strides#0 overflow<nsw> : index
        %ep_rb0_store_off_b = arith.muli %ep_rb0_store_row_b, %strides#0 overflow<nsw> : index

        %ep_rb1_row0 = affine.apply #map64()[%thread_id_x]
        %ep_rb1_row1 = affine.apply #map65()[%thread_id_x]
        %ep_rb1_row2 = affine.apply #map66()[%thread_id_x]
        %ep_rb1_row3 = affine.apply #map67()[%thread_id_x]
        %ep_rb1_store_row_a = arith.select %ep_is_even, %ep_rb1_row0, %ep_rb1_row2 : index
        %ep_rb1_store_row_b = arith.select %ep_is_even, %ep_rb1_row1, %ep_rb1_row3 : index
        %ep_rb1_store_off_a = arith.muli %ep_rb1_store_row_a, %strides#0 overflow<nsw> : index
        %ep_rb1_store_off_b = arith.muli %ep_rb1_store_row_b, %strides#0 overflow<nsw> : index

        %ep_rb2_row0 = affine.apply #map68()[%thread_id_x]
        %ep_rb2_row1 = affine.apply #map69()[%thread_id_x]
        %ep_rb2_row2 = affine.apply #map70()[%thread_id_x]
        %ep_rb2_row3 = affine.apply #map71()[%thread_id_x]
        %ep_rb2_store_row_a = arith.select %ep_is_even, %ep_rb2_row0, %ep_rb2_row2 : index
        %ep_rb2_store_row_b = arith.select %ep_is_even, %ep_rb2_row1, %ep_rb2_row3 : index
        %ep_rb2_store_off_a = arith.muli %ep_rb2_store_row_a, %strides#0 overflow<nsw> : index
        %ep_rb2_store_off_b = arith.muli %ep_rb2_store_row_b, %strides#0 overflow<nsw> : index

        %ep_rb3_row0 = affine.apply #map72()[%thread_id_x]
        %ep_rb3_row1 = affine.apply #map73()[%thread_id_x]
        %ep_rb3_row2 = affine.apply #map74()[%thread_id_x]
        %ep_rb3_row3 = affine.apply #map75()[%thread_id_x]
        %ep_rb3_store_row_a = arith.select %ep_is_even, %ep_rb3_row0, %ep_rb3_row2 : index
        %ep_rb3_store_row_b = arith.select %ep_is_even, %ep_rb3_row1, %ep_rb3_row3 : index
        %ep_rb3_store_off_a = arith.muli %ep_rb3_store_row_a, %strides#0 overflow<nsw> : index
        %ep_rb3_store_off_b = arith.muli %ep_rb3_store_row_b, %strides#0 overflow<nsw> : index

        %ep_col0 = affine.apply #map53()[%thread_id_x, %thread_id_y]
        %ep_col0_adj = arith.subi %ep_col0, %ep_lane_parity : index
        %ep_col1 = affine.apply #map57()[%thread_id_x, %thread_id_y]
        %ep_col1_adj = arith.subi %ep_col1, %ep_lane_parity : index
        %ep_col2 = affine.apply #map58()[%thread_id_x, %thread_id_y]
        %ep_col2_adj = arith.subi %ep_col2, %ep_lane_parity : index
        %ep_col3 = affine.apply #map59()[%thread_id_x, %thread_id_y]
        %ep_col3_adj = arith.subi %ep_col3, %ep_lane_parity : index
        %ep_col4 = affine.apply #map60()[%thread_id_x, %thread_id_y]
        %ep_col4_adj = arith.subi %ep_col4, %ep_lane_parity : index
        %ep_col5 = affine.apply #map61()[%thread_id_x, %thread_id_y]
        %ep_col5_adj = arith.subi %ep_col5, %ep_lane_parity : index
        %ep_col6 = affine.apply #map62()[%thread_id_x, %thread_id_y]
        %ep_col6_adj = arith.subi %ep_col6, %ep_lane_parity : index
        %ep_col7 = affine.apply #map63()[%thread_id_x, %thread_id_y]
        %ep_col7_adj = arith.subi %ep_col7, %ep_lane_parity : index

        // MFMA 0: %267 → rb0, col0
        %ep_v267_row0 = vector.extract %267[0] : f32 from vector<4xf32>
        %ep_v267_row1 = vector.extract %267[1] : f32 from vector<4xf32>
        %ep_v267_row2 = vector.extract %267[2] : f32 from vector<4xf32>
        %ep_v267_row3 = vector.extract %267[3] : f32 from vector<4xf32>
        %ep_v267_row0_nbr, %ep_v267_row0_valid = gpu.shuffle xor %ep_v267_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v267_row1_nbr, %ep_v267_row1_valid = gpu.shuffle xor %ep_v267_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v267_row2_nbr, %ep_v267_row2_valid = gpu.shuffle xor %ep_v267_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v267_row3_nbr, %ep_v267_row3_valid = gpu.shuffle xor %ep_v267_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v267_row0_lo = arith.select %ep_is_even, %ep_v267_row0,     %ep_v267_row0_nbr : f32
        %ep_v267_row0_hi = arith.select %ep_is_even, %ep_v267_row0_nbr, %ep_v267_row0     : f32
        %ep_v267_row1_lo = arith.select %ep_is_even, %ep_v267_row1,     %ep_v267_row1_nbr : f32
        %ep_v267_row1_hi = arith.select %ep_is_even, %ep_v267_row1_nbr, %ep_v267_row1     : f32
        %ep_v267_row2_lo = arith.select %ep_is_even, %ep_v267_row2,     %ep_v267_row2_nbr : f32
        %ep_v267_row2_hi = arith.select %ep_is_even, %ep_v267_row2_nbr, %ep_v267_row2     : f32
        %ep_v267_row3_lo = arith.select %ep_is_even, %ep_v267_row3,     %ep_v267_row3_nbr : f32
        %ep_v267_row3_hi = arith.select %ep_is_even, %ep_v267_row3_nbr, %ep_v267_row3     : f32
        %ep_v267_store_a_lo = arith.select %ep_is_even, %ep_v267_row0_lo, %ep_v267_row2_lo : f32
        %ep_v267_store_a_hi = arith.select %ep_is_even, %ep_v267_row0_hi, %ep_v267_row2_hi : f32
        %ep_v267_store_b_lo = arith.select %ep_is_even, %ep_v267_row1_lo, %ep_v267_row3_lo : f32
        %ep_v267_store_b_hi = arith.select %ep_is_even, %ep_v267_row1_hi, %ep_v267_row3_hi : f32
        %ep_v267_pair_a_0 = vector.broadcast %ep_v267_store_a_lo : f32 to vector<2xf32>
        %ep_v267_pair_a   = vector.insert %ep_v267_store_a_hi, %ep_v267_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v267_store_a  = arith.truncf %ep_v267_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v267_pair_b_0 = vector.broadcast %ep_v267_store_b_lo : f32 to vector<2xf32>
        %ep_v267_pair_b   = vector.insert %ep_v267_store_b_hi, %ep_v267_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v267_store_b  = arith.truncf %ep_v267_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v267_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col0_adj overflow<nsw> : index
        %ep_v267_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col0_adj overflow<nsw> : index
        vector.store %ep_v267_store_a, %ep_buffer[%ep_v267_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v267_store_b, %ep_buffer[%ep_v267_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 1: %269 → rb0, col1
        %ep_v269_row0 = vector.extract %269[0] : f32 from vector<4xf32>
        %ep_v269_row1 = vector.extract %269[1] : f32 from vector<4xf32>
        %ep_v269_row2 = vector.extract %269[2] : f32 from vector<4xf32>
        %ep_v269_row3 = vector.extract %269[3] : f32 from vector<4xf32>
        %ep_v269_row0_nbr, %ep_v269_row0_valid = gpu.shuffle xor %ep_v269_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v269_row1_nbr, %ep_v269_row1_valid = gpu.shuffle xor %ep_v269_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v269_row2_nbr, %ep_v269_row2_valid = gpu.shuffle xor %ep_v269_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v269_row3_nbr, %ep_v269_row3_valid = gpu.shuffle xor %ep_v269_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v269_row0_lo = arith.select %ep_is_even, %ep_v269_row0,     %ep_v269_row0_nbr : f32
        %ep_v269_row0_hi = arith.select %ep_is_even, %ep_v269_row0_nbr, %ep_v269_row0     : f32
        %ep_v269_row1_lo = arith.select %ep_is_even, %ep_v269_row1,     %ep_v269_row1_nbr : f32
        %ep_v269_row1_hi = arith.select %ep_is_even, %ep_v269_row1_nbr, %ep_v269_row1     : f32
        %ep_v269_row2_lo = arith.select %ep_is_even, %ep_v269_row2,     %ep_v269_row2_nbr : f32
        %ep_v269_row2_hi = arith.select %ep_is_even, %ep_v269_row2_nbr, %ep_v269_row2     : f32
        %ep_v269_row3_lo = arith.select %ep_is_even, %ep_v269_row3,     %ep_v269_row3_nbr : f32
        %ep_v269_row3_hi = arith.select %ep_is_even, %ep_v269_row3_nbr, %ep_v269_row3     : f32
        %ep_v269_store_a_lo = arith.select %ep_is_even, %ep_v269_row0_lo, %ep_v269_row2_lo : f32
        %ep_v269_store_a_hi = arith.select %ep_is_even, %ep_v269_row0_hi, %ep_v269_row2_hi : f32
        %ep_v269_store_b_lo = arith.select %ep_is_even, %ep_v269_row1_lo, %ep_v269_row3_lo : f32
        %ep_v269_store_b_hi = arith.select %ep_is_even, %ep_v269_row1_hi, %ep_v269_row3_hi : f32
        %ep_v269_pair_a_0 = vector.broadcast %ep_v269_store_a_lo : f32 to vector<2xf32>
        %ep_v269_pair_a   = vector.insert %ep_v269_store_a_hi, %ep_v269_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v269_store_a  = arith.truncf %ep_v269_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v269_pair_b_0 = vector.broadcast %ep_v269_store_b_lo : f32 to vector<2xf32>
        %ep_v269_pair_b   = vector.insert %ep_v269_store_b_hi, %ep_v269_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v269_store_b  = arith.truncf %ep_v269_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v269_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col1_adj overflow<nsw> : index
        %ep_v269_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col1_adj overflow<nsw> : index
        vector.store %ep_v269_store_a, %ep_buffer[%ep_v269_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v269_store_b, %ep_buffer[%ep_v269_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 2: %271 → rb0, col2
        %ep_v271_row0 = vector.extract %271[0] : f32 from vector<4xf32>
        %ep_v271_row1 = vector.extract %271[1] : f32 from vector<4xf32>
        %ep_v271_row2 = vector.extract %271[2] : f32 from vector<4xf32>
        %ep_v271_row3 = vector.extract %271[3] : f32 from vector<4xf32>
        %ep_v271_row0_nbr, %ep_v271_row0_valid = gpu.shuffle xor %ep_v271_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v271_row1_nbr, %ep_v271_row1_valid = gpu.shuffle xor %ep_v271_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v271_row2_nbr, %ep_v271_row2_valid = gpu.shuffle xor %ep_v271_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v271_row3_nbr, %ep_v271_row3_valid = gpu.shuffle xor %ep_v271_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v271_row0_lo = arith.select %ep_is_even, %ep_v271_row0,     %ep_v271_row0_nbr : f32
        %ep_v271_row0_hi = arith.select %ep_is_even, %ep_v271_row0_nbr, %ep_v271_row0     : f32
        %ep_v271_row1_lo = arith.select %ep_is_even, %ep_v271_row1,     %ep_v271_row1_nbr : f32
        %ep_v271_row1_hi = arith.select %ep_is_even, %ep_v271_row1_nbr, %ep_v271_row1     : f32
        %ep_v271_row2_lo = arith.select %ep_is_even, %ep_v271_row2,     %ep_v271_row2_nbr : f32
        %ep_v271_row2_hi = arith.select %ep_is_even, %ep_v271_row2_nbr, %ep_v271_row2     : f32
        %ep_v271_row3_lo = arith.select %ep_is_even, %ep_v271_row3,     %ep_v271_row3_nbr : f32
        %ep_v271_row3_hi = arith.select %ep_is_even, %ep_v271_row3_nbr, %ep_v271_row3     : f32
        %ep_v271_store_a_lo = arith.select %ep_is_even, %ep_v271_row0_lo, %ep_v271_row2_lo : f32
        %ep_v271_store_a_hi = arith.select %ep_is_even, %ep_v271_row0_hi, %ep_v271_row2_hi : f32
        %ep_v271_store_b_lo = arith.select %ep_is_even, %ep_v271_row1_lo, %ep_v271_row3_lo : f32
        %ep_v271_store_b_hi = arith.select %ep_is_even, %ep_v271_row1_hi, %ep_v271_row3_hi : f32
        %ep_v271_pair_a_0 = vector.broadcast %ep_v271_store_a_lo : f32 to vector<2xf32>
        %ep_v271_pair_a   = vector.insert %ep_v271_store_a_hi, %ep_v271_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v271_store_a  = arith.truncf %ep_v271_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v271_pair_b_0 = vector.broadcast %ep_v271_store_b_lo : f32 to vector<2xf32>
        %ep_v271_pair_b   = vector.insert %ep_v271_store_b_hi, %ep_v271_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v271_store_b  = arith.truncf %ep_v271_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v271_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col2_adj overflow<nsw> : index
        %ep_v271_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col2_adj overflow<nsw> : index
        vector.store %ep_v271_store_a, %ep_buffer[%ep_v271_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v271_store_b, %ep_buffer[%ep_v271_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 3: %273 → rb0, col3
        %ep_v273_row0 = vector.extract %273[0] : f32 from vector<4xf32>
        %ep_v273_row1 = vector.extract %273[1] : f32 from vector<4xf32>
        %ep_v273_row2 = vector.extract %273[2] : f32 from vector<4xf32>
        %ep_v273_row3 = vector.extract %273[3] : f32 from vector<4xf32>
        %ep_v273_row0_nbr, %ep_v273_row0_valid = gpu.shuffle xor %ep_v273_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v273_row1_nbr, %ep_v273_row1_valid = gpu.shuffle xor %ep_v273_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v273_row2_nbr, %ep_v273_row2_valid = gpu.shuffle xor %ep_v273_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v273_row3_nbr, %ep_v273_row3_valid = gpu.shuffle xor %ep_v273_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v273_row0_lo = arith.select %ep_is_even, %ep_v273_row0,     %ep_v273_row0_nbr : f32
        %ep_v273_row0_hi = arith.select %ep_is_even, %ep_v273_row0_nbr, %ep_v273_row0     : f32
        %ep_v273_row1_lo = arith.select %ep_is_even, %ep_v273_row1,     %ep_v273_row1_nbr : f32
        %ep_v273_row1_hi = arith.select %ep_is_even, %ep_v273_row1_nbr, %ep_v273_row1     : f32
        %ep_v273_row2_lo = arith.select %ep_is_even, %ep_v273_row2,     %ep_v273_row2_nbr : f32
        %ep_v273_row2_hi = arith.select %ep_is_even, %ep_v273_row2_nbr, %ep_v273_row2     : f32
        %ep_v273_row3_lo = arith.select %ep_is_even, %ep_v273_row3,     %ep_v273_row3_nbr : f32
        %ep_v273_row3_hi = arith.select %ep_is_even, %ep_v273_row3_nbr, %ep_v273_row3     : f32
        %ep_v273_store_a_lo = arith.select %ep_is_even, %ep_v273_row0_lo, %ep_v273_row2_lo : f32
        %ep_v273_store_a_hi = arith.select %ep_is_even, %ep_v273_row0_hi, %ep_v273_row2_hi : f32
        %ep_v273_store_b_lo = arith.select %ep_is_even, %ep_v273_row1_lo, %ep_v273_row3_lo : f32
        %ep_v273_store_b_hi = arith.select %ep_is_even, %ep_v273_row1_hi, %ep_v273_row3_hi : f32
        %ep_v273_pair_a_0 = vector.broadcast %ep_v273_store_a_lo : f32 to vector<2xf32>
        %ep_v273_pair_a   = vector.insert %ep_v273_store_a_hi, %ep_v273_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v273_store_a  = arith.truncf %ep_v273_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v273_pair_b_0 = vector.broadcast %ep_v273_store_b_lo : f32 to vector<2xf32>
        %ep_v273_pair_b   = vector.insert %ep_v273_store_b_hi, %ep_v273_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v273_store_b  = arith.truncf %ep_v273_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v273_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col3_adj overflow<nsw> : index
        %ep_v273_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col3_adj overflow<nsw> : index
        vector.store %ep_v273_store_a, %ep_buffer[%ep_v273_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v273_store_b, %ep_buffer[%ep_v273_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 4: %275 → rb0, col4
        %ep_v275_row0 = vector.extract %275[0] : f32 from vector<4xf32>
        %ep_v275_row1 = vector.extract %275[1] : f32 from vector<4xf32>
        %ep_v275_row2 = vector.extract %275[2] : f32 from vector<4xf32>
        %ep_v275_row3 = vector.extract %275[3] : f32 from vector<4xf32>
        %ep_v275_row0_nbr, %ep_v275_row0_valid = gpu.shuffle xor %ep_v275_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v275_row1_nbr, %ep_v275_row1_valid = gpu.shuffle xor %ep_v275_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v275_row2_nbr, %ep_v275_row2_valid = gpu.shuffle xor %ep_v275_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v275_row3_nbr, %ep_v275_row3_valid = gpu.shuffle xor %ep_v275_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v275_row0_lo = arith.select %ep_is_even, %ep_v275_row0,     %ep_v275_row0_nbr : f32
        %ep_v275_row0_hi = arith.select %ep_is_even, %ep_v275_row0_nbr, %ep_v275_row0     : f32
        %ep_v275_row1_lo = arith.select %ep_is_even, %ep_v275_row1,     %ep_v275_row1_nbr : f32
        %ep_v275_row1_hi = arith.select %ep_is_even, %ep_v275_row1_nbr, %ep_v275_row1     : f32
        %ep_v275_row2_lo = arith.select %ep_is_even, %ep_v275_row2,     %ep_v275_row2_nbr : f32
        %ep_v275_row2_hi = arith.select %ep_is_even, %ep_v275_row2_nbr, %ep_v275_row2     : f32
        %ep_v275_row3_lo = arith.select %ep_is_even, %ep_v275_row3,     %ep_v275_row3_nbr : f32
        %ep_v275_row3_hi = arith.select %ep_is_even, %ep_v275_row3_nbr, %ep_v275_row3     : f32
        %ep_v275_store_a_lo = arith.select %ep_is_even, %ep_v275_row0_lo, %ep_v275_row2_lo : f32
        %ep_v275_store_a_hi = arith.select %ep_is_even, %ep_v275_row0_hi, %ep_v275_row2_hi : f32
        %ep_v275_store_b_lo = arith.select %ep_is_even, %ep_v275_row1_lo, %ep_v275_row3_lo : f32
        %ep_v275_store_b_hi = arith.select %ep_is_even, %ep_v275_row1_hi, %ep_v275_row3_hi : f32
        %ep_v275_pair_a_0 = vector.broadcast %ep_v275_store_a_lo : f32 to vector<2xf32>
        %ep_v275_pair_a   = vector.insert %ep_v275_store_a_hi, %ep_v275_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v275_store_a  = arith.truncf %ep_v275_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v275_pair_b_0 = vector.broadcast %ep_v275_store_b_lo : f32 to vector<2xf32>
        %ep_v275_pair_b   = vector.insert %ep_v275_store_b_hi, %ep_v275_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v275_store_b  = arith.truncf %ep_v275_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v275_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col4_adj overflow<nsw> : index
        %ep_v275_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col4_adj overflow<nsw> : index
        vector.store %ep_v275_store_a, %ep_buffer[%ep_v275_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v275_store_b, %ep_buffer[%ep_v275_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 5: %277 → rb0, col5
        %ep_v277_row0 = vector.extract %277[0] : f32 from vector<4xf32>
        %ep_v277_row1 = vector.extract %277[1] : f32 from vector<4xf32>
        %ep_v277_row2 = vector.extract %277[2] : f32 from vector<4xf32>
        %ep_v277_row3 = vector.extract %277[3] : f32 from vector<4xf32>
        %ep_v277_row0_nbr, %ep_v277_row0_valid = gpu.shuffle xor %ep_v277_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v277_row1_nbr, %ep_v277_row1_valid = gpu.shuffle xor %ep_v277_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v277_row2_nbr, %ep_v277_row2_valid = gpu.shuffle xor %ep_v277_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v277_row3_nbr, %ep_v277_row3_valid = gpu.shuffle xor %ep_v277_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v277_row0_lo = arith.select %ep_is_even, %ep_v277_row0,     %ep_v277_row0_nbr : f32
        %ep_v277_row0_hi = arith.select %ep_is_even, %ep_v277_row0_nbr, %ep_v277_row0     : f32
        %ep_v277_row1_lo = arith.select %ep_is_even, %ep_v277_row1,     %ep_v277_row1_nbr : f32
        %ep_v277_row1_hi = arith.select %ep_is_even, %ep_v277_row1_nbr, %ep_v277_row1     : f32
        %ep_v277_row2_lo = arith.select %ep_is_even, %ep_v277_row2,     %ep_v277_row2_nbr : f32
        %ep_v277_row2_hi = arith.select %ep_is_even, %ep_v277_row2_nbr, %ep_v277_row2     : f32
        %ep_v277_row3_lo = arith.select %ep_is_even, %ep_v277_row3,     %ep_v277_row3_nbr : f32
        %ep_v277_row3_hi = arith.select %ep_is_even, %ep_v277_row3_nbr, %ep_v277_row3     : f32
        %ep_v277_store_a_lo = arith.select %ep_is_even, %ep_v277_row0_lo, %ep_v277_row2_lo : f32
        %ep_v277_store_a_hi = arith.select %ep_is_even, %ep_v277_row0_hi, %ep_v277_row2_hi : f32
        %ep_v277_store_b_lo = arith.select %ep_is_even, %ep_v277_row1_lo, %ep_v277_row3_lo : f32
        %ep_v277_store_b_hi = arith.select %ep_is_even, %ep_v277_row1_hi, %ep_v277_row3_hi : f32
        %ep_v277_pair_a_0 = vector.broadcast %ep_v277_store_a_lo : f32 to vector<2xf32>
        %ep_v277_pair_a   = vector.insert %ep_v277_store_a_hi, %ep_v277_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v277_store_a  = arith.truncf %ep_v277_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v277_pair_b_0 = vector.broadcast %ep_v277_store_b_lo : f32 to vector<2xf32>
        %ep_v277_pair_b   = vector.insert %ep_v277_store_b_hi, %ep_v277_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v277_store_b  = arith.truncf %ep_v277_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v277_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col5_adj overflow<nsw> : index
        %ep_v277_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col5_adj overflow<nsw> : index
        vector.store %ep_v277_store_a, %ep_buffer[%ep_v277_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v277_store_b, %ep_buffer[%ep_v277_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 6: %279 → rb0, col6
        %ep_v279_row0 = vector.extract %279[0] : f32 from vector<4xf32>
        %ep_v279_row1 = vector.extract %279[1] : f32 from vector<4xf32>
        %ep_v279_row2 = vector.extract %279[2] : f32 from vector<4xf32>
        %ep_v279_row3 = vector.extract %279[3] : f32 from vector<4xf32>
        %ep_v279_row0_nbr, %ep_v279_row0_valid = gpu.shuffle xor %ep_v279_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v279_row1_nbr, %ep_v279_row1_valid = gpu.shuffle xor %ep_v279_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v279_row2_nbr, %ep_v279_row2_valid = gpu.shuffle xor %ep_v279_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v279_row3_nbr, %ep_v279_row3_valid = gpu.shuffle xor %ep_v279_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v279_row0_lo = arith.select %ep_is_even, %ep_v279_row0,     %ep_v279_row0_nbr : f32
        %ep_v279_row0_hi = arith.select %ep_is_even, %ep_v279_row0_nbr, %ep_v279_row0     : f32
        %ep_v279_row1_lo = arith.select %ep_is_even, %ep_v279_row1,     %ep_v279_row1_nbr : f32
        %ep_v279_row1_hi = arith.select %ep_is_even, %ep_v279_row1_nbr, %ep_v279_row1     : f32
        %ep_v279_row2_lo = arith.select %ep_is_even, %ep_v279_row2,     %ep_v279_row2_nbr : f32
        %ep_v279_row2_hi = arith.select %ep_is_even, %ep_v279_row2_nbr, %ep_v279_row2     : f32
        %ep_v279_row3_lo = arith.select %ep_is_even, %ep_v279_row3,     %ep_v279_row3_nbr : f32
        %ep_v279_row3_hi = arith.select %ep_is_even, %ep_v279_row3_nbr, %ep_v279_row3     : f32
        %ep_v279_store_a_lo = arith.select %ep_is_even, %ep_v279_row0_lo, %ep_v279_row2_lo : f32
        %ep_v279_store_a_hi = arith.select %ep_is_even, %ep_v279_row0_hi, %ep_v279_row2_hi : f32
        %ep_v279_store_b_lo = arith.select %ep_is_even, %ep_v279_row1_lo, %ep_v279_row3_lo : f32
        %ep_v279_store_b_hi = arith.select %ep_is_even, %ep_v279_row1_hi, %ep_v279_row3_hi : f32
        %ep_v279_pair_a_0 = vector.broadcast %ep_v279_store_a_lo : f32 to vector<2xf32>
        %ep_v279_pair_a   = vector.insert %ep_v279_store_a_hi, %ep_v279_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v279_store_a  = arith.truncf %ep_v279_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v279_pair_b_0 = vector.broadcast %ep_v279_store_b_lo : f32 to vector<2xf32>
        %ep_v279_pair_b   = vector.insert %ep_v279_store_b_hi, %ep_v279_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v279_store_b  = arith.truncf %ep_v279_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v279_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col6_adj overflow<nsw> : index
        %ep_v279_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col6_adj overflow<nsw> : index
        vector.store %ep_v279_store_a, %ep_buffer[%ep_v279_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v279_store_b, %ep_buffer[%ep_v279_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 7: %281 → rb0, col7
        %ep_v281_row0 = vector.extract %281[0] : f32 from vector<4xf32>
        %ep_v281_row1 = vector.extract %281[1] : f32 from vector<4xf32>
        %ep_v281_row2 = vector.extract %281[2] : f32 from vector<4xf32>
        %ep_v281_row3 = vector.extract %281[3] : f32 from vector<4xf32>
        %ep_v281_row0_nbr, %ep_v281_row0_valid = gpu.shuffle xor %ep_v281_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v281_row1_nbr, %ep_v281_row1_valid = gpu.shuffle xor %ep_v281_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v281_row2_nbr, %ep_v281_row2_valid = gpu.shuffle xor %ep_v281_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v281_row3_nbr, %ep_v281_row3_valid = gpu.shuffle xor %ep_v281_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v281_row0_lo = arith.select %ep_is_even, %ep_v281_row0,     %ep_v281_row0_nbr : f32
        %ep_v281_row0_hi = arith.select %ep_is_even, %ep_v281_row0_nbr, %ep_v281_row0     : f32
        %ep_v281_row1_lo = arith.select %ep_is_even, %ep_v281_row1,     %ep_v281_row1_nbr : f32
        %ep_v281_row1_hi = arith.select %ep_is_even, %ep_v281_row1_nbr, %ep_v281_row1     : f32
        %ep_v281_row2_lo = arith.select %ep_is_even, %ep_v281_row2,     %ep_v281_row2_nbr : f32
        %ep_v281_row2_hi = arith.select %ep_is_even, %ep_v281_row2_nbr, %ep_v281_row2     : f32
        %ep_v281_row3_lo = arith.select %ep_is_even, %ep_v281_row3,     %ep_v281_row3_nbr : f32
        %ep_v281_row3_hi = arith.select %ep_is_even, %ep_v281_row3_nbr, %ep_v281_row3     : f32
        %ep_v281_store_a_lo = arith.select %ep_is_even, %ep_v281_row0_lo, %ep_v281_row2_lo : f32
        %ep_v281_store_a_hi = arith.select %ep_is_even, %ep_v281_row0_hi, %ep_v281_row2_hi : f32
        %ep_v281_store_b_lo = arith.select %ep_is_even, %ep_v281_row1_lo, %ep_v281_row3_lo : f32
        %ep_v281_store_b_hi = arith.select %ep_is_even, %ep_v281_row1_hi, %ep_v281_row3_hi : f32
        %ep_v281_pair_a_0 = vector.broadcast %ep_v281_store_a_lo : f32 to vector<2xf32>
        %ep_v281_pair_a   = vector.insert %ep_v281_store_a_hi, %ep_v281_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v281_store_a  = arith.truncf %ep_v281_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v281_pair_b_0 = vector.broadcast %ep_v281_store_b_lo : f32 to vector<2xf32>
        %ep_v281_pair_b   = vector.insert %ep_v281_store_b_hi, %ep_v281_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v281_store_b  = arith.truncf %ep_v281_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v281_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col7_adj overflow<nsw> : index
        %ep_v281_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col7_adj overflow<nsw> : index
        vector.store %ep_v281_store_a, %ep_buffer[%ep_v281_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v281_store_b, %ep_buffer[%ep_v281_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 8: %283 → rb1, col0
        %ep_v283_row0 = vector.extract %283[0] : f32 from vector<4xf32>
        %ep_v283_row1 = vector.extract %283[1] : f32 from vector<4xf32>
        %ep_v283_row2 = vector.extract %283[2] : f32 from vector<4xf32>
        %ep_v283_row3 = vector.extract %283[3] : f32 from vector<4xf32>
        %ep_v283_row0_nbr, %ep_v283_row0_valid = gpu.shuffle xor %ep_v283_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v283_row1_nbr, %ep_v283_row1_valid = gpu.shuffle xor %ep_v283_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v283_row2_nbr, %ep_v283_row2_valid = gpu.shuffle xor %ep_v283_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v283_row3_nbr, %ep_v283_row3_valid = gpu.shuffle xor %ep_v283_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v283_row0_lo = arith.select %ep_is_even, %ep_v283_row0,     %ep_v283_row0_nbr : f32
        %ep_v283_row0_hi = arith.select %ep_is_even, %ep_v283_row0_nbr, %ep_v283_row0     : f32
        %ep_v283_row1_lo = arith.select %ep_is_even, %ep_v283_row1,     %ep_v283_row1_nbr : f32
        %ep_v283_row1_hi = arith.select %ep_is_even, %ep_v283_row1_nbr, %ep_v283_row1     : f32
        %ep_v283_row2_lo = arith.select %ep_is_even, %ep_v283_row2,     %ep_v283_row2_nbr : f32
        %ep_v283_row2_hi = arith.select %ep_is_even, %ep_v283_row2_nbr, %ep_v283_row2     : f32
        %ep_v283_row3_lo = arith.select %ep_is_even, %ep_v283_row3,     %ep_v283_row3_nbr : f32
        %ep_v283_row3_hi = arith.select %ep_is_even, %ep_v283_row3_nbr, %ep_v283_row3     : f32
        %ep_v283_store_a_lo = arith.select %ep_is_even, %ep_v283_row0_lo, %ep_v283_row2_lo : f32
        %ep_v283_store_a_hi = arith.select %ep_is_even, %ep_v283_row0_hi, %ep_v283_row2_hi : f32
        %ep_v283_store_b_lo = arith.select %ep_is_even, %ep_v283_row1_lo, %ep_v283_row3_lo : f32
        %ep_v283_store_b_hi = arith.select %ep_is_even, %ep_v283_row1_hi, %ep_v283_row3_hi : f32
        %ep_v283_pair_a_0 = vector.broadcast %ep_v283_store_a_lo : f32 to vector<2xf32>
        %ep_v283_pair_a   = vector.insert %ep_v283_store_a_hi, %ep_v283_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v283_store_a  = arith.truncf %ep_v283_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v283_pair_b_0 = vector.broadcast %ep_v283_store_b_lo : f32 to vector<2xf32>
        %ep_v283_pair_b   = vector.insert %ep_v283_store_b_hi, %ep_v283_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v283_store_b  = arith.truncf %ep_v283_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v283_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col0_adj overflow<nsw> : index
        %ep_v283_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col0_adj overflow<nsw> : index
        vector.store %ep_v283_store_a, %ep_buffer[%ep_v283_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v283_store_b, %ep_buffer[%ep_v283_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 9: %285 → rb1, col1
        %ep_v285_row0 = vector.extract %285[0] : f32 from vector<4xf32>
        %ep_v285_row1 = vector.extract %285[1] : f32 from vector<4xf32>
        %ep_v285_row2 = vector.extract %285[2] : f32 from vector<4xf32>
        %ep_v285_row3 = vector.extract %285[3] : f32 from vector<4xf32>
        %ep_v285_row0_nbr, %ep_v285_row0_valid = gpu.shuffle xor %ep_v285_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v285_row1_nbr, %ep_v285_row1_valid = gpu.shuffle xor %ep_v285_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v285_row2_nbr, %ep_v285_row2_valid = gpu.shuffle xor %ep_v285_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v285_row3_nbr, %ep_v285_row3_valid = gpu.shuffle xor %ep_v285_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v285_row0_lo = arith.select %ep_is_even, %ep_v285_row0,     %ep_v285_row0_nbr : f32
        %ep_v285_row0_hi = arith.select %ep_is_even, %ep_v285_row0_nbr, %ep_v285_row0     : f32
        %ep_v285_row1_lo = arith.select %ep_is_even, %ep_v285_row1,     %ep_v285_row1_nbr : f32
        %ep_v285_row1_hi = arith.select %ep_is_even, %ep_v285_row1_nbr, %ep_v285_row1     : f32
        %ep_v285_row2_lo = arith.select %ep_is_even, %ep_v285_row2,     %ep_v285_row2_nbr : f32
        %ep_v285_row2_hi = arith.select %ep_is_even, %ep_v285_row2_nbr, %ep_v285_row2     : f32
        %ep_v285_row3_lo = arith.select %ep_is_even, %ep_v285_row3,     %ep_v285_row3_nbr : f32
        %ep_v285_row3_hi = arith.select %ep_is_even, %ep_v285_row3_nbr, %ep_v285_row3     : f32
        %ep_v285_store_a_lo = arith.select %ep_is_even, %ep_v285_row0_lo, %ep_v285_row2_lo : f32
        %ep_v285_store_a_hi = arith.select %ep_is_even, %ep_v285_row0_hi, %ep_v285_row2_hi : f32
        %ep_v285_store_b_lo = arith.select %ep_is_even, %ep_v285_row1_lo, %ep_v285_row3_lo : f32
        %ep_v285_store_b_hi = arith.select %ep_is_even, %ep_v285_row1_hi, %ep_v285_row3_hi : f32
        %ep_v285_pair_a_0 = vector.broadcast %ep_v285_store_a_lo : f32 to vector<2xf32>
        %ep_v285_pair_a   = vector.insert %ep_v285_store_a_hi, %ep_v285_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v285_store_a  = arith.truncf %ep_v285_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v285_pair_b_0 = vector.broadcast %ep_v285_store_b_lo : f32 to vector<2xf32>
        %ep_v285_pair_b   = vector.insert %ep_v285_store_b_hi, %ep_v285_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v285_store_b  = arith.truncf %ep_v285_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v285_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col1_adj overflow<nsw> : index
        %ep_v285_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col1_adj overflow<nsw> : index
        vector.store %ep_v285_store_a, %ep_buffer[%ep_v285_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v285_store_b, %ep_buffer[%ep_v285_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 10: %287 → rb1, col2
        %ep_v287_row0 = vector.extract %287[0] : f32 from vector<4xf32>
        %ep_v287_row1 = vector.extract %287[1] : f32 from vector<4xf32>
        %ep_v287_row2 = vector.extract %287[2] : f32 from vector<4xf32>
        %ep_v287_row3 = vector.extract %287[3] : f32 from vector<4xf32>
        %ep_v287_row0_nbr, %ep_v287_row0_valid = gpu.shuffle xor %ep_v287_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v287_row1_nbr, %ep_v287_row1_valid = gpu.shuffle xor %ep_v287_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v287_row2_nbr, %ep_v287_row2_valid = gpu.shuffle xor %ep_v287_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v287_row3_nbr, %ep_v287_row3_valid = gpu.shuffle xor %ep_v287_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v287_row0_lo = arith.select %ep_is_even, %ep_v287_row0,     %ep_v287_row0_nbr : f32
        %ep_v287_row0_hi = arith.select %ep_is_even, %ep_v287_row0_nbr, %ep_v287_row0     : f32
        %ep_v287_row1_lo = arith.select %ep_is_even, %ep_v287_row1,     %ep_v287_row1_nbr : f32
        %ep_v287_row1_hi = arith.select %ep_is_even, %ep_v287_row1_nbr, %ep_v287_row1     : f32
        %ep_v287_row2_lo = arith.select %ep_is_even, %ep_v287_row2,     %ep_v287_row2_nbr : f32
        %ep_v287_row2_hi = arith.select %ep_is_even, %ep_v287_row2_nbr, %ep_v287_row2     : f32
        %ep_v287_row3_lo = arith.select %ep_is_even, %ep_v287_row3,     %ep_v287_row3_nbr : f32
        %ep_v287_row3_hi = arith.select %ep_is_even, %ep_v287_row3_nbr, %ep_v287_row3     : f32
        %ep_v287_store_a_lo = arith.select %ep_is_even, %ep_v287_row0_lo, %ep_v287_row2_lo : f32
        %ep_v287_store_a_hi = arith.select %ep_is_even, %ep_v287_row0_hi, %ep_v287_row2_hi : f32
        %ep_v287_store_b_lo = arith.select %ep_is_even, %ep_v287_row1_lo, %ep_v287_row3_lo : f32
        %ep_v287_store_b_hi = arith.select %ep_is_even, %ep_v287_row1_hi, %ep_v287_row3_hi : f32
        %ep_v287_pair_a_0 = vector.broadcast %ep_v287_store_a_lo : f32 to vector<2xf32>
        %ep_v287_pair_a   = vector.insert %ep_v287_store_a_hi, %ep_v287_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v287_store_a  = arith.truncf %ep_v287_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v287_pair_b_0 = vector.broadcast %ep_v287_store_b_lo : f32 to vector<2xf32>
        %ep_v287_pair_b   = vector.insert %ep_v287_store_b_hi, %ep_v287_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v287_store_b  = arith.truncf %ep_v287_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v287_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col2_adj overflow<nsw> : index
        %ep_v287_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col2_adj overflow<nsw> : index
        vector.store %ep_v287_store_a, %ep_buffer[%ep_v287_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v287_store_b, %ep_buffer[%ep_v287_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 11: %289 → rb1, col3
        %ep_v289_row0 = vector.extract %289[0] : f32 from vector<4xf32>
        %ep_v289_row1 = vector.extract %289[1] : f32 from vector<4xf32>
        %ep_v289_row2 = vector.extract %289[2] : f32 from vector<4xf32>
        %ep_v289_row3 = vector.extract %289[3] : f32 from vector<4xf32>
        %ep_v289_row0_nbr, %ep_v289_row0_valid = gpu.shuffle xor %ep_v289_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v289_row1_nbr, %ep_v289_row1_valid = gpu.shuffle xor %ep_v289_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v289_row2_nbr, %ep_v289_row2_valid = gpu.shuffle xor %ep_v289_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v289_row3_nbr, %ep_v289_row3_valid = gpu.shuffle xor %ep_v289_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v289_row0_lo = arith.select %ep_is_even, %ep_v289_row0,     %ep_v289_row0_nbr : f32
        %ep_v289_row0_hi = arith.select %ep_is_even, %ep_v289_row0_nbr, %ep_v289_row0     : f32
        %ep_v289_row1_lo = arith.select %ep_is_even, %ep_v289_row1,     %ep_v289_row1_nbr : f32
        %ep_v289_row1_hi = arith.select %ep_is_even, %ep_v289_row1_nbr, %ep_v289_row1     : f32
        %ep_v289_row2_lo = arith.select %ep_is_even, %ep_v289_row2,     %ep_v289_row2_nbr : f32
        %ep_v289_row2_hi = arith.select %ep_is_even, %ep_v289_row2_nbr, %ep_v289_row2     : f32
        %ep_v289_row3_lo = arith.select %ep_is_even, %ep_v289_row3,     %ep_v289_row3_nbr : f32
        %ep_v289_row3_hi = arith.select %ep_is_even, %ep_v289_row3_nbr, %ep_v289_row3     : f32
        %ep_v289_store_a_lo = arith.select %ep_is_even, %ep_v289_row0_lo, %ep_v289_row2_lo : f32
        %ep_v289_store_a_hi = arith.select %ep_is_even, %ep_v289_row0_hi, %ep_v289_row2_hi : f32
        %ep_v289_store_b_lo = arith.select %ep_is_even, %ep_v289_row1_lo, %ep_v289_row3_lo : f32
        %ep_v289_store_b_hi = arith.select %ep_is_even, %ep_v289_row1_hi, %ep_v289_row3_hi : f32
        %ep_v289_pair_a_0 = vector.broadcast %ep_v289_store_a_lo : f32 to vector<2xf32>
        %ep_v289_pair_a   = vector.insert %ep_v289_store_a_hi, %ep_v289_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v289_store_a  = arith.truncf %ep_v289_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v289_pair_b_0 = vector.broadcast %ep_v289_store_b_lo : f32 to vector<2xf32>
        %ep_v289_pair_b   = vector.insert %ep_v289_store_b_hi, %ep_v289_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v289_store_b  = arith.truncf %ep_v289_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v289_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col3_adj overflow<nsw> : index
        %ep_v289_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col3_adj overflow<nsw> : index
        vector.store %ep_v289_store_a, %ep_buffer[%ep_v289_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v289_store_b, %ep_buffer[%ep_v289_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 12: %291 → rb1, col4
        %ep_v291_row0 = vector.extract %291[0] : f32 from vector<4xf32>
        %ep_v291_row1 = vector.extract %291[1] : f32 from vector<4xf32>
        %ep_v291_row2 = vector.extract %291[2] : f32 from vector<4xf32>
        %ep_v291_row3 = vector.extract %291[3] : f32 from vector<4xf32>
        %ep_v291_row0_nbr, %ep_v291_row0_valid = gpu.shuffle xor %ep_v291_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v291_row1_nbr, %ep_v291_row1_valid = gpu.shuffle xor %ep_v291_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v291_row2_nbr, %ep_v291_row2_valid = gpu.shuffle xor %ep_v291_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v291_row3_nbr, %ep_v291_row3_valid = gpu.shuffle xor %ep_v291_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v291_row0_lo = arith.select %ep_is_even, %ep_v291_row0,     %ep_v291_row0_nbr : f32
        %ep_v291_row0_hi = arith.select %ep_is_even, %ep_v291_row0_nbr, %ep_v291_row0     : f32
        %ep_v291_row1_lo = arith.select %ep_is_even, %ep_v291_row1,     %ep_v291_row1_nbr : f32
        %ep_v291_row1_hi = arith.select %ep_is_even, %ep_v291_row1_nbr, %ep_v291_row1     : f32
        %ep_v291_row2_lo = arith.select %ep_is_even, %ep_v291_row2,     %ep_v291_row2_nbr : f32
        %ep_v291_row2_hi = arith.select %ep_is_even, %ep_v291_row2_nbr, %ep_v291_row2     : f32
        %ep_v291_row3_lo = arith.select %ep_is_even, %ep_v291_row3,     %ep_v291_row3_nbr : f32
        %ep_v291_row3_hi = arith.select %ep_is_even, %ep_v291_row3_nbr, %ep_v291_row3     : f32
        %ep_v291_store_a_lo = arith.select %ep_is_even, %ep_v291_row0_lo, %ep_v291_row2_lo : f32
        %ep_v291_store_a_hi = arith.select %ep_is_even, %ep_v291_row0_hi, %ep_v291_row2_hi : f32
        %ep_v291_store_b_lo = arith.select %ep_is_even, %ep_v291_row1_lo, %ep_v291_row3_lo : f32
        %ep_v291_store_b_hi = arith.select %ep_is_even, %ep_v291_row1_hi, %ep_v291_row3_hi : f32
        %ep_v291_pair_a_0 = vector.broadcast %ep_v291_store_a_lo : f32 to vector<2xf32>
        %ep_v291_pair_a   = vector.insert %ep_v291_store_a_hi, %ep_v291_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v291_store_a  = arith.truncf %ep_v291_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v291_pair_b_0 = vector.broadcast %ep_v291_store_b_lo : f32 to vector<2xf32>
        %ep_v291_pair_b   = vector.insert %ep_v291_store_b_hi, %ep_v291_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v291_store_b  = arith.truncf %ep_v291_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v291_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col4_adj overflow<nsw> : index
        %ep_v291_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col4_adj overflow<nsw> : index
        vector.store %ep_v291_store_a, %ep_buffer[%ep_v291_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v291_store_b, %ep_buffer[%ep_v291_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 13: %293 → rb1, col5
        %ep_v293_row0 = vector.extract %293[0] : f32 from vector<4xf32>
        %ep_v293_row1 = vector.extract %293[1] : f32 from vector<4xf32>
        %ep_v293_row2 = vector.extract %293[2] : f32 from vector<4xf32>
        %ep_v293_row3 = vector.extract %293[3] : f32 from vector<4xf32>
        %ep_v293_row0_nbr, %ep_v293_row0_valid = gpu.shuffle xor %ep_v293_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v293_row1_nbr, %ep_v293_row1_valid = gpu.shuffle xor %ep_v293_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v293_row2_nbr, %ep_v293_row2_valid = gpu.shuffle xor %ep_v293_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v293_row3_nbr, %ep_v293_row3_valid = gpu.shuffle xor %ep_v293_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v293_row0_lo = arith.select %ep_is_even, %ep_v293_row0,     %ep_v293_row0_nbr : f32
        %ep_v293_row0_hi = arith.select %ep_is_even, %ep_v293_row0_nbr, %ep_v293_row0     : f32
        %ep_v293_row1_lo = arith.select %ep_is_even, %ep_v293_row1,     %ep_v293_row1_nbr : f32
        %ep_v293_row1_hi = arith.select %ep_is_even, %ep_v293_row1_nbr, %ep_v293_row1     : f32
        %ep_v293_row2_lo = arith.select %ep_is_even, %ep_v293_row2,     %ep_v293_row2_nbr : f32
        %ep_v293_row2_hi = arith.select %ep_is_even, %ep_v293_row2_nbr, %ep_v293_row2     : f32
        %ep_v293_row3_lo = arith.select %ep_is_even, %ep_v293_row3,     %ep_v293_row3_nbr : f32
        %ep_v293_row3_hi = arith.select %ep_is_even, %ep_v293_row3_nbr, %ep_v293_row3     : f32
        %ep_v293_store_a_lo = arith.select %ep_is_even, %ep_v293_row0_lo, %ep_v293_row2_lo : f32
        %ep_v293_store_a_hi = arith.select %ep_is_even, %ep_v293_row0_hi, %ep_v293_row2_hi : f32
        %ep_v293_store_b_lo = arith.select %ep_is_even, %ep_v293_row1_lo, %ep_v293_row3_lo : f32
        %ep_v293_store_b_hi = arith.select %ep_is_even, %ep_v293_row1_hi, %ep_v293_row3_hi : f32
        %ep_v293_pair_a_0 = vector.broadcast %ep_v293_store_a_lo : f32 to vector<2xf32>
        %ep_v293_pair_a   = vector.insert %ep_v293_store_a_hi, %ep_v293_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v293_store_a  = arith.truncf %ep_v293_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v293_pair_b_0 = vector.broadcast %ep_v293_store_b_lo : f32 to vector<2xf32>
        %ep_v293_pair_b   = vector.insert %ep_v293_store_b_hi, %ep_v293_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v293_store_b  = arith.truncf %ep_v293_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v293_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col5_adj overflow<nsw> : index
        %ep_v293_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col5_adj overflow<nsw> : index
        vector.store %ep_v293_store_a, %ep_buffer[%ep_v293_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v293_store_b, %ep_buffer[%ep_v293_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 14: %295 → rb1, col6
        %ep_v295_row0 = vector.extract %295[0] : f32 from vector<4xf32>
        %ep_v295_row1 = vector.extract %295[1] : f32 from vector<4xf32>
        %ep_v295_row2 = vector.extract %295[2] : f32 from vector<4xf32>
        %ep_v295_row3 = vector.extract %295[3] : f32 from vector<4xf32>
        %ep_v295_row0_nbr, %ep_v295_row0_valid = gpu.shuffle xor %ep_v295_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v295_row1_nbr, %ep_v295_row1_valid = gpu.shuffle xor %ep_v295_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v295_row2_nbr, %ep_v295_row2_valid = gpu.shuffle xor %ep_v295_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v295_row3_nbr, %ep_v295_row3_valid = gpu.shuffle xor %ep_v295_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v295_row0_lo = arith.select %ep_is_even, %ep_v295_row0,     %ep_v295_row0_nbr : f32
        %ep_v295_row0_hi = arith.select %ep_is_even, %ep_v295_row0_nbr, %ep_v295_row0     : f32
        %ep_v295_row1_lo = arith.select %ep_is_even, %ep_v295_row1,     %ep_v295_row1_nbr : f32
        %ep_v295_row1_hi = arith.select %ep_is_even, %ep_v295_row1_nbr, %ep_v295_row1     : f32
        %ep_v295_row2_lo = arith.select %ep_is_even, %ep_v295_row2,     %ep_v295_row2_nbr : f32
        %ep_v295_row2_hi = arith.select %ep_is_even, %ep_v295_row2_nbr, %ep_v295_row2     : f32
        %ep_v295_row3_lo = arith.select %ep_is_even, %ep_v295_row3,     %ep_v295_row3_nbr : f32
        %ep_v295_row3_hi = arith.select %ep_is_even, %ep_v295_row3_nbr, %ep_v295_row3     : f32
        %ep_v295_store_a_lo = arith.select %ep_is_even, %ep_v295_row0_lo, %ep_v295_row2_lo : f32
        %ep_v295_store_a_hi = arith.select %ep_is_even, %ep_v295_row0_hi, %ep_v295_row2_hi : f32
        %ep_v295_store_b_lo = arith.select %ep_is_even, %ep_v295_row1_lo, %ep_v295_row3_lo : f32
        %ep_v295_store_b_hi = arith.select %ep_is_even, %ep_v295_row1_hi, %ep_v295_row3_hi : f32
        %ep_v295_pair_a_0 = vector.broadcast %ep_v295_store_a_lo : f32 to vector<2xf32>
        %ep_v295_pair_a   = vector.insert %ep_v295_store_a_hi, %ep_v295_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v295_store_a  = arith.truncf %ep_v295_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v295_pair_b_0 = vector.broadcast %ep_v295_store_b_lo : f32 to vector<2xf32>
        %ep_v295_pair_b   = vector.insert %ep_v295_store_b_hi, %ep_v295_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v295_store_b  = arith.truncf %ep_v295_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v295_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col6_adj overflow<nsw> : index
        %ep_v295_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col6_adj overflow<nsw> : index
        vector.store %ep_v295_store_a, %ep_buffer[%ep_v295_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v295_store_b, %ep_buffer[%ep_v295_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 15: %297 → rb1, col7
        %ep_v297_row0 = vector.extract %297[0] : f32 from vector<4xf32>
        %ep_v297_row1 = vector.extract %297[1] : f32 from vector<4xf32>
        %ep_v297_row2 = vector.extract %297[2] : f32 from vector<4xf32>
        %ep_v297_row3 = vector.extract %297[3] : f32 from vector<4xf32>
        %ep_v297_row0_nbr, %ep_v297_row0_valid = gpu.shuffle xor %ep_v297_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v297_row1_nbr, %ep_v297_row1_valid = gpu.shuffle xor %ep_v297_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v297_row2_nbr, %ep_v297_row2_valid = gpu.shuffle xor %ep_v297_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v297_row3_nbr, %ep_v297_row3_valid = gpu.shuffle xor %ep_v297_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v297_row0_lo = arith.select %ep_is_even, %ep_v297_row0,     %ep_v297_row0_nbr : f32
        %ep_v297_row0_hi = arith.select %ep_is_even, %ep_v297_row0_nbr, %ep_v297_row0     : f32
        %ep_v297_row1_lo = arith.select %ep_is_even, %ep_v297_row1,     %ep_v297_row1_nbr : f32
        %ep_v297_row1_hi = arith.select %ep_is_even, %ep_v297_row1_nbr, %ep_v297_row1     : f32
        %ep_v297_row2_lo = arith.select %ep_is_even, %ep_v297_row2,     %ep_v297_row2_nbr : f32
        %ep_v297_row2_hi = arith.select %ep_is_even, %ep_v297_row2_nbr, %ep_v297_row2     : f32
        %ep_v297_row3_lo = arith.select %ep_is_even, %ep_v297_row3,     %ep_v297_row3_nbr : f32
        %ep_v297_row3_hi = arith.select %ep_is_even, %ep_v297_row3_nbr, %ep_v297_row3     : f32
        %ep_v297_store_a_lo = arith.select %ep_is_even, %ep_v297_row0_lo, %ep_v297_row2_lo : f32
        %ep_v297_store_a_hi = arith.select %ep_is_even, %ep_v297_row0_hi, %ep_v297_row2_hi : f32
        %ep_v297_store_b_lo = arith.select %ep_is_even, %ep_v297_row1_lo, %ep_v297_row3_lo : f32
        %ep_v297_store_b_hi = arith.select %ep_is_even, %ep_v297_row1_hi, %ep_v297_row3_hi : f32
        %ep_v297_pair_a_0 = vector.broadcast %ep_v297_store_a_lo : f32 to vector<2xf32>
        %ep_v297_pair_a   = vector.insert %ep_v297_store_a_hi, %ep_v297_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v297_store_a  = arith.truncf %ep_v297_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v297_pair_b_0 = vector.broadcast %ep_v297_store_b_lo : f32 to vector<2xf32>
        %ep_v297_pair_b   = vector.insert %ep_v297_store_b_hi, %ep_v297_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v297_store_b  = arith.truncf %ep_v297_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v297_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col7_adj overflow<nsw> : index
        %ep_v297_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col7_adj overflow<nsw> : index
        vector.store %ep_v297_store_a, %ep_buffer[%ep_v297_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v297_store_b, %ep_buffer[%ep_v297_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 16: %299 → rb2, col0
        %ep_v299_row0 = vector.extract %299[0] : f32 from vector<4xf32>
        %ep_v299_row1 = vector.extract %299[1] : f32 from vector<4xf32>
        %ep_v299_row2 = vector.extract %299[2] : f32 from vector<4xf32>
        %ep_v299_row3 = vector.extract %299[3] : f32 from vector<4xf32>
        %ep_v299_row0_nbr, %ep_v299_row0_valid = gpu.shuffle xor %ep_v299_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v299_row1_nbr, %ep_v299_row1_valid = gpu.shuffle xor %ep_v299_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v299_row2_nbr, %ep_v299_row2_valid = gpu.shuffle xor %ep_v299_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v299_row3_nbr, %ep_v299_row3_valid = gpu.shuffle xor %ep_v299_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v299_row0_lo = arith.select %ep_is_even, %ep_v299_row0,     %ep_v299_row0_nbr : f32
        %ep_v299_row0_hi = arith.select %ep_is_even, %ep_v299_row0_nbr, %ep_v299_row0     : f32
        %ep_v299_row1_lo = arith.select %ep_is_even, %ep_v299_row1,     %ep_v299_row1_nbr : f32
        %ep_v299_row1_hi = arith.select %ep_is_even, %ep_v299_row1_nbr, %ep_v299_row1     : f32
        %ep_v299_row2_lo = arith.select %ep_is_even, %ep_v299_row2,     %ep_v299_row2_nbr : f32
        %ep_v299_row2_hi = arith.select %ep_is_even, %ep_v299_row2_nbr, %ep_v299_row2     : f32
        %ep_v299_row3_lo = arith.select %ep_is_even, %ep_v299_row3,     %ep_v299_row3_nbr : f32
        %ep_v299_row3_hi = arith.select %ep_is_even, %ep_v299_row3_nbr, %ep_v299_row3     : f32
        %ep_v299_store_a_lo = arith.select %ep_is_even, %ep_v299_row0_lo, %ep_v299_row2_lo : f32
        %ep_v299_store_a_hi = arith.select %ep_is_even, %ep_v299_row0_hi, %ep_v299_row2_hi : f32
        %ep_v299_store_b_lo = arith.select %ep_is_even, %ep_v299_row1_lo, %ep_v299_row3_lo : f32
        %ep_v299_store_b_hi = arith.select %ep_is_even, %ep_v299_row1_hi, %ep_v299_row3_hi : f32
        %ep_v299_pair_a_0 = vector.broadcast %ep_v299_store_a_lo : f32 to vector<2xf32>
        %ep_v299_pair_a   = vector.insert %ep_v299_store_a_hi, %ep_v299_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v299_store_a  = arith.truncf %ep_v299_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v299_pair_b_0 = vector.broadcast %ep_v299_store_b_lo : f32 to vector<2xf32>
        %ep_v299_pair_b   = vector.insert %ep_v299_store_b_hi, %ep_v299_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v299_store_b  = arith.truncf %ep_v299_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v299_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col0_adj overflow<nsw> : index
        %ep_v299_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col0_adj overflow<nsw> : index
        vector.store %ep_v299_store_a, %ep_buffer[%ep_v299_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v299_store_b, %ep_buffer[%ep_v299_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 17: %301 → rb2, col1
        %ep_v301_row0 = vector.extract %301[0] : f32 from vector<4xf32>
        %ep_v301_row1 = vector.extract %301[1] : f32 from vector<4xf32>
        %ep_v301_row2 = vector.extract %301[2] : f32 from vector<4xf32>
        %ep_v301_row3 = vector.extract %301[3] : f32 from vector<4xf32>
        %ep_v301_row0_nbr, %ep_v301_row0_valid = gpu.shuffle xor %ep_v301_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v301_row1_nbr, %ep_v301_row1_valid = gpu.shuffle xor %ep_v301_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v301_row2_nbr, %ep_v301_row2_valid = gpu.shuffle xor %ep_v301_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v301_row3_nbr, %ep_v301_row3_valid = gpu.shuffle xor %ep_v301_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v301_row0_lo = arith.select %ep_is_even, %ep_v301_row0,     %ep_v301_row0_nbr : f32
        %ep_v301_row0_hi = arith.select %ep_is_even, %ep_v301_row0_nbr, %ep_v301_row0     : f32
        %ep_v301_row1_lo = arith.select %ep_is_even, %ep_v301_row1,     %ep_v301_row1_nbr : f32
        %ep_v301_row1_hi = arith.select %ep_is_even, %ep_v301_row1_nbr, %ep_v301_row1     : f32
        %ep_v301_row2_lo = arith.select %ep_is_even, %ep_v301_row2,     %ep_v301_row2_nbr : f32
        %ep_v301_row2_hi = arith.select %ep_is_even, %ep_v301_row2_nbr, %ep_v301_row2     : f32
        %ep_v301_row3_lo = arith.select %ep_is_even, %ep_v301_row3,     %ep_v301_row3_nbr : f32
        %ep_v301_row3_hi = arith.select %ep_is_even, %ep_v301_row3_nbr, %ep_v301_row3     : f32
        %ep_v301_store_a_lo = arith.select %ep_is_even, %ep_v301_row0_lo, %ep_v301_row2_lo : f32
        %ep_v301_store_a_hi = arith.select %ep_is_even, %ep_v301_row0_hi, %ep_v301_row2_hi : f32
        %ep_v301_store_b_lo = arith.select %ep_is_even, %ep_v301_row1_lo, %ep_v301_row3_lo : f32
        %ep_v301_store_b_hi = arith.select %ep_is_even, %ep_v301_row1_hi, %ep_v301_row3_hi : f32
        %ep_v301_pair_a_0 = vector.broadcast %ep_v301_store_a_lo : f32 to vector<2xf32>
        %ep_v301_pair_a   = vector.insert %ep_v301_store_a_hi, %ep_v301_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v301_store_a  = arith.truncf %ep_v301_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v301_pair_b_0 = vector.broadcast %ep_v301_store_b_lo : f32 to vector<2xf32>
        %ep_v301_pair_b   = vector.insert %ep_v301_store_b_hi, %ep_v301_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v301_store_b  = arith.truncf %ep_v301_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v301_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col1_adj overflow<nsw> : index
        %ep_v301_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col1_adj overflow<nsw> : index
        vector.store %ep_v301_store_a, %ep_buffer[%ep_v301_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v301_store_b, %ep_buffer[%ep_v301_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 18: %303 → rb2, col2
        %ep_v303_row0 = vector.extract %303[0] : f32 from vector<4xf32>
        %ep_v303_row1 = vector.extract %303[1] : f32 from vector<4xf32>
        %ep_v303_row2 = vector.extract %303[2] : f32 from vector<4xf32>
        %ep_v303_row3 = vector.extract %303[3] : f32 from vector<4xf32>
        %ep_v303_row0_nbr, %ep_v303_row0_valid = gpu.shuffle xor %ep_v303_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v303_row1_nbr, %ep_v303_row1_valid = gpu.shuffle xor %ep_v303_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v303_row2_nbr, %ep_v303_row2_valid = gpu.shuffle xor %ep_v303_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v303_row3_nbr, %ep_v303_row3_valid = gpu.shuffle xor %ep_v303_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v303_row0_lo = arith.select %ep_is_even, %ep_v303_row0,     %ep_v303_row0_nbr : f32
        %ep_v303_row0_hi = arith.select %ep_is_even, %ep_v303_row0_nbr, %ep_v303_row0     : f32
        %ep_v303_row1_lo = arith.select %ep_is_even, %ep_v303_row1,     %ep_v303_row1_nbr : f32
        %ep_v303_row1_hi = arith.select %ep_is_even, %ep_v303_row1_nbr, %ep_v303_row1     : f32
        %ep_v303_row2_lo = arith.select %ep_is_even, %ep_v303_row2,     %ep_v303_row2_nbr : f32
        %ep_v303_row2_hi = arith.select %ep_is_even, %ep_v303_row2_nbr, %ep_v303_row2     : f32
        %ep_v303_row3_lo = arith.select %ep_is_even, %ep_v303_row3,     %ep_v303_row3_nbr : f32
        %ep_v303_row3_hi = arith.select %ep_is_even, %ep_v303_row3_nbr, %ep_v303_row3     : f32
        %ep_v303_store_a_lo = arith.select %ep_is_even, %ep_v303_row0_lo, %ep_v303_row2_lo : f32
        %ep_v303_store_a_hi = arith.select %ep_is_even, %ep_v303_row0_hi, %ep_v303_row2_hi : f32
        %ep_v303_store_b_lo = arith.select %ep_is_even, %ep_v303_row1_lo, %ep_v303_row3_lo : f32
        %ep_v303_store_b_hi = arith.select %ep_is_even, %ep_v303_row1_hi, %ep_v303_row3_hi : f32
        %ep_v303_pair_a_0 = vector.broadcast %ep_v303_store_a_lo : f32 to vector<2xf32>
        %ep_v303_pair_a   = vector.insert %ep_v303_store_a_hi, %ep_v303_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v303_store_a  = arith.truncf %ep_v303_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v303_pair_b_0 = vector.broadcast %ep_v303_store_b_lo : f32 to vector<2xf32>
        %ep_v303_pair_b   = vector.insert %ep_v303_store_b_hi, %ep_v303_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v303_store_b  = arith.truncf %ep_v303_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v303_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col2_adj overflow<nsw> : index
        %ep_v303_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col2_adj overflow<nsw> : index
        vector.store %ep_v303_store_a, %ep_buffer[%ep_v303_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v303_store_b, %ep_buffer[%ep_v303_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 19: %305 → rb2, col3
        %ep_v305_row0 = vector.extract %305[0] : f32 from vector<4xf32>
        %ep_v305_row1 = vector.extract %305[1] : f32 from vector<4xf32>
        %ep_v305_row2 = vector.extract %305[2] : f32 from vector<4xf32>
        %ep_v305_row3 = vector.extract %305[3] : f32 from vector<4xf32>
        %ep_v305_row0_nbr, %ep_v305_row0_valid = gpu.shuffle xor %ep_v305_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v305_row1_nbr, %ep_v305_row1_valid = gpu.shuffle xor %ep_v305_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v305_row2_nbr, %ep_v305_row2_valid = gpu.shuffle xor %ep_v305_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v305_row3_nbr, %ep_v305_row3_valid = gpu.shuffle xor %ep_v305_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v305_row0_lo = arith.select %ep_is_even, %ep_v305_row0,     %ep_v305_row0_nbr : f32
        %ep_v305_row0_hi = arith.select %ep_is_even, %ep_v305_row0_nbr, %ep_v305_row0     : f32
        %ep_v305_row1_lo = arith.select %ep_is_even, %ep_v305_row1,     %ep_v305_row1_nbr : f32
        %ep_v305_row1_hi = arith.select %ep_is_even, %ep_v305_row1_nbr, %ep_v305_row1     : f32
        %ep_v305_row2_lo = arith.select %ep_is_even, %ep_v305_row2,     %ep_v305_row2_nbr : f32
        %ep_v305_row2_hi = arith.select %ep_is_even, %ep_v305_row2_nbr, %ep_v305_row2     : f32
        %ep_v305_row3_lo = arith.select %ep_is_even, %ep_v305_row3,     %ep_v305_row3_nbr : f32
        %ep_v305_row3_hi = arith.select %ep_is_even, %ep_v305_row3_nbr, %ep_v305_row3     : f32
        %ep_v305_store_a_lo = arith.select %ep_is_even, %ep_v305_row0_lo, %ep_v305_row2_lo : f32
        %ep_v305_store_a_hi = arith.select %ep_is_even, %ep_v305_row0_hi, %ep_v305_row2_hi : f32
        %ep_v305_store_b_lo = arith.select %ep_is_even, %ep_v305_row1_lo, %ep_v305_row3_lo : f32
        %ep_v305_store_b_hi = arith.select %ep_is_even, %ep_v305_row1_hi, %ep_v305_row3_hi : f32
        %ep_v305_pair_a_0 = vector.broadcast %ep_v305_store_a_lo : f32 to vector<2xf32>
        %ep_v305_pair_a   = vector.insert %ep_v305_store_a_hi, %ep_v305_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v305_store_a  = arith.truncf %ep_v305_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v305_pair_b_0 = vector.broadcast %ep_v305_store_b_lo : f32 to vector<2xf32>
        %ep_v305_pair_b   = vector.insert %ep_v305_store_b_hi, %ep_v305_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v305_store_b  = arith.truncf %ep_v305_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v305_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col3_adj overflow<nsw> : index
        %ep_v305_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col3_adj overflow<nsw> : index
        vector.store %ep_v305_store_a, %ep_buffer[%ep_v305_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v305_store_b, %ep_buffer[%ep_v305_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 20: %307 → rb2, col4
        %ep_v307_row0 = vector.extract %307[0] : f32 from vector<4xf32>
        %ep_v307_row1 = vector.extract %307[1] : f32 from vector<4xf32>
        %ep_v307_row2 = vector.extract %307[2] : f32 from vector<4xf32>
        %ep_v307_row3 = vector.extract %307[3] : f32 from vector<4xf32>
        %ep_v307_row0_nbr, %ep_v307_row0_valid = gpu.shuffle xor %ep_v307_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v307_row1_nbr, %ep_v307_row1_valid = gpu.shuffle xor %ep_v307_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v307_row2_nbr, %ep_v307_row2_valid = gpu.shuffle xor %ep_v307_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v307_row3_nbr, %ep_v307_row3_valid = gpu.shuffle xor %ep_v307_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v307_row0_lo = arith.select %ep_is_even, %ep_v307_row0,     %ep_v307_row0_nbr : f32
        %ep_v307_row0_hi = arith.select %ep_is_even, %ep_v307_row0_nbr, %ep_v307_row0     : f32
        %ep_v307_row1_lo = arith.select %ep_is_even, %ep_v307_row1,     %ep_v307_row1_nbr : f32
        %ep_v307_row1_hi = arith.select %ep_is_even, %ep_v307_row1_nbr, %ep_v307_row1     : f32
        %ep_v307_row2_lo = arith.select %ep_is_even, %ep_v307_row2,     %ep_v307_row2_nbr : f32
        %ep_v307_row2_hi = arith.select %ep_is_even, %ep_v307_row2_nbr, %ep_v307_row2     : f32
        %ep_v307_row3_lo = arith.select %ep_is_even, %ep_v307_row3,     %ep_v307_row3_nbr : f32
        %ep_v307_row3_hi = arith.select %ep_is_even, %ep_v307_row3_nbr, %ep_v307_row3     : f32
        %ep_v307_store_a_lo = arith.select %ep_is_even, %ep_v307_row0_lo, %ep_v307_row2_lo : f32
        %ep_v307_store_a_hi = arith.select %ep_is_even, %ep_v307_row0_hi, %ep_v307_row2_hi : f32
        %ep_v307_store_b_lo = arith.select %ep_is_even, %ep_v307_row1_lo, %ep_v307_row3_lo : f32
        %ep_v307_store_b_hi = arith.select %ep_is_even, %ep_v307_row1_hi, %ep_v307_row3_hi : f32
        %ep_v307_pair_a_0 = vector.broadcast %ep_v307_store_a_lo : f32 to vector<2xf32>
        %ep_v307_pair_a   = vector.insert %ep_v307_store_a_hi, %ep_v307_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v307_store_a  = arith.truncf %ep_v307_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v307_pair_b_0 = vector.broadcast %ep_v307_store_b_lo : f32 to vector<2xf32>
        %ep_v307_pair_b   = vector.insert %ep_v307_store_b_hi, %ep_v307_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v307_store_b  = arith.truncf %ep_v307_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v307_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col4_adj overflow<nsw> : index
        %ep_v307_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col4_adj overflow<nsw> : index
        vector.store %ep_v307_store_a, %ep_buffer[%ep_v307_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v307_store_b, %ep_buffer[%ep_v307_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 21: %309 → rb2, col5
        %ep_v309_row0 = vector.extract %309[0] : f32 from vector<4xf32>
        %ep_v309_row1 = vector.extract %309[1] : f32 from vector<4xf32>
        %ep_v309_row2 = vector.extract %309[2] : f32 from vector<4xf32>
        %ep_v309_row3 = vector.extract %309[3] : f32 from vector<4xf32>
        %ep_v309_row0_nbr, %ep_v309_row0_valid = gpu.shuffle xor %ep_v309_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v309_row1_nbr, %ep_v309_row1_valid = gpu.shuffle xor %ep_v309_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v309_row2_nbr, %ep_v309_row2_valid = gpu.shuffle xor %ep_v309_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v309_row3_nbr, %ep_v309_row3_valid = gpu.shuffle xor %ep_v309_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v309_row0_lo = arith.select %ep_is_even, %ep_v309_row0,     %ep_v309_row0_nbr : f32
        %ep_v309_row0_hi = arith.select %ep_is_even, %ep_v309_row0_nbr, %ep_v309_row0     : f32
        %ep_v309_row1_lo = arith.select %ep_is_even, %ep_v309_row1,     %ep_v309_row1_nbr : f32
        %ep_v309_row1_hi = arith.select %ep_is_even, %ep_v309_row1_nbr, %ep_v309_row1     : f32
        %ep_v309_row2_lo = arith.select %ep_is_even, %ep_v309_row2,     %ep_v309_row2_nbr : f32
        %ep_v309_row2_hi = arith.select %ep_is_even, %ep_v309_row2_nbr, %ep_v309_row2     : f32
        %ep_v309_row3_lo = arith.select %ep_is_even, %ep_v309_row3,     %ep_v309_row3_nbr : f32
        %ep_v309_row3_hi = arith.select %ep_is_even, %ep_v309_row3_nbr, %ep_v309_row3     : f32
        %ep_v309_store_a_lo = arith.select %ep_is_even, %ep_v309_row0_lo, %ep_v309_row2_lo : f32
        %ep_v309_store_a_hi = arith.select %ep_is_even, %ep_v309_row0_hi, %ep_v309_row2_hi : f32
        %ep_v309_store_b_lo = arith.select %ep_is_even, %ep_v309_row1_lo, %ep_v309_row3_lo : f32
        %ep_v309_store_b_hi = arith.select %ep_is_even, %ep_v309_row1_hi, %ep_v309_row3_hi : f32
        %ep_v309_pair_a_0 = vector.broadcast %ep_v309_store_a_lo : f32 to vector<2xf32>
        %ep_v309_pair_a   = vector.insert %ep_v309_store_a_hi, %ep_v309_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v309_store_a  = arith.truncf %ep_v309_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v309_pair_b_0 = vector.broadcast %ep_v309_store_b_lo : f32 to vector<2xf32>
        %ep_v309_pair_b   = vector.insert %ep_v309_store_b_hi, %ep_v309_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v309_store_b  = arith.truncf %ep_v309_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v309_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col5_adj overflow<nsw> : index
        %ep_v309_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col5_adj overflow<nsw> : index
        vector.store %ep_v309_store_a, %ep_buffer[%ep_v309_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v309_store_b, %ep_buffer[%ep_v309_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 22: %311 → rb2, col6
        %ep_v311_row0 = vector.extract %311[0] : f32 from vector<4xf32>
        %ep_v311_row1 = vector.extract %311[1] : f32 from vector<4xf32>
        %ep_v311_row2 = vector.extract %311[2] : f32 from vector<4xf32>
        %ep_v311_row3 = vector.extract %311[3] : f32 from vector<4xf32>
        %ep_v311_row0_nbr, %ep_v311_row0_valid = gpu.shuffle xor %ep_v311_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v311_row1_nbr, %ep_v311_row1_valid = gpu.shuffle xor %ep_v311_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v311_row2_nbr, %ep_v311_row2_valid = gpu.shuffle xor %ep_v311_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v311_row3_nbr, %ep_v311_row3_valid = gpu.shuffle xor %ep_v311_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v311_row0_lo = arith.select %ep_is_even, %ep_v311_row0,     %ep_v311_row0_nbr : f32
        %ep_v311_row0_hi = arith.select %ep_is_even, %ep_v311_row0_nbr, %ep_v311_row0     : f32
        %ep_v311_row1_lo = arith.select %ep_is_even, %ep_v311_row1,     %ep_v311_row1_nbr : f32
        %ep_v311_row1_hi = arith.select %ep_is_even, %ep_v311_row1_nbr, %ep_v311_row1     : f32
        %ep_v311_row2_lo = arith.select %ep_is_even, %ep_v311_row2,     %ep_v311_row2_nbr : f32
        %ep_v311_row2_hi = arith.select %ep_is_even, %ep_v311_row2_nbr, %ep_v311_row2     : f32
        %ep_v311_row3_lo = arith.select %ep_is_even, %ep_v311_row3,     %ep_v311_row3_nbr : f32
        %ep_v311_row3_hi = arith.select %ep_is_even, %ep_v311_row3_nbr, %ep_v311_row3     : f32
        %ep_v311_store_a_lo = arith.select %ep_is_even, %ep_v311_row0_lo, %ep_v311_row2_lo : f32
        %ep_v311_store_a_hi = arith.select %ep_is_even, %ep_v311_row0_hi, %ep_v311_row2_hi : f32
        %ep_v311_store_b_lo = arith.select %ep_is_even, %ep_v311_row1_lo, %ep_v311_row3_lo : f32
        %ep_v311_store_b_hi = arith.select %ep_is_even, %ep_v311_row1_hi, %ep_v311_row3_hi : f32
        %ep_v311_pair_a_0 = vector.broadcast %ep_v311_store_a_lo : f32 to vector<2xf32>
        %ep_v311_pair_a   = vector.insert %ep_v311_store_a_hi, %ep_v311_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v311_store_a  = arith.truncf %ep_v311_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v311_pair_b_0 = vector.broadcast %ep_v311_store_b_lo : f32 to vector<2xf32>
        %ep_v311_pair_b   = vector.insert %ep_v311_store_b_hi, %ep_v311_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v311_store_b  = arith.truncf %ep_v311_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v311_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col6_adj overflow<nsw> : index
        %ep_v311_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col6_adj overflow<nsw> : index
        vector.store %ep_v311_store_a, %ep_buffer[%ep_v311_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v311_store_b, %ep_buffer[%ep_v311_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 23: %313 → rb2, col7
        %ep_v313_row0 = vector.extract %313[0] : f32 from vector<4xf32>
        %ep_v313_row1 = vector.extract %313[1] : f32 from vector<4xf32>
        %ep_v313_row2 = vector.extract %313[2] : f32 from vector<4xf32>
        %ep_v313_row3 = vector.extract %313[3] : f32 from vector<4xf32>
        %ep_v313_row0_nbr, %ep_v313_row0_valid = gpu.shuffle xor %ep_v313_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v313_row1_nbr, %ep_v313_row1_valid = gpu.shuffle xor %ep_v313_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v313_row2_nbr, %ep_v313_row2_valid = gpu.shuffle xor %ep_v313_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v313_row3_nbr, %ep_v313_row3_valid = gpu.shuffle xor %ep_v313_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v313_row0_lo = arith.select %ep_is_even, %ep_v313_row0,     %ep_v313_row0_nbr : f32
        %ep_v313_row0_hi = arith.select %ep_is_even, %ep_v313_row0_nbr, %ep_v313_row0     : f32
        %ep_v313_row1_lo = arith.select %ep_is_even, %ep_v313_row1,     %ep_v313_row1_nbr : f32
        %ep_v313_row1_hi = arith.select %ep_is_even, %ep_v313_row1_nbr, %ep_v313_row1     : f32
        %ep_v313_row2_lo = arith.select %ep_is_even, %ep_v313_row2,     %ep_v313_row2_nbr : f32
        %ep_v313_row2_hi = arith.select %ep_is_even, %ep_v313_row2_nbr, %ep_v313_row2     : f32
        %ep_v313_row3_lo = arith.select %ep_is_even, %ep_v313_row3,     %ep_v313_row3_nbr : f32
        %ep_v313_row3_hi = arith.select %ep_is_even, %ep_v313_row3_nbr, %ep_v313_row3     : f32
        %ep_v313_store_a_lo = arith.select %ep_is_even, %ep_v313_row0_lo, %ep_v313_row2_lo : f32
        %ep_v313_store_a_hi = arith.select %ep_is_even, %ep_v313_row0_hi, %ep_v313_row2_hi : f32
        %ep_v313_store_b_lo = arith.select %ep_is_even, %ep_v313_row1_lo, %ep_v313_row3_lo : f32
        %ep_v313_store_b_hi = arith.select %ep_is_even, %ep_v313_row1_hi, %ep_v313_row3_hi : f32
        %ep_v313_pair_a_0 = vector.broadcast %ep_v313_store_a_lo : f32 to vector<2xf32>
        %ep_v313_pair_a   = vector.insert %ep_v313_store_a_hi, %ep_v313_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v313_store_a  = arith.truncf %ep_v313_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v313_pair_b_0 = vector.broadcast %ep_v313_store_b_lo : f32 to vector<2xf32>
        %ep_v313_pair_b   = vector.insert %ep_v313_store_b_hi, %ep_v313_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v313_store_b  = arith.truncf %ep_v313_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v313_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col7_adj overflow<nsw> : index
        %ep_v313_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col7_adj overflow<nsw> : index
        vector.store %ep_v313_store_a, %ep_buffer[%ep_v313_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v313_store_b, %ep_buffer[%ep_v313_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 24: %315 → rb3, col0
        %ep_v315_row0 = vector.extract %315[0] : f32 from vector<4xf32>
        %ep_v315_row1 = vector.extract %315[1] : f32 from vector<4xf32>
        %ep_v315_row2 = vector.extract %315[2] : f32 from vector<4xf32>
        %ep_v315_row3 = vector.extract %315[3] : f32 from vector<4xf32>
        %ep_v315_row0_nbr, %ep_v315_row0_valid = gpu.shuffle xor %ep_v315_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v315_row1_nbr, %ep_v315_row1_valid = gpu.shuffle xor %ep_v315_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v315_row2_nbr, %ep_v315_row2_valid = gpu.shuffle xor %ep_v315_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v315_row3_nbr, %ep_v315_row3_valid = gpu.shuffle xor %ep_v315_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v315_row0_lo = arith.select %ep_is_even, %ep_v315_row0,     %ep_v315_row0_nbr : f32
        %ep_v315_row0_hi = arith.select %ep_is_even, %ep_v315_row0_nbr, %ep_v315_row0     : f32
        %ep_v315_row1_lo = arith.select %ep_is_even, %ep_v315_row1,     %ep_v315_row1_nbr : f32
        %ep_v315_row1_hi = arith.select %ep_is_even, %ep_v315_row1_nbr, %ep_v315_row1     : f32
        %ep_v315_row2_lo = arith.select %ep_is_even, %ep_v315_row2,     %ep_v315_row2_nbr : f32
        %ep_v315_row2_hi = arith.select %ep_is_even, %ep_v315_row2_nbr, %ep_v315_row2     : f32
        %ep_v315_row3_lo = arith.select %ep_is_even, %ep_v315_row3,     %ep_v315_row3_nbr : f32
        %ep_v315_row3_hi = arith.select %ep_is_even, %ep_v315_row3_nbr, %ep_v315_row3     : f32
        %ep_v315_store_a_lo = arith.select %ep_is_even, %ep_v315_row0_lo, %ep_v315_row2_lo : f32
        %ep_v315_store_a_hi = arith.select %ep_is_even, %ep_v315_row0_hi, %ep_v315_row2_hi : f32
        %ep_v315_store_b_lo = arith.select %ep_is_even, %ep_v315_row1_lo, %ep_v315_row3_lo : f32
        %ep_v315_store_b_hi = arith.select %ep_is_even, %ep_v315_row1_hi, %ep_v315_row3_hi : f32
        %ep_v315_pair_a_0 = vector.broadcast %ep_v315_store_a_lo : f32 to vector<2xf32>
        %ep_v315_pair_a   = vector.insert %ep_v315_store_a_hi, %ep_v315_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v315_store_a  = arith.truncf %ep_v315_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v315_pair_b_0 = vector.broadcast %ep_v315_store_b_lo : f32 to vector<2xf32>
        %ep_v315_pair_b   = vector.insert %ep_v315_store_b_hi, %ep_v315_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v315_store_b  = arith.truncf %ep_v315_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v315_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col0_adj overflow<nsw> : index
        %ep_v315_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col0_adj overflow<nsw> : index
        vector.store %ep_v315_store_a, %ep_buffer[%ep_v315_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v315_store_b, %ep_buffer[%ep_v315_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 25: %317 → rb3, col1
        %ep_v317_row0 = vector.extract %317[0] : f32 from vector<4xf32>
        %ep_v317_row1 = vector.extract %317[1] : f32 from vector<4xf32>
        %ep_v317_row2 = vector.extract %317[2] : f32 from vector<4xf32>
        %ep_v317_row3 = vector.extract %317[3] : f32 from vector<4xf32>
        %ep_v317_row0_nbr, %ep_v317_row0_valid = gpu.shuffle xor %ep_v317_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v317_row1_nbr, %ep_v317_row1_valid = gpu.shuffle xor %ep_v317_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v317_row2_nbr, %ep_v317_row2_valid = gpu.shuffle xor %ep_v317_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v317_row3_nbr, %ep_v317_row3_valid = gpu.shuffle xor %ep_v317_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v317_row0_lo = arith.select %ep_is_even, %ep_v317_row0,     %ep_v317_row0_nbr : f32
        %ep_v317_row0_hi = arith.select %ep_is_even, %ep_v317_row0_nbr, %ep_v317_row0     : f32
        %ep_v317_row1_lo = arith.select %ep_is_even, %ep_v317_row1,     %ep_v317_row1_nbr : f32
        %ep_v317_row1_hi = arith.select %ep_is_even, %ep_v317_row1_nbr, %ep_v317_row1     : f32
        %ep_v317_row2_lo = arith.select %ep_is_even, %ep_v317_row2,     %ep_v317_row2_nbr : f32
        %ep_v317_row2_hi = arith.select %ep_is_even, %ep_v317_row2_nbr, %ep_v317_row2     : f32
        %ep_v317_row3_lo = arith.select %ep_is_even, %ep_v317_row3,     %ep_v317_row3_nbr : f32
        %ep_v317_row3_hi = arith.select %ep_is_even, %ep_v317_row3_nbr, %ep_v317_row3     : f32
        %ep_v317_store_a_lo = arith.select %ep_is_even, %ep_v317_row0_lo, %ep_v317_row2_lo : f32
        %ep_v317_store_a_hi = arith.select %ep_is_even, %ep_v317_row0_hi, %ep_v317_row2_hi : f32
        %ep_v317_store_b_lo = arith.select %ep_is_even, %ep_v317_row1_lo, %ep_v317_row3_lo : f32
        %ep_v317_store_b_hi = arith.select %ep_is_even, %ep_v317_row1_hi, %ep_v317_row3_hi : f32
        %ep_v317_pair_a_0 = vector.broadcast %ep_v317_store_a_lo : f32 to vector<2xf32>
        %ep_v317_pair_a   = vector.insert %ep_v317_store_a_hi, %ep_v317_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v317_store_a  = arith.truncf %ep_v317_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v317_pair_b_0 = vector.broadcast %ep_v317_store_b_lo : f32 to vector<2xf32>
        %ep_v317_pair_b   = vector.insert %ep_v317_store_b_hi, %ep_v317_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v317_store_b  = arith.truncf %ep_v317_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v317_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col1_adj overflow<nsw> : index
        %ep_v317_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col1_adj overflow<nsw> : index
        vector.store %ep_v317_store_a, %ep_buffer[%ep_v317_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v317_store_b, %ep_buffer[%ep_v317_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 26: %319 → rb3, col2
        %ep_v319_row0 = vector.extract %319[0] : f32 from vector<4xf32>
        %ep_v319_row1 = vector.extract %319[1] : f32 from vector<4xf32>
        %ep_v319_row2 = vector.extract %319[2] : f32 from vector<4xf32>
        %ep_v319_row3 = vector.extract %319[3] : f32 from vector<4xf32>
        %ep_v319_row0_nbr, %ep_v319_row0_valid = gpu.shuffle xor %ep_v319_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v319_row1_nbr, %ep_v319_row1_valid = gpu.shuffle xor %ep_v319_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v319_row2_nbr, %ep_v319_row2_valid = gpu.shuffle xor %ep_v319_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v319_row3_nbr, %ep_v319_row3_valid = gpu.shuffle xor %ep_v319_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v319_row0_lo = arith.select %ep_is_even, %ep_v319_row0,     %ep_v319_row0_nbr : f32
        %ep_v319_row0_hi = arith.select %ep_is_even, %ep_v319_row0_nbr, %ep_v319_row0     : f32
        %ep_v319_row1_lo = arith.select %ep_is_even, %ep_v319_row1,     %ep_v319_row1_nbr : f32
        %ep_v319_row1_hi = arith.select %ep_is_even, %ep_v319_row1_nbr, %ep_v319_row1     : f32
        %ep_v319_row2_lo = arith.select %ep_is_even, %ep_v319_row2,     %ep_v319_row2_nbr : f32
        %ep_v319_row2_hi = arith.select %ep_is_even, %ep_v319_row2_nbr, %ep_v319_row2     : f32
        %ep_v319_row3_lo = arith.select %ep_is_even, %ep_v319_row3,     %ep_v319_row3_nbr : f32
        %ep_v319_row3_hi = arith.select %ep_is_even, %ep_v319_row3_nbr, %ep_v319_row3     : f32
        %ep_v319_store_a_lo = arith.select %ep_is_even, %ep_v319_row0_lo, %ep_v319_row2_lo : f32
        %ep_v319_store_a_hi = arith.select %ep_is_even, %ep_v319_row0_hi, %ep_v319_row2_hi : f32
        %ep_v319_store_b_lo = arith.select %ep_is_even, %ep_v319_row1_lo, %ep_v319_row3_lo : f32
        %ep_v319_store_b_hi = arith.select %ep_is_even, %ep_v319_row1_hi, %ep_v319_row3_hi : f32
        %ep_v319_pair_a_0 = vector.broadcast %ep_v319_store_a_lo : f32 to vector<2xf32>
        %ep_v319_pair_a   = vector.insert %ep_v319_store_a_hi, %ep_v319_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v319_store_a  = arith.truncf %ep_v319_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v319_pair_b_0 = vector.broadcast %ep_v319_store_b_lo : f32 to vector<2xf32>
        %ep_v319_pair_b   = vector.insert %ep_v319_store_b_hi, %ep_v319_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v319_store_b  = arith.truncf %ep_v319_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v319_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col2_adj overflow<nsw> : index
        %ep_v319_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col2_adj overflow<nsw> : index
        vector.store %ep_v319_store_a, %ep_buffer[%ep_v319_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v319_store_b, %ep_buffer[%ep_v319_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 27: %321 → rb3, col3
        %ep_v321_row0 = vector.extract %321[0] : f32 from vector<4xf32>
        %ep_v321_row1 = vector.extract %321[1] : f32 from vector<4xf32>
        %ep_v321_row2 = vector.extract %321[2] : f32 from vector<4xf32>
        %ep_v321_row3 = vector.extract %321[3] : f32 from vector<4xf32>
        %ep_v321_row0_nbr, %ep_v321_row0_valid = gpu.shuffle xor %ep_v321_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v321_row1_nbr, %ep_v321_row1_valid = gpu.shuffle xor %ep_v321_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v321_row2_nbr, %ep_v321_row2_valid = gpu.shuffle xor %ep_v321_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v321_row3_nbr, %ep_v321_row3_valid = gpu.shuffle xor %ep_v321_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v321_row0_lo = arith.select %ep_is_even, %ep_v321_row0,     %ep_v321_row0_nbr : f32
        %ep_v321_row0_hi = arith.select %ep_is_even, %ep_v321_row0_nbr, %ep_v321_row0     : f32
        %ep_v321_row1_lo = arith.select %ep_is_even, %ep_v321_row1,     %ep_v321_row1_nbr : f32
        %ep_v321_row1_hi = arith.select %ep_is_even, %ep_v321_row1_nbr, %ep_v321_row1     : f32
        %ep_v321_row2_lo = arith.select %ep_is_even, %ep_v321_row2,     %ep_v321_row2_nbr : f32
        %ep_v321_row2_hi = arith.select %ep_is_even, %ep_v321_row2_nbr, %ep_v321_row2     : f32
        %ep_v321_row3_lo = arith.select %ep_is_even, %ep_v321_row3,     %ep_v321_row3_nbr : f32
        %ep_v321_row3_hi = arith.select %ep_is_even, %ep_v321_row3_nbr, %ep_v321_row3     : f32
        %ep_v321_store_a_lo = arith.select %ep_is_even, %ep_v321_row0_lo, %ep_v321_row2_lo : f32
        %ep_v321_store_a_hi = arith.select %ep_is_even, %ep_v321_row0_hi, %ep_v321_row2_hi : f32
        %ep_v321_store_b_lo = arith.select %ep_is_even, %ep_v321_row1_lo, %ep_v321_row3_lo : f32
        %ep_v321_store_b_hi = arith.select %ep_is_even, %ep_v321_row1_hi, %ep_v321_row3_hi : f32
        %ep_v321_pair_a_0 = vector.broadcast %ep_v321_store_a_lo : f32 to vector<2xf32>
        %ep_v321_pair_a   = vector.insert %ep_v321_store_a_hi, %ep_v321_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v321_store_a  = arith.truncf %ep_v321_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v321_pair_b_0 = vector.broadcast %ep_v321_store_b_lo : f32 to vector<2xf32>
        %ep_v321_pair_b   = vector.insert %ep_v321_store_b_hi, %ep_v321_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v321_store_b  = arith.truncf %ep_v321_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v321_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col3_adj overflow<nsw> : index
        %ep_v321_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col3_adj overflow<nsw> : index
        vector.store %ep_v321_store_a, %ep_buffer[%ep_v321_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v321_store_b, %ep_buffer[%ep_v321_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 28: %323 → rb3, col4
        %ep_v323_row0 = vector.extract %323[0] : f32 from vector<4xf32>
        %ep_v323_row1 = vector.extract %323[1] : f32 from vector<4xf32>
        %ep_v323_row2 = vector.extract %323[2] : f32 from vector<4xf32>
        %ep_v323_row3 = vector.extract %323[3] : f32 from vector<4xf32>
        %ep_v323_row0_nbr, %ep_v323_row0_valid = gpu.shuffle xor %ep_v323_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v323_row1_nbr, %ep_v323_row1_valid = gpu.shuffle xor %ep_v323_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v323_row2_nbr, %ep_v323_row2_valid = gpu.shuffle xor %ep_v323_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v323_row3_nbr, %ep_v323_row3_valid = gpu.shuffle xor %ep_v323_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v323_row0_lo = arith.select %ep_is_even, %ep_v323_row0,     %ep_v323_row0_nbr : f32
        %ep_v323_row0_hi = arith.select %ep_is_even, %ep_v323_row0_nbr, %ep_v323_row0     : f32
        %ep_v323_row1_lo = arith.select %ep_is_even, %ep_v323_row1,     %ep_v323_row1_nbr : f32
        %ep_v323_row1_hi = arith.select %ep_is_even, %ep_v323_row1_nbr, %ep_v323_row1     : f32
        %ep_v323_row2_lo = arith.select %ep_is_even, %ep_v323_row2,     %ep_v323_row2_nbr : f32
        %ep_v323_row2_hi = arith.select %ep_is_even, %ep_v323_row2_nbr, %ep_v323_row2     : f32
        %ep_v323_row3_lo = arith.select %ep_is_even, %ep_v323_row3,     %ep_v323_row3_nbr : f32
        %ep_v323_row3_hi = arith.select %ep_is_even, %ep_v323_row3_nbr, %ep_v323_row3     : f32
        %ep_v323_store_a_lo = arith.select %ep_is_even, %ep_v323_row0_lo, %ep_v323_row2_lo : f32
        %ep_v323_store_a_hi = arith.select %ep_is_even, %ep_v323_row0_hi, %ep_v323_row2_hi : f32
        %ep_v323_store_b_lo = arith.select %ep_is_even, %ep_v323_row1_lo, %ep_v323_row3_lo : f32
        %ep_v323_store_b_hi = arith.select %ep_is_even, %ep_v323_row1_hi, %ep_v323_row3_hi : f32
        %ep_v323_pair_a_0 = vector.broadcast %ep_v323_store_a_lo : f32 to vector<2xf32>
        %ep_v323_pair_a   = vector.insert %ep_v323_store_a_hi, %ep_v323_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v323_store_a  = arith.truncf %ep_v323_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v323_pair_b_0 = vector.broadcast %ep_v323_store_b_lo : f32 to vector<2xf32>
        %ep_v323_pair_b   = vector.insert %ep_v323_store_b_hi, %ep_v323_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v323_store_b  = arith.truncf %ep_v323_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v323_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col4_adj overflow<nsw> : index
        %ep_v323_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col4_adj overflow<nsw> : index
        vector.store %ep_v323_store_a, %ep_buffer[%ep_v323_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v323_store_b, %ep_buffer[%ep_v323_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 29: %325 → rb3, col5
        %ep_v325_row0 = vector.extract %325[0] : f32 from vector<4xf32>
        %ep_v325_row1 = vector.extract %325[1] : f32 from vector<4xf32>
        %ep_v325_row2 = vector.extract %325[2] : f32 from vector<4xf32>
        %ep_v325_row3 = vector.extract %325[3] : f32 from vector<4xf32>
        %ep_v325_row0_nbr, %ep_v325_row0_valid = gpu.shuffle xor %ep_v325_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v325_row1_nbr, %ep_v325_row1_valid = gpu.shuffle xor %ep_v325_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v325_row2_nbr, %ep_v325_row2_valid = gpu.shuffle xor %ep_v325_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v325_row3_nbr, %ep_v325_row3_valid = gpu.shuffle xor %ep_v325_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v325_row0_lo = arith.select %ep_is_even, %ep_v325_row0,     %ep_v325_row0_nbr : f32
        %ep_v325_row0_hi = arith.select %ep_is_even, %ep_v325_row0_nbr, %ep_v325_row0     : f32
        %ep_v325_row1_lo = arith.select %ep_is_even, %ep_v325_row1,     %ep_v325_row1_nbr : f32
        %ep_v325_row1_hi = arith.select %ep_is_even, %ep_v325_row1_nbr, %ep_v325_row1     : f32
        %ep_v325_row2_lo = arith.select %ep_is_even, %ep_v325_row2,     %ep_v325_row2_nbr : f32
        %ep_v325_row2_hi = arith.select %ep_is_even, %ep_v325_row2_nbr, %ep_v325_row2     : f32
        %ep_v325_row3_lo = arith.select %ep_is_even, %ep_v325_row3,     %ep_v325_row3_nbr : f32
        %ep_v325_row3_hi = arith.select %ep_is_even, %ep_v325_row3_nbr, %ep_v325_row3     : f32
        %ep_v325_store_a_lo = arith.select %ep_is_even, %ep_v325_row0_lo, %ep_v325_row2_lo : f32
        %ep_v325_store_a_hi = arith.select %ep_is_even, %ep_v325_row0_hi, %ep_v325_row2_hi : f32
        %ep_v325_store_b_lo = arith.select %ep_is_even, %ep_v325_row1_lo, %ep_v325_row3_lo : f32
        %ep_v325_store_b_hi = arith.select %ep_is_even, %ep_v325_row1_hi, %ep_v325_row3_hi : f32
        %ep_v325_pair_a_0 = vector.broadcast %ep_v325_store_a_lo : f32 to vector<2xf32>
        %ep_v325_pair_a   = vector.insert %ep_v325_store_a_hi, %ep_v325_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v325_store_a  = arith.truncf %ep_v325_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v325_pair_b_0 = vector.broadcast %ep_v325_store_b_lo : f32 to vector<2xf32>
        %ep_v325_pair_b   = vector.insert %ep_v325_store_b_hi, %ep_v325_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v325_store_b  = arith.truncf %ep_v325_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v325_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col5_adj overflow<nsw> : index
        %ep_v325_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col5_adj overflow<nsw> : index
        vector.store %ep_v325_store_a, %ep_buffer[%ep_v325_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v325_store_b, %ep_buffer[%ep_v325_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 30: %327 → rb3, col6
        %ep_v327_row0 = vector.extract %327[0] : f32 from vector<4xf32>
        %ep_v327_row1 = vector.extract %327[1] : f32 from vector<4xf32>
        %ep_v327_row2 = vector.extract %327[2] : f32 from vector<4xf32>
        %ep_v327_row3 = vector.extract %327[3] : f32 from vector<4xf32>
        %ep_v327_row0_nbr, %ep_v327_row0_valid = gpu.shuffle xor %ep_v327_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v327_row1_nbr, %ep_v327_row1_valid = gpu.shuffle xor %ep_v327_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v327_row2_nbr, %ep_v327_row2_valid = gpu.shuffle xor %ep_v327_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v327_row3_nbr, %ep_v327_row3_valid = gpu.shuffle xor %ep_v327_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v327_row0_lo = arith.select %ep_is_even, %ep_v327_row0,     %ep_v327_row0_nbr : f32
        %ep_v327_row0_hi = arith.select %ep_is_even, %ep_v327_row0_nbr, %ep_v327_row0     : f32
        %ep_v327_row1_lo = arith.select %ep_is_even, %ep_v327_row1,     %ep_v327_row1_nbr : f32
        %ep_v327_row1_hi = arith.select %ep_is_even, %ep_v327_row1_nbr, %ep_v327_row1     : f32
        %ep_v327_row2_lo = arith.select %ep_is_even, %ep_v327_row2,     %ep_v327_row2_nbr : f32
        %ep_v327_row2_hi = arith.select %ep_is_even, %ep_v327_row2_nbr, %ep_v327_row2     : f32
        %ep_v327_row3_lo = arith.select %ep_is_even, %ep_v327_row3,     %ep_v327_row3_nbr : f32
        %ep_v327_row3_hi = arith.select %ep_is_even, %ep_v327_row3_nbr, %ep_v327_row3     : f32
        %ep_v327_store_a_lo = arith.select %ep_is_even, %ep_v327_row0_lo, %ep_v327_row2_lo : f32
        %ep_v327_store_a_hi = arith.select %ep_is_even, %ep_v327_row0_hi, %ep_v327_row2_hi : f32
        %ep_v327_store_b_lo = arith.select %ep_is_even, %ep_v327_row1_lo, %ep_v327_row3_lo : f32
        %ep_v327_store_b_hi = arith.select %ep_is_even, %ep_v327_row1_hi, %ep_v327_row3_hi : f32
        %ep_v327_pair_a_0 = vector.broadcast %ep_v327_store_a_lo : f32 to vector<2xf32>
        %ep_v327_pair_a   = vector.insert %ep_v327_store_a_hi, %ep_v327_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v327_store_a  = arith.truncf %ep_v327_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v327_pair_b_0 = vector.broadcast %ep_v327_store_b_lo : f32 to vector<2xf32>
        %ep_v327_pair_b   = vector.insert %ep_v327_store_b_hi, %ep_v327_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v327_store_b  = arith.truncf %ep_v327_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v327_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col6_adj overflow<nsw> : index
        %ep_v327_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col6_adj overflow<nsw> : index
        vector.store %ep_v327_store_a, %ep_buffer[%ep_v327_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v327_store_b, %ep_buffer[%ep_v327_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 31: %329 → rb3, col7
        %ep_v329_row0 = vector.extract %329[0] : f32 from vector<4xf32>
        %ep_v329_row1 = vector.extract %329[1] : f32 from vector<4xf32>
        %ep_v329_row2 = vector.extract %329[2] : f32 from vector<4xf32>
        %ep_v329_row3 = vector.extract %329[3] : f32 from vector<4xf32>
        %ep_v329_row0_nbr, %ep_v329_row0_valid = gpu.shuffle xor %ep_v329_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v329_row1_nbr, %ep_v329_row1_valid = gpu.shuffle xor %ep_v329_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v329_row2_nbr, %ep_v329_row2_valid = gpu.shuffle xor %ep_v329_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v329_row3_nbr, %ep_v329_row3_valid = gpu.shuffle xor %ep_v329_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v329_row0_lo = arith.select %ep_is_even, %ep_v329_row0,     %ep_v329_row0_nbr : f32
        %ep_v329_row0_hi = arith.select %ep_is_even, %ep_v329_row0_nbr, %ep_v329_row0     : f32
        %ep_v329_row1_lo = arith.select %ep_is_even, %ep_v329_row1,     %ep_v329_row1_nbr : f32
        %ep_v329_row1_hi = arith.select %ep_is_even, %ep_v329_row1_nbr, %ep_v329_row1     : f32
        %ep_v329_row2_lo = arith.select %ep_is_even, %ep_v329_row2,     %ep_v329_row2_nbr : f32
        %ep_v329_row2_hi = arith.select %ep_is_even, %ep_v329_row2_nbr, %ep_v329_row2     : f32
        %ep_v329_row3_lo = arith.select %ep_is_even, %ep_v329_row3,     %ep_v329_row3_nbr : f32
        %ep_v329_row3_hi = arith.select %ep_is_even, %ep_v329_row3_nbr, %ep_v329_row3     : f32
        %ep_v329_store_a_lo = arith.select %ep_is_even, %ep_v329_row0_lo, %ep_v329_row2_lo : f32
        %ep_v329_store_a_hi = arith.select %ep_is_even, %ep_v329_row0_hi, %ep_v329_row2_hi : f32
        %ep_v329_store_b_lo = arith.select %ep_is_even, %ep_v329_row1_lo, %ep_v329_row3_lo : f32
        %ep_v329_store_b_hi = arith.select %ep_is_even, %ep_v329_row1_hi, %ep_v329_row3_hi : f32
        %ep_v329_pair_a_0 = vector.broadcast %ep_v329_store_a_lo : f32 to vector<2xf32>
        %ep_v329_pair_a   = vector.insert %ep_v329_store_a_hi, %ep_v329_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v329_store_a  = arith.truncf %ep_v329_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v329_pair_b_0 = vector.broadcast %ep_v329_store_b_lo : f32 to vector<2xf32>
        %ep_v329_pair_b   = vector.insert %ep_v329_store_b_hi, %ep_v329_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v329_store_b  = arith.truncf %ep_v329_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v329_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col7_adj overflow<nsw> : index
        %ep_v329_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col7_adj overflow<nsw> : index
        vector.store %ep_v329_store_a, %ep_buffer[%ep_v329_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v329_store_b, %ep_buffer[%ep_v329_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

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
