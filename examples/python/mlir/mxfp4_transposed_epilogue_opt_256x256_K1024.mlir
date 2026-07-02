#map = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
#map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
#map2 = affine_map<()[s0] -> (s0 mod 8)>
#map3 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8) floordiv 256) * 131072)>
#map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
#map5 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 256) * 131072 + 32768)>
#map6 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
#map7 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 128) floordiv 256) * 131072 + 65536)>
#map8 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
#map9 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 192) floordiv 256) * 131072 + 98304)>
#map10 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64)>
#map11 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 1024)>
#map12 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64)>
#map13 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 1024)>
#map14 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 2048)>
#map15 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 3072)>
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
#map31 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 131072 + s1 * 16384 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8) floordiv 256) * 131072 + 128)>
#map32 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 131072 + s1 * 16384 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 256) * 131072 + 32896)>
#map33 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 131072 + s1 * 16384 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 128) floordiv 256) * 131072 + 65664)>
#map34 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 131072 + s1 * 16384 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 192) floordiv 256) * 131072 + 98432)>
#map35 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4 + s2 * 256 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 256)>
#map36 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4 + s2 * 256 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 1280)>
#map37 = affine_map<()[s0, s1, s2, s3] -> (s0 * 8192 + s1 * 4096 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 - (s3 floordiv 16) * 64 + 256)>
#map38 = affine_map<()[s0, s1, s2, s3] -> (s0 * 8192 + s1 * 4096 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 - (s3 floordiv 16) * 64 + 1280)>
#map39 = affine_map<()[s0, s1, s2, s3] -> (s0 * 8192 + s1 * 4096 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 - (s3 floordiv 16) * 64 + 2304)>
#map40 = affine_map<()[s0, s1, s2, s3] -> (s0 * 8192 + s1 * 4096 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 - (s3 floordiv 16) * 64 + 3328)>
#map41 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8) floordiv 256) * 131072 + 384)>
#map42 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 256) * 131072 + 33152)>
#map43 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 128) floordiv 256) * 131072 + 65920)>
#map44 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 192) floordiv 256) * 131072 + 98688)>
#map45 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 768)>
#map46 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 1792)>
#map47 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 768)>
#map48 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 1792)>
#map49 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 2816)>
#map50 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 3840)>
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
#map_blk256 = affine_map<()[s0] -> (s0 * 256)>
#map_Nbase_ep = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 32) * 8)>
#map_Mbase_ep = affine_map<()[s0, s1] -> (s1 * 128 + s0 - (s0 floordiv 16) * 16)>
#map_lane_parity_ep = affine_map<()[s0] -> ((s0 mod 32) floordiv 16)>
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
        %c64_i14 = arith.constant 64 : i14
        %c1024_i14 = arith.constant 512 : i14
        %c2147483645_i64 = arith.constant 2147483645 : i64
        %c6 = arith.constant 2 : index
        %c65536_i64 = arith.constant 32768 : i64
        %c2 = arith.constant 2 : index
        %c1048576_i64 = arith.constant 524288 : i64
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
        %12 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c1048576_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
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
        %23 = amdgpu.fat_raw_buffer_cast %cast_5 validBytes(%c1048576_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %23[%22], %alloc_0[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %24 = affine.apply #map5()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%24], %alloc_0[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %25 = affine.apply #map7()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%25], %alloc_0[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %26 = affine.apply #map9()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%26], %alloc_0[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_6 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_7 = memref.cast %reinterpret_cast_6 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %27 = amdgpu.fat_raw_buffer_cast %cast_7 validBytes(%c65536_i64) cacheSwizzleStride(%c64_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %28 = affine.apply #map10()[%block_id_x, %thread_id_x]
        %29 = vector.load %27[%28] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %30 = affine.apply #map11()[%block_id_x, %thread_id_x]
        %31 = vector.load %27[%30] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %reinterpret_cast_8 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_9 = memref.cast %reinterpret_cast_8 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %32 = amdgpu.fat_raw_buffer_cast %cast_9 validBytes(%c65536_i64) cacheSwizzleStride(%c64_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
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
        %73:38 = scf.for %arg10 = %c0 to %c6 step %c2 iter_args(%arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %cst, %arg39 = %cst, %arg40 = %cst, %arg41 = %cst, %arg42 = %cst, %arg43 = %29, %arg44 = %31, %arg45 = %34, %arg46 = %36, %arg47 = %38, %arg48 = %40) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>) {
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
        %330 = arith.truncf %267 : vector<4xf32> to vector<4xbf16>
        %331 = arith.truncf %269 : vector<4xf32> to vector<4xbf16>
        %332 = arith.truncf %271 : vector<4xf32> to vector<4xbf16>
        %333 = arith.truncf %273 : vector<4xf32> to vector<4xbf16>
        %334 = arith.truncf %275 : vector<4xf32> to vector<4xbf16>
        %335 = arith.truncf %277 : vector<4xf32> to vector<4xbf16>
        %336 = arith.truncf %279 : vector<4xf32> to vector<4xbf16>
        %337 = arith.truncf %281 : vector<4xf32> to vector<4xbf16>
        %338 = arith.truncf %283 : vector<4xf32> to vector<4xbf16>
        %339 = arith.truncf %285 : vector<4xf32> to vector<4xbf16>
        %340 = arith.truncf %287 : vector<4xf32> to vector<4xbf16>
        %341 = arith.truncf %289 : vector<4xf32> to vector<4xbf16>
        %342 = arith.truncf %291 : vector<4xf32> to vector<4xbf16>
        %343 = arith.truncf %293 : vector<4xf32> to vector<4xbf16>
        %344 = arith.truncf %295 : vector<4xf32> to vector<4xbf16>
        %345 = arith.truncf %297 : vector<4xf32> to vector<4xbf16>
        %346 = arith.truncf %299 : vector<4xf32> to vector<4xbf16>
        %347 = arith.truncf %301 : vector<4xf32> to vector<4xbf16>
        %348 = arith.truncf %303 : vector<4xf32> to vector<4xbf16>
        %349 = arith.truncf %305 : vector<4xf32> to vector<4xbf16>
        %350 = arith.truncf %307 : vector<4xf32> to vector<4xbf16>
        %351 = arith.truncf %309 : vector<4xf32> to vector<4xbf16>
        %352 = arith.truncf %311 : vector<4xf32> to vector<4xbf16>
        %353 = arith.truncf %313 : vector<4xf32> to vector<4xbf16>
        %354 = arith.truncf %315 : vector<4xf32> to vector<4xbf16>
        %355 = arith.truncf %317 : vector<4xf32> to vector<4xbf16>
        %356 = arith.truncf %319 : vector<4xf32> to vector<4xbf16>
        %357 = arith.truncf %321 : vector<4xf32> to vector<4xbf16>
        %358 = arith.truncf %323 : vector<4xf32> to vector<4xbf16>
        %359 = arith.truncf %325 : vector<4xf32> to vector<4xbf16>
        %360 = arith.truncf %327 : vector<4xf32> to vector<4xbf16>
        %361 = arith.truncf %329 : vector<4xf32> to vector<4xbf16>

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
        // === N-group 0 (bn0): bf16 pairs %330/%331 .. %336/%337 ===
        // Pair 0: accumulators %330, %331 → bn0, M-row-pair 0
        %ep_p0_swa = amdgpu.permlane_swap %330 16 : vector<4xbf16>
        %ep_p0_swb = amdgpu.permlane_swap %331 16 : vector<4xbf16>
        %ep_p0_hi = vector.shuffle %330, %ep_p0_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p0_lo = vector.shuffle %ep_p0_swb, %331 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p0_vec = arith.select %ep_is_even, %ep_p0_hi, %ep_p0_lo : vector<8xbf16>
        %ep_p0_rsel = arith.select %ep_is_even, %ep_mt0, %ep_mt1 : index
        %ep_p0_addr = arith.addi %ep_bn0, %ep_p0_rsel overflow<nsw> : index
        vector.store %ep_p0_vec, %ep_buf[%ep_p0_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 1: accumulators %332, %333 → bn0, M-row-pair 1
        %ep_p1_swa = amdgpu.permlane_swap %332 16 : vector<4xbf16>
        %ep_p1_swb = amdgpu.permlane_swap %333 16 : vector<4xbf16>
        %ep_p1_hi = vector.shuffle %332, %ep_p1_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p1_lo = vector.shuffle %ep_p1_swb, %333 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p1_vec = arith.select %ep_is_even, %ep_p1_hi, %ep_p1_lo : vector<8xbf16>
        %ep_p1_rsel = arith.select %ep_is_even, %ep_mt2, %ep_mt3 : index
        %ep_p1_addr = arith.addi %ep_bn0, %ep_p1_rsel overflow<nsw> : index
        vector.store %ep_p1_vec, %ep_buf[%ep_p1_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 2: accumulators %334, %335 → bn0, M-row-pair 2
        %ep_p2_swa = amdgpu.permlane_swap %334 16 : vector<4xbf16>
        %ep_p2_swb = amdgpu.permlane_swap %335 16 : vector<4xbf16>
        %ep_p2_hi = vector.shuffle %334, %ep_p2_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p2_lo = vector.shuffle %ep_p2_swb, %335 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p2_vec = arith.select %ep_is_even, %ep_p2_hi, %ep_p2_lo : vector<8xbf16>
        %ep_p2_rsel = arith.select %ep_is_even, %ep_mt4, %ep_mt5 : index
        %ep_p2_addr = arith.addi %ep_bn0, %ep_p2_rsel overflow<nsw> : index
        vector.store %ep_p2_vec, %ep_buf[%ep_p2_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 3: accumulators %336, %337 → bn0, M-row-pair 3
        %ep_p3_swa = amdgpu.permlane_swap %336 16 : vector<4xbf16>
        %ep_p3_swb = amdgpu.permlane_swap %337 16 : vector<4xbf16>
        %ep_p3_hi = vector.shuffle %336, %ep_p3_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p3_lo = vector.shuffle %ep_p3_swb, %337 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p3_vec = arith.select %ep_is_even, %ep_p3_hi, %ep_p3_lo : vector<8xbf16>
        %ep_p3_rsel = arith.select %ep_is_even, %ep_mt6, %ep_mt7 : index
        %ep_p3_addr = arith.addi %ep_bn0, %ep_p3_rsel overflow<nsw> : index
        vector.store %ep_p3_vec, %ep_buf[%ep_p3_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // === N-group 1 (bn1): bf16 pairs %338/%339 .. %344/%345 ===
        // Pair 4: accumulators %338, %339 → bn1, M-row-pair 0
        %ep_p4_swa = amdgpu.permlane_swap %338 16 : vector<4xbf16>
        %ep_p4_swb = amdgpu.permlane_swap %339 16 : vector<4xbf16>
        %ep_p4_hi = vector.shuffle %338, %ep_p4_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p4_lo = vector.shuffle %ep_p4_swb, %339 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p4_vec = arith.select %ep_is_even, %ep_p4_hi, %ep_p4_lo : vector<8xbf16>
        %ep_p4_rsel = arith.select %ep_is_even, %ep_mt0, %ep_mt1 : index
        %ep_p4_addr = arith.addi %ep_bn1, %ep_p4_rsel overflow<nsw> : index
        vector.store %ep_p4_vec, %ep_buf[%ep_p4_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 5: accumulators %340, %341 → bn1, M-row-pair 1
        %ep_p5_swa = amdgpu.permlane_swap %340 16 : vector<4xbf16>
        %ep_p5_swb = amdgpu.permlane_swap %341 16 : vector<4xbf16>
        %ep_p5_hi = vector.shuffle %340, %ep_p5_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p5_lo = vector.shuffle %ep_p5_swb, %341 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p5_vec = arith.select %ep_is_even, %ep_p5_hi, %ep_p5_lo : vector<8xbf16>
        %ep_p5_rsel = arith.select %ep_is_even, %ep_mt2, %ep_mt3 : index
        %ep_p5_addr = arith.addi %ep_bn1, %ep_p5_rsel overflow<nsw> : index
        vector.store %ep_p5_vec, %ep_buf[%ep_p5_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 6: accumulators %342, %343 → bn1, M-row-pair 2
        %ep_p6_swa = amdgpu.permlane_swap %342 16 : vector<4xbf16>
        %ep_p6_swb = amdgpu.permlane_swap %343 16 : vector<4xbf16>
        %ep_p6_hi = vector.shuffle %342, %ep_p6_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p6_lo = vector.shuffle %ep_p6_swb, %343 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p6_vec = arith.select %ep_is_even, %ep_p6_hi, %ep_p6_lo : vector<8xbf16>
        %ep_p6_rsel = arith.select %ep_is_even, %ep_mt4, %ep_mt5 : index
        %ep_p6_addr = arith.addi %ep_bn1, %ep_p6_rsel overflow<nsw> : index
        vector.store %ep_p6_vec, %ep_buf[%ep_p6_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 7: accumulators %344, %345 → bn1, M-row-pair 3
        %ep_p7_swa = amdgpu.permlane_swap %344 16 : vector<4xbf16>
        %ep_p7_swb = amdgpu.permlane_swap %345 16 : vector<4xbf16>
        %ep_p7_hi = vector.shuffle %344, %ep_p7_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p7_lo = vector.shuffle %ep_p7_swb, %345 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p7_vec = arith.select %ep_is_even, %ep_p7_hi, %ep_p7_lo : vector<8xbf16>
        %ep_p7_rsel = arith.select %ep_is_even, %ep_mt6, %ep_mt7 : index
        %ep_p7_addr = arith.addi %ep_bn1, %ep_p7_rsel overflow<nsw> : index
        vector.store %ep_p7_vec, %ep_buf[%ep_p7_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // === N-group 2 (bn2): bf16 pairs %346/%347 .. %352/%353 ===
        // Pair 8: accumulators %346, %347 → bn2, M-row-pair 0
        %ep_p8_swa = amdgpu.permlane_swap %346 16 : vector<4xbf16>
        %ep_p8_swb = amdgpu.permlane_swap %347 16 : vector<4xbf16>
        %ep_p8_hi = vector.shuffle %346, %ep_p8_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p8_lo = vector.shuffle %ep_p8_swb, %347 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p8_vec = arith.select %ep_is_even, %ep_p8_hi, %ep_p8_lo : vector<8xbf16>
        %ep_p8_rsel = arith.select %ep_is_even, %ep_mt0, %ep_mt1 : index
        %ep_p8_addr = arith.addi %ep_bn2, %ep_p8_rsel overflow<nsw> : index
        vector.store %ep_p8_vec, %ep_buf[%ep_p8_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 9: accumulators %348, %349 → bn2, M-row-pair 1
        %ep_p9_swa = amdgpu.permlane_swap %348 16 : vector<4xbf16>
        %ep_p9_swb = amdgpu.permlane_swap %349 16 : vector<4xbf16>
        %ep_p9_hi = vector.shuffle %348, %ep_p9_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p9_lo = vector.shuffle %ep_p9_swb, %349 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p9_vec = arith.select %ep_is_even, %ep_p9_hi, %ep_p9_lo : vector<8xbf16>
        %ep_p9_rsel = arith.select %ep_is_even, %ep_mt2, %ep_mt3 : index
        %ep_p9_addr = arith.addi %ep_bn2, %ep_p9_rsel overflow<nsw> : index
        vector.store %ep_p9_vec, %ep_buf[%ep_p9_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 10: accumulators %350, %351 → bn2, M-row-pair 2
        %ep_p10_swa = amdgpu.permlane_swap %350 16 : vector<4xbf16>
        %ep_p10_swb = amdgpu.permlane_swap %351 16 : vector<4xbf16>
        %ep_p10_hi = vector.shuffle %350, %ep_p10_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p10_lo = vector.shuffle %ep_p10_swb, %351 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p10_vec = arith.select %ep_is_even, %ep_p10_hi, %ep_p10_lo : vector<8xbf16>
        %ep_p10_rsel = arith.select %ep_is_even, %ep_mt4, %ep_mt5 : index
        %ep_p10_addr = arith.addi %ep_bn2, %ep_p10_rsel overflow<nsw> : index
        vector.store %ep_p10_vec, %ep_buf[%ep_p10_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 11: accumulators %352, %353 → bn2, M-row-pair 3
        %ep_p11_swa = amdgpu.permlane_swap %352 16 : vector<4xbf16>
        %ep_p11_swb = amdgpu.permlane_swap %353 16 : vector<4xbf16>
        %ep_p11_hi = vector.shuffle %352, %ep_p11_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p11_lo = vector.shuffle %ep_p11_swb, %353 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p11_vec = arith.select %ep_is_even, %ep_p11_hi, %ep_p11_lo : vector<8xbf16>
        %ep_p11_rsel = arith.select %ep_is_even, %ep_mt6, %ep_mt7 : index
        %ep_p11_addr = arith.addi %ep_bn2, %ep_p11_rsel overflow<nsw> : index
        vector.store %ep_p11_vec, %ep_buf[%ep_p11_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // === N-group 3 (bn3): bf16 pairs %354/%355 .. %360/%361 [256x256 extension] ===
        // Pair 12: accumulators %354, %355 → bn3, M-row-pair 0
        %ep_p12_swa = amdgpu.permlane_swap %354 16 : vector<4xbf16>
        %ep_p12_swb = amdgpu.permlane_swap %355 16 : vector<4xbf16>
        %ep_p12_hi = vector.shuffle %354, %ep_p12_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p12_lo = vector.shuffle %ep_p12_swb, %355 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p12_vec = arith.select %ep_is_even, %ep_p12_hi, %ep_p12_lo : vector<8xbf16>
        %ep_p12_rsel = arith.select %ep_is_even, %ep_mt0, %ep_mt1 : index
        %ep_p12_addr = arith.addi %ep_bn3, %ep_p12_rsel overflow<nsw> : index
        vector.store %ep_p12_vec, %ep_buf[%ep_p12_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 13: accumulators %356, %357 → bn3, M-row-pair 1
        %ep_p13_swa = amdgpu.permlane_swap %356 16 : vector<4xbf16>
        %ep_p13_swb = amdgpu.permlane_swap %357 16 : vector<4xbf16>
        %ep_p13_hi = vector.shuffle %356, %ep_p13_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p13_lo = vector.shuffle %ep_p13_swb, %357 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p13_vec = arith.select %ep_is_even, %ep_p13_hi, %ep_p13_lo : vector<8xbf16>
        %ep_p13_rsel = arith.select %ep_is_even, %ep_mt2, %ep_mt3 : index
        %ep_p13_addr = arith.addi %ep_bn3, %ep_p13_rsel overflow<nsw> : index
        vector.store %ep_p13_vec, %ep_buf[%ep_p13_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 14: accumulators %358, %359 → bn3, M-row-pair 2
        %ep_p14_swa = amdgpu.permlane_swap %358 16 : vector<4xbf16>
        %ep_p14_swb = amdgpu.permlane_swap %359 16 : vector<4xbf16>
        %ep_p14_hi = vector.shuffle %358, %ep_p14_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p14_lo = vector.shuffle %ep_p14_swb, %359 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p14_vec = arith.select %ep_is_even, %ep_p14_hi, %ep_p14_lo : vector<8xbf16>
        %ep_p14_rsel = arith.select %ep_is_even, %ep_mt4, %ep_mt5 : index
        %ep_p14_addr = arith.addi %ep_bn3, %ep_p14_rsel overflow<nsw> : index
        vector.store %ep_p14_vec, %ep_buf[%ep_p14_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        // Pair 15: accumulators %360, %361 → bn3, M-row-pair 3
        %ep_p15_swa = amdgpu.permlane_swap %360 16 : vector<4xbf16>
        %ep_p15_swb = amdgpu.permlane_swap %361 16 : vector<4xbf16>
        %ep_p15_hi = vector.shuffle %360, %ep_p15_swa [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p15_lo = vector.shuffle %ep_p15_swb, %361 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %ep_p15_vec = arith.select %ep_is_even, %ep_p15_hi, %ep_p15_lo : vector<8xbf16>
        %ep_p15_rsel = arith.select %ep_is_even, %ep_mt6, %ep_mt7 : index
        %ep_p15_addr = arith.addi %ep_bn3, %ep_p15_rsel overflow<nsw> : index
        vector.store %ep_p15_vec, %ep_buf[%ep_p15_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: !hal.fence, %arg11: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg10) => %arg0 : !hal.buffer_view -> tensor<1024x1024xi8>
    %1 = hal.tensor.import wait(%arg10) => %arg1 : !hal.buffer_view -> tensor<1024x64xi8>
    %2 = hal.tensor.import wait(%arg10) => %arg2 : !hal.buffer_view -> tensor<1024x1024xi8>
    %3 = hal.tensor.import wait(%arg10) => %arg3 : !hal.buffer_view -> tensor<1024x64xi8>
    %4 = hal.tensor.import wait(%arg10) => %arg4 : !hal.buffer_view -> tensor<1024x1024xbf16>
    %5 = flow.dispatch @gemm::@gemm[%arg5, %arg6, %arg7, %arg8, %arg9](%0, %1, %2, %3, %4, %arg5, %arg6, %arg7, %arg8, %arg9) : (tensor<1024x1024xi8>, tensor<1024x64xi8>, tensor<1024x1024xi8>, tensor<1024x64xi8>, tensor<1024x1024xbf16>, index, index, index, index, index) -> %4
    %6 = hal.tensor.barrier join(%5 : tensor<1024x1024xbf16>) => %arg11 : !hal.fence
    %7 = hal.tensor.export %6 : tensor<1024x1024xbf16> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
