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
#map10 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32)>
#map11 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32 + 1024)>
#map12 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32)>
#map13 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 1024)>
#map14 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 2048)>
#map15 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 3072)>
#map16 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
#map17 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8) floordiv 256) * 131072 + 128)>
#map18 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 256) * 131072 + 32896)>
#map19 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 128) floordiv 256) * 131072 + 65664)>
#map20 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 192) floordiv 256) * 131072 + 98432)>
#map21 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
#map22 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048)>
#map23 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048 + 2048)>
#map24 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048 + 4096)>
#map25 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048 + 6144)>
#map26 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048)>
#map27 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 2048)>
#map28 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 4096)>
#map29 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 6144)>
#map30 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 8192)>
#map31 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 10240)>
#map32 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 12288)>
#map33 = affine_map<()[s0, s1, s2] -> (s0 * 16384 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 14336)>
#map34 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32 + 256)>
#map35 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32 + 1280)>
#map36 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 256)>
#map37 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 1280)>
#map38 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 2304)>
#map39 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 3328)>
#map40 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
#map41 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8) floordiv 256) * 131072 + 256)>
#map42 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 256) * 131072 + 33024)>
#map43 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 128) floordiv 256) * 131072 + 65792)>
#map44 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 192) floordiv 256) * 131072 + 98560)>
#map45 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32 + 512)>
#map46 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32 + 1536)>
#map47 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 512)>
#map48 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 1536)>
#map49 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 2560)>
#map50 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 3584)>
#map51 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8) floordiv 256) * 131072 + 384)>
#map52 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 256) * 131072 + 33152)>
#map53 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 128) floordiv 256) * 131072 + 65920)>
#map54 = affine_map<()[s0, s1, s2, s3] -> (s0 * 131072 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 192) floordiv 256) * 131072 + 98688)>
#map55 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32 + 768)>
#map56 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 4 + (s1 floordiv 64) * 2048 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32 + 1792)>
#map57 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 768)>
#map58 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 1792)>
#map59 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 2816)>
#map60 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 4096 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 3840)>
#map61 = affine_map<()[s0] -> (s0 * 256)>
#map62 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
#map63 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
#map64 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map65 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map66 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map67 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
#map68 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
#map69 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
#map70 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
#map71 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
#map72 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
#map73 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
#map74 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map75 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map76 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map77 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#map78 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
#map79 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
#map80 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
#map81 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
#map82 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
#map83 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
#map84 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
#map85 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      stream.return %c32, %c32, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index) attributes {translation_info = #translation} {
        %c4_i32 = arith.constant 4 : i32
        %c32_i14 = arith.constant 32 : i14
        %c512_i14 = arith.constant 512 : i14
        %c2147483645_i64 = arith.constant 2147483645 : i64
        %c262144_i64 = arith.constant 262144 : i64
        %c4194304_i64 = arith.constant 4194304 : i64
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
        %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<bf16>
        %block_id_x = gpu.block_id x upper_bound 32
        %block_id_y = gpu.block_id y upper_bound 32
        %thread_id_x = gpu.thread_id x upper_bound 256
        %thread_id_y = gpu.thread_id y upper_bound 2
        %reinterpret_cast = memref.reinterpret_cast %4 to offset: [0], sizes: [8192, 8192], strides: [%arg9, 1] : memref<bf16> to memref<8192x8192xbf16, strided<[?, 1]>>
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
        %12 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c4194304_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
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
        %23 = amdgpu.fat_raw_buffer_cast %cast_5 validBytes(%c4194304_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %23[%22], %alloc_0[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %24 = affine.apply #map5()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%24], %alloc_0[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %25 = affine.apply #map7()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%25], %alloc_0[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %26 = affine.apply #map9()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%26], %alloc_0[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_6 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_7 = memref.cast %reinterpret_cast_6 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %27 = amdgpu.fat_raw_buffer_cast %cast_7 validBytes(%c262144_i64) cacheSwizzleStride(%c32_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %28 = affine.apply #map10()[%block_id_x, %thread_id_x]
        %29 = vector.load %27[%28] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %30 = vector.bitcast %29 : vector<4xi8> to vector<4xf8E8M0FNU>
        %31 = affine.apply #map11()[%block_id_x, %thread_id_x]
        %32 = vector.load %27[%31] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %33 = vector.bitcast %32 : vector<4xi8> to vector<4xf8E8M0FNU>
        %reinterpret_cast_8 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_9 = memref.cast %reinterpret_cast_8 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %34 = amdgpu.fat_raw_buffer_cast %cast_9 validBytes(%c262144_i64) cacheSwizzleStride(%c32_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %35 = affine.apply #map12()[%block_id_y, %thread_id_y, %thread_id_x]
        %36 = vector.load %34[%35] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %37 = vector.bitcast %36 : vector<4xi8> to vector<4xf8E8M0FNU>
        %38 = affine.apply #map13()[%block_id_y, %thread_id_y, %thread_id_x]
        %39 = vector.load %34[%38] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %40 = vector.bitcast %39 : vector<4xi8> to vector<4xf8E8M0FNU>
        %41 = affine.apply #map14()[%block_id_y, %thread_id_y, %thread_id_x]
        %42 = vector.load %34[%41] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %43 = vector.bitcast %42 : vector<4xi8> to vector<4xf8E8M0FNU>
        %44 = affine.apply #map15()[%block_id_y, %thread_id_y, %thread_id_x]
        %45 = vector.load %34[%44] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %46 = vector.bitcast %45 : vector<4xi8> to vector<4xf8E8M0FNU>
        amdgpu.memory_counter_wait load(6)
        rocdl.s.barrier
        %47 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %48 = arith.index_cast %47 : index to i32
        %49 = arith.cmpi sge, %48, %c4_i32 : i32
        %50 = arith.cmpi slt, %48, %c4_i32 : i32
        scf.if %49 {
          rocdl.s.barrier
        }
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(0)
        rocdl.s.barrier
        %51 = affine.apply #map17()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%51], %alloc_1[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %52 = affine.apply #map18()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%52], %alloc_1[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %53 = affine.apply #map19()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%53], %alloc_1[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %54 = affine.apply #map20()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%54], %alloc_1[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %55 = affine.apply #map17()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%55], %alloc[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %56 = affine.apply #map18()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%56], %alloc[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %57 = affine.apply #map19()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%57], %alloc[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %58 = affine.apply #map20()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%58], %alloc[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        rocdl.sched.barrier 0
        %reinterpret_cast_10 = memref.reinterpret_cast %alloc_2 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
        %59 = affine.apply #map21()[%thread_id_x]
        %60 = arith.xori %59, %9 : index
        %61 = affine.apply #map22()[%thread_id_x, %60]
        %62 = vector.load %reinterpret_cast_10[%61] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %63 = affine.apply #map23()[%thread_id_x, %60]
        %64 = vector.load %reinterpret_cast_10[%63] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %65 = affine.apply #map24()[%thread_id_x, %60]
        %66 = vector.load %reinterpret_cast_10[%65] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %67 = affine.apply #map25()[%thread_id_x, %60]
        %68 = vector.load %reinterpret_cast_10[%67] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %reinterpret_cast_11 = memref.reinterpret_cast %alloc_0 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
        %69 = affine.apply #map26()[%thread_id_y, %thread_id_x, %60]
        %70 = vector.load %reinterpret_cast_11[%69] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %71 = affine.apply #map27()[%thread_id_y, %thread_id_x, %60]
        %72 = vector.load %reinterpret_cast_11[%71] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %73 = affine.apply #map28()[%thread_id_y, %thread_id_x, %60]
        %74 = vector.load %reinterpret_cast_11[%73] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %75 = affine.apply #map29()[%thread_id_y, %thread_id_x, %60]
        %76 = vector.load %reinterpret_cast_11[%75] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %77 = affine.apply #map30()[%thread_id_y, %thread_id_x, %60]
        %78 = vector.load %reinterpret_cast_11[%77] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %79 = affine.apply #map31()[%thread_id_y, %thread_id_x, %60]
        %80 = vector.load %reinterpret_cast_11[%79] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %81 = affine.apply #map32()[%thread_id_y, %thread_id_x, %60]
        %82 = vector.load %reinterpret_cast_11[%81] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %83 = affine.apply #map33()[%thread_id_y, %thread_id_x, %60]
        %84 = vector.load %reinterpret_cast_11[%83] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %85 = vector.bitcast %62 : vector<16xi8> to vector<32xf4E2M1FN>
        %86 = vector.bitcast %64 : vector<16xi8> to vector<32xf4E2M1FN>
        %87 = vector.bitcast %66 : vector<16xi8> to vector<32xf4E2M1FN>
        %88 = vector.bitcast %68 : vector<16xi8> to vector<32xf4E2M1FN>
        %89 = vector.bitcast %70 : vector<16xi8> to vector<32xf4E2M1FN>
        %90 = vector.bitcast %72 : vector<16xi8> to vector<32xf4E2M1FN>
        %91 = vector.bitcast %74 : vector<16xi8> to vector<32xf4E2M1FN>
        %92 = vector.bitcast %76 : vector<16xi8> to vector<32xf4E2M1FN>
        %93 = vector.bitcast %78 : vector<16xi8> to vector<32xf4E2M1FN>
        %94 = vector.bitcast %80 : vector<16xi8> to vector<32xf4E2M1FN>
        %95 = vector.bitcast %82 : vector<16xi8> to vector<32xf4E2M1FN>
        %96 = vector.bitcast %84 : vector<16xi8> to vector<32xf4E2M1FN>
        %97 = affine.apply #map34()[%block_id_x, %thread_id_x]
        %98 = vector.load %27[%97] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %99 = vector.bitcast %98 : vector<4xi8> to vector<4xf8E8M0FNU>
        %100 = affine.apply #map35()[%block_id_x, %thread_id_x]
        %101 = vector.load %27[%100] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %102 = vector.bitcast %101 : vector<4xi8> to vector<4xf8E8M0FNU>
        %103 = affine.apply #map36()[%block_id_y, %thread_id_y, %thread_id_x]
        %104 = vector.load %34[%103] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %105 = vector.bitcast %104 : vector<4xi8> to vector<4xf8E8M0FNU>
        %106 = affine.apply #map37()[%block_id_y, %thread_id_y, %thread_id_x]
        %107 = vector.load %34[%106] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %108 = vector.bitcast %107 : vector<4xi8> to vector<4xf8E8M0FNU>
        %109 = affine.apply #map38()[%block_id_y, %thread_id_y, %thread_id_x]
        %110 = vector.load %34[%109] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %111 = vector.bitcast %110 : vector<4xi8> to vector<4xf8E8M0FNU>
        %112 = affine.apply #map39()[%block_id_y, %thread_id_y, %thread_id_x]
        %113 = vector.load %34[%112] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %114 = vector.bitcast %113 : vector<4xi8> to vector<4xf8E8M0FNU>
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %115 = amdgpu.scaled_mfma 16x16x128 (%30[0] * %85) * (%37[0] * %89) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %116 = amdgpu.scaled_mfma 16x16x128 (%30[0] * %85) * (%37[1] * %90) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %117 = amdgpu.scaled_mfma 16x16x128 (%30[0] * %85) * (%40[0] * %91) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %118 = amdgpu.scaled_mfma 16x16x128 (%30[0] * %85) * (%40[1] * %92) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %119 = amdgpu.scaled_mfma 16x16x128 (%30[0] * %85) * (%43[0] * %93) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %120 = amdgpu.scaled_mfma 16x16x128 (%30[0] * %85) * (%43[1] * %94) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %121 = amdgpu.scaled_mfma 16x16x128 (%30[0] * %85) * (%46[0] * %95) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %122 = amdgpu.scaled_mfma 16x16x128 (%30[0] * %85) * (%46[1] * %96) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %123 = amdgpu.scaled_mfma 16x16x128 (%30[1] * %86) * (%37[0] * %89) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %124 = amdgpu.scaled_mfma 16x16x128 (%30[1] * %86) * (%37[1] * %90) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %125 = amdgpu.scaled_mfma 16x16x128 (%30[1] * %86) * (%40[0] * %91) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %126 = amdgpu.scaled_mfma 16x16x128 (%30[1] * %86) * (%40[1] * %92) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %127 = amdgpu.scaled_mfma 16x16x128 (%30[1] * %86) * (%43[0] * %93) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %128 = amdgpu.scaled_mfma 16x16x128 (%30[1] * %86) * (%43[1] * %94) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %129 = amdgpu.scaled_mfma 16x16x128 (%30[1] * %86) * (%46[0] * %95) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %130 = amdgpu.scaled_mfma 16x16x128 (%30[1] * %86) * (%46[1] * %96) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %131 = amdgpu.scaled_mfma 16x16x128 (%33[0] * %87) * (%37[0] * %89) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %132 = amdgpu.scaled_mfma 16x16x128 (%33[0] * %87) * (%37[1] * %90) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %133 = amdgpu.scaled_mfma 16x16x128 (%33[0] * %87) * (%40[0] * %91) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %134 = amdgpu.scaled_mfma 16x16x128 (%33[0] * %87) * (%40[1] * %92) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %135 = amdgpu.scaled_mfma 16x16x128 (%33[0] * %87) * (%43[0] * %93) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %136 = amdgpu.scaled_mfma 16x16x128 (%33[0] * %87) * (%43[1] * %94) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %137 = amdgpu.scaled_mfma 16x16x128 (%33[0] * %87) * (%46[0] * %95) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %138 = amdgpu.scaled_mfma 16x16x128 (%33[0] * %87) * (%46[1] * %96) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %139 = amdgpu.scaled_mfma 16x16x128 (%33[1] * %88) * (%37[0] * %89) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %140 = amdgpu.scaled_mfma 16x16x128 (%33[1] * %88) * (%37[1] * %90) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %141 = amdgpu.scaled_mfma 16x16x128 (%33[1] * %88) * (%40[0] * %91) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %142 = amdgpu.scaled_mfma 16x16x128 (%33[1] * %88) * (%40[1] * %92) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %143 = amdgpu.scaled_mfma 16x16x128 (%33[1] * %88) * (%43[0] * %93) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %144 = amdgpu.scaled_mfma 16x16x128 (%33[1] * %88) * (%43[1] * %94) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %145 = amdgpu.scaled_mfma 16x16x128 (%33[1] * %88) * (%46[0] * %95) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %146 = amdgpu.scaled_mfma 16x16x128 (%33[1] * %88) * (%46[1] * %96) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        %147 = affine.apply #map40()[%thread_id_x]
        %148 = arith.xori %147, %9 : index
        %149 = affine.apply #map22()[%thread_id_x, %148]
        %150 = vector.load %reinterpret_cast_10[%149] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %151 = affine.apply #map23()[%thread_id_x, %148]
        %152 = vector.load %reinterpret_cast_10[%151] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %153 = affine.apply #map24()[%thread_id_x, %148]
        %154 = vector.load %reinterpret_cast_10[%153] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %155 = affine.apply #map25()[%thread_id_x, %148]
        %156 = vector.load %reinterpret_cast_10[%155] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %157 = affine.apply #map26()[%thread_id_y, %thread_id_x, %148]
        %158 = vector.load %reinterpret_cast_11[%157] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %159 = affine.apply #map27()[%thread_id_y, %thread_id_x, %148]
        %160 = vector.load %reinterpret_cast_11[%159] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %161 = affine.apply #map28()[%thread_id_y, %thread_id_x, %148]
        %162 = vector.load %reinterpret_cast_11[%161] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %163 = affine.apply #map29()[%thread_id_y, %thread_id_x, %148]
        %164 = vector.load %reinterpret_cast_11[%163] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %165 = affine.apply #map30()[%thread_id_y, %thread_id_x, %148]
        %166 = vector.load %reinterpret_cast_11[%165] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %167 = affine.apply #map31()[%thread_id_y, %thread_id_x, %148]
        %168 = vector.load %reinterpret_cast_11[%167] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %169 = affine.apply #map32()[%thread_id_y, %thread_id_x, %148]
        %170 = vector.load %reinterpret_cast_11[%169] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %171 = affine.apply #map33()[%thread_id_y, %thread_id_x, %148]
        %172 = vector.load %reinterpret_cast_11[%171] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %173 = vector.bitcast %150 : vector<16xi8> to vector<32xf4E2M1FN>
        %174 = vector.bitcast %152 : vector<16xi8> to vector<32xf4E2M1FN>
        %175 = vector.bitcast %154 : vector<16xi8> to vector<32xf4E2M1FN>
        %176 = vector.bitcast %156 : vector<16xi8> to vector<32xf4E2M1FN>
        %177 = vector.bitcast %158 : vector<16xi8> to vector<32xf4E2M1FN>
        %178 = vector.bitcast %160 : vector<16xi8> to vector<32xf4E2M1FN>
        %179 = vector.bitcast %162 : vector<16xi8> to vector<32xf4E2M1FN>
        %180 = vector.bitcast %164 : vector<16xi8> to vector<32xf4E2M1FN>
        %181 = vector.bitcast %166 : vector<16xi8> to vector<32xf4E2M1FN>
        %182 = vector.bitcast %168 : vector<16xi8> to vector<32xf4E2M1FN>
        %183 = vector.bitcast %170 : vector<16xi8> to vector<32xf4E2M1FN>
        %184 = vector.bitcast %172 : vector<16xi8> to vector<32xf4E2M1FN>
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(6)
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %185 = amdgpu.scaled_mfma 16x16x128 (%30[2] * %173) * (%37[2] * %177) + %115 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %186 = amdgpu.scaled_mfma 16x16x128 (%30[2] * %173) * (%37[3] * %178) + %116 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %187 = amdgpu.scaled_mfma 16x16x128 (%30[2] * %173) * (%40[2] * %179) + %117 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %188 = amdgpu.scaled_mfma 16x16x128 (%30[2] * %173) * (%40[3] * %180) + %118 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %189 = amdgpu.scaled_mfma 16x16x128 (%30[2] * %173) * (%43[2] * %181) + %119 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %190 = amdgpu.scaled_mfma 16x16x128 (%30[2] * %173) * (%43[3] * %182) + %120 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %191 = amdgpu.scaled_mfma 16x16x128 (%30[2] * %173) * (%46[2] * %183) + %121 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %192 = amdgpu.scaled_mfma 16x16x128 (%30[2] * %173) * (%46[3] * %184) + %122 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %193 = amdgpu.scaled_mfma 16x16x128 (%30[3] * %174) * (%37[2] * %177) + %123 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %194 = amdgpu.scaled_mfma 16x16x128 (%30[3] * %174) * (%37[3] * %178) + %124 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %195 = amdgpu.scaled_mfma 16x16x128 (%30[3] * %174) * (%40[2] * %179) + %125 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %196 = amdgpu.scaled_mfma 16x16x128 (%30[3] * %174) * (%40[3] * %180) + %126 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %197 = amdgpu.scaled_mfma 16x16x128 (%30[3] * %174) * (%43[2] * %181) + %127 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %198 = amdgpu.scaled_mfma 16x16x128 (%30[3] * %174) * (%43[3] * %182) + %128 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %199 = amdgpu.scaled_mfma 16x16x128 (%30[3] * %174) * (%46[2] * %183) + %129 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %200 = amdgpu.scaled_mfma 16x16x128 (%30[3] * %174) * (%46[3] * %184) + %130 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %201 = amdgpu.scaled_mfma 16x16x128 (%33[2] * %175) * (%37[2] * %177) + %131 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %202 = amdgpu.scaled_mfma 16x16x128 (%33[2] * %175) * (%37[3] * %178) + %132 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %203 = amdgpu.scaled_mfma 16x16x128 (%33[2] * %175) * (%40[2] * %179) + %133 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %204 = amdgpu.scaled_mfma 16x16x128 (%33[2] * %175) * (%40[3] * %180) + %134 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %205 = amdgpu.scaled_mfma 16x16x128 (%33[2] * %175) * (%43[2] * %181) + %135 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %206 = amdgpu.scaled_mfma 16x16x128 (%33[2] * %175) * (%43[3] * %182) + %136 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %207 = amdgpu.scaled_mfma 16x16x128 (%33[2] * %175) * (%46[2] * %183) + %137 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %208 = amdgpu.scaled_mfma 16x16x128 (%33[2] * %175) * (%46[3] * %184) + %138 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %209 = amdgpu.scaled_mfma 16x16x128 (%33[3] * %176) * (%37[2] * %177) + %139 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %210 = amdgpu.scaled_mfma 16x16x128 (%33[3] * %176) * (%37[3] * %178) + %140 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %211 = amdgpu.scaled_mfma 16x16x128 (%33[3] * %176) * (%40[2] * %179) + %141 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %212 = amdgpu.scaled_mfma 16x16x128 (%33[3] * %176) * (%40[3] * %180) + %142 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %213 = amdgpu.scaled_mfma 16x16x128 (%33[3] * %176) * (%43[2] * %181) + %143 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %214 = amdgpu.scaled_mfma 16x16x128 (%33[3] * %176) * (%43[3] * %182) + %144 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %215 = amdgpu.scaled_mfma 16x16x128 (%33[3] * %176) * (%46[2] * %183) + %145 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %216 = amdgpu.scaled_mfma 16x16x128 (%33[3] * %176) * (%46[3] * %184) + %146 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(0)
        rocdl.s.barrier
        %217 = affine.apply #map41()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%217], %alloc_2[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %218 = affine.apply #map42()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%218], %alloc_2[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %219 = affine.apply #map43()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%219], %alloc_2[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %220 = affine.apply #map44()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%220], %alloc_2[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %221 = affine.apply #map41()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%221], %alloc_0[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %222 = affine.apply #map42()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%222], %alloc_0[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %223 = affine.apply #map43()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%223], %alloc_0[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %224 = affine.apply #map44()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%224], %alloc_0[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        rocdl.sched.barrier 0
        %reinterpret_cast_12 = memref.reinterpret_cast %alloc_1 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
        %225 = vector.load %reinterpret_cast_12[%61] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %226 = vector.load %reinterpret_cast_12[%63] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %227 = vector.load %reinterpret_cast_12[%65] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %228 = vector.load %reinterpret_cast_12[%67] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %reinterpret_cast_13 = memref.reinterpret_cast %alloc to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
        %229 = vector.load %reinterpret_cast_13[%69] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %230 = vector.load %reinterpret_cast_13[%71] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %231 = vector.load %reinterpret_cast_13[%73] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %232 = vector.load %reinterpret_cast_13[%75] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %233 = vector.load %reinterpret_cast_13[%77] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %234 = vector.load %reinterpret_cast_13[%79] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %235 = vector.load %reinterpret_cast_13[%81] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %236 = vector.load %reinterpret_cast_13[%83] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %237 = vector.bitcast %225 : vector<16xi8> to vector<32xf4E2M1FN>
        %238 = vector.bitcast %226 : vector<16xi8> to vector<32xf4E2M1FN>
        %239 = vector.bitcast %227 : vector<16xi8> to vector<32xf4E2M1FN>
        %240 = vector.bitcast %228 : vector<16xi8> to vector<32xf4E2M1FN>
        %241 = vector.bitcast %229 : vector<16xi8> to vector<32xf4E2M1FN>
        %242 = vector.bitcast %230 : vector<16xi8> to vector<32xf4E2M1FN>
        %243 = vector.bitcast %231 : vector<16xi8> to vector<32xf4E2M1FN>
        %244 = vector.bitcast %232 : vector<16xi8> to vector<32xf4E2M1FN>
        %245 = vector.bitcast %233 : vector<16xi8> to vector<32xf4E2M1FN>
        %246 = vector.bitcast %234 : vector<16xi8> to vector<32xf4E2M1FN>
        %247 = vector.bitcast %235 : vector<16xi8> to vector<32xf4E2M1FN>
        %248 = vector.bitcast %236 : vector<16xi8> to vector<32xf4E2M1FN>
        %249 = affine.apply #map45()[%block_id_x, %thread_id_x]
        %250 = vector.load %27[%249] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %251 = vector.bitcast %250 : vector<4xi8> to vector<4xf8E8M0FNU>
        %252 = affine.apply #map46()[%block_id_x, %thread_id_x]
        %253 = vector.load %27[%252] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %254 = vector.bitcast %253 : vector<4xi8> to vector<4xf8E8M0FNU>
        %255 = affine.apply #map47()[%block_id_y, %thread_id_y, %thread_id_x]
        %256 = vector.load %34[%255] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %257 = vector.bitcast %256 : vector<4xi8> to vector<4xf8E8M0FNU>
        %258 = affine.apply #map48()[%block_id_y, %thread_id_y, %thread_id_x]
        %259 = vector.load %34[%258] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %260 = vector.bitcast %259 : vector<4xi8> to vector<4xf8E8M0FNU>
        %261 = affine.apply #map49()[%block_id_y, %thread_id_y, %thread_id_x]
        %262 = vector.load %34[%261] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %263 = vector.bitcast %262 : vector<4xi8> to vector<4xf8E8M0FNU>
        %264 = affine.apply #map50()[%block_id_y, %thread_id_y, %thread_id_x]
        %265 = vector.load %34[%264] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %266 = vector.bitcast %265 : vector<4xi8> to vector<4xf8E8M0FNU>
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %267 = amdgpu.scaled_mfma 16x16x128 (%99[0] * %237) * (%105[0] * %241) + %185 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %268 = amdgpu.scaled_mfma 16x16x128 (%99[0] * %237) * (%105[1] * %242) + %186 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %269 = amdgpu.scaled_mfma 16x16x128 (%99[0] * %237) * (%108[0] * %243) + %187 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %270 = amdgpu.scaled_mfma 16x16x128 (%99[0] * %237) * (%108[1] * %244) + %188 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %271 = amdgpu.scaled_mfma 16x16x128 (%99[0] * %237) * (%111[0] * %245) + %189 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %272 = amdgpu.scaled_mfma 16x16x128 (%99[0] * %237) * (%111[1] * %246) + %190 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %273 = amdgpu.scaled_mfma 16x16x128 (%99[0] * %237) * (%114[0] * %247) + %191 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %274 = amdgpu.scaled_mfma 16x16x128 (%99[0] * %237) * (%114[1] * %248) + %192 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %275 = amdgpu.scaled_mfma 16x16x128 (%99[1] * %238) * (%105[0] * %241) + %193 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %276 = amdgpu.scaled_mfma 16x16x128 (%99[1] * %238) * (%105[1] * %242) + %194 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %277 = amdgpu.scaled_mfma 16x16x128 (%99[1] * %238) * (%108[0] * %243) + %195 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %278 = amdgpu.scaled_mfma 16x16x128 (%99[1] * %238) * (%108[1] * %244) + %196 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %279 = amdgpu.scaled_mfma 16x16x128 (%99[1] * %238) * (%111[0] * %245) + %197 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %280 = amdgpu.scaled_mfma 16x16x128 (%99[1] * %238) * (%111[1] * %246) + %198 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %281 = amdgpu.scaled_mfma 16x16x128 (%99[1] * %238) * (%114[0] * %247) + %199 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %282 = amdgpu.scaled_mfma 16x16x128 (%99[1] * %238) * (%114[1] * %248) + %200 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %283 = amdgpu.scaled_mfma 16x16x128 (%102[0] * %239) * (%105[0] * %241) + %201 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %284 = amdgpu.scaled_mfma 16x16x128 (%102[0] * %239) * (%105[1] * %242) + %202 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %285 = amdgpu.scaled_mfma 16x16x128 (%102[0] * %239) * (%108[0] * %243) + %203 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %286 = amdgpu.scaled_mfma 16x16x128 (%102[0] * %239) * (%108[1] * %244) + %204 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %287 = amdgpu.scaled_mfma 16x16x128 (%102[0] * %239) * (%111[0] * %245) + %205 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %288 = amdgpu.scaled_mfma 16x16x128 (%102[0] * %239) * (%111[1] * %246) + %206 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %289 = amdgpu.scaled_mfma 16x16x128 (%102[0] * %239) * (%114[0] * %247) + %207 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %290 = amdgpu.scaled_mfma 16x16x128 (%102[0] * %239) * (%114[1] * %248) + %208 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %291 = amdgpu.scaled_mfma 16x16x128 (%102[1] * %240) * (%105[0] * %241) + %209 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %292 = amdgpu.scaled_mfma 16x16x128 (%102[1] * %240) * (%105[1] * %242) + %210 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %293 = amdgpu.scaled_mfma 16x16x128 (%102[1] * %240) * (%108[0] * %243) + %211 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %294 = amdgpu.scaled_mfma 16x16x128 (%102[1] * %240) * (%108[1] * %244) + %212 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %295 = amdgpu.scaled_mfma 16x16x128 (%102[1] * %240) * (%111[0] * %245) + %213 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %296 = amdgpu.scaled_mfma 16x16x128 (%102[1] * %240) * (%111[1] * %246) + %214 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %297 = amdgpu.scaled_mfma 16x16x128 (%102[1] * %240) * (%114[0] * %247) + %215 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %298 = amdgpu.scaled_mfma 16x16x128 (%102[1] * %240) * (%114[1] * %248) + %216 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        %299 = vector.load %reinterpret_cast_12[%149] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %300 = vector.load %reinterpret_cast_12[%151] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %301 = vector.load %reinterpret_cast_12[%153] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %302 = vector.load %reinterpret_cast_12[%155] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %303 = vector.load %reinterpret_cast_13[%157] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %304 = vector.load %reinterpret_cast_13[%159] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %305 = vector.load %reinterpret_cast_13[%161] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %306 = vector.load %reinterpret_cast_13[%163] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %307 = vector.load %reinterpret_cast_13[%165] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %308 = vector.load %reinterpret_cast_13[%167] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %309 = vector.load %reinterpret_cast_13[%169] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %310 = vector.load %reinterpret_cast_13[%171] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %311 = vector.bitcast %299 : vector<16xi8> to vector<32xf4E2M1FN>
        %312 = vector.bitcast %300 : vector<16xi8> to vector<32xf4E2M1FN>
        %313 = vector.bitcast %301 : vector<16xi8> to vector<32xf4E2M1FN>
        %314 = vector.bitcast %302 : vector<16xi8> to vector<32xf4E2M1FN>
        %315 = vector.bitcast %303 : vector<16xi8> to vector<32xf4E2M1FN>
        %316 = vector.bitcast %304 : vector<16xi8> to vector<32xf4E2M1FN>
        %317 = vector.bitcast %305 : vector<16xi8> to vector<32xf4E2M1FN>
        %318 = vector.bitcast %306 : vector<16xi8> to vector<32xf4E2M1FN>
        %319 = vector.bitcast %307 : vector<16xi8> to vector<32xf4E2M1FN>
        %320 = vector.bitcast %308 : vector<16xi8> to vector<32xf4E2M1FN>
        %321 = vector.bitcast %309 : vector<16xi8> to vector<32xf4E2M1FN>
        %322 = vector.bitcast %310 : vector<16xi8> to vector<32xf4E2M1FN>
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(6)
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %323 = amdgpu.scaled_mfma 16x16x128 (%99[2] * %311) * (%105[2] * %315) + %267 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %324 = amdgpu.scaled_mfma 16x16x128 (%99[2] * %311) * (%105[3] * %316) + %268 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %325 = amdgpu.scaled_mfma 16x16x128 (%99[2] * %311) * (%108[2] * %317) + %269 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %326 = amdgpu.scaled_mfma 16x16x128 (%99[2] * %311) * (%108[3] * %318) + %270 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %327 = amdgpu.scaled_mfma 16x16x128 (%99[2] * %311) * (%111[2] * %319) + %271 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %328 = amdgpu.scaled_mfma 16x16x128 (%99[2] * %311) * (%111[3] * %320) + %272 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %329 = amdgpu.scaled_mfma 16x16x128 (%99[2] * %311) * (%114[2] * %321) + %273 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %330 = amdgpu.scaled_mfma 16x16x128 (%99[2] * %311) * (%114[3] * %322) + %274 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %331 = amdgpu.scaled_mfma 16x16x128 (%99[3] * %312) * (%105[2] * %315) + %275 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %332 = amdgpu.scaled_mfma 16x16x128 (%99[3] * %312) * (%105[3] * %316) + %276 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %333 = amdgpu.scaled_mfma 16x16x128 (%99[3] * %312) * (%108[2] * %317) + %277 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %334 = amdgpu.scaled_mfma 16x16x128 (%99[3] * %312) * (%108[3] * %318) + %278 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %335 = amdgpu.scaled_mfma 16x16x128 (%99[3] * %312) * (%111[2] * %319) + %279 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %336 = amdgpu.scaled_mfma 16x16x128 (%99[3] * %312) * (%111[3] * %320) + %280 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %337 = amdgpu.scaled_mfma 16x16x128 (%99[3] * %312) * (%114[2] * %321) + %281 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %338 = amdgpu.scaled_mfma 16x16x128 (%99[3] * %312) * (%114[3] * %322) + %282 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %339 = amdgpu.scaled_mfma 16x16x128 (%102[2] * %313) * (%105[2] * %315) + %283 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %340 = amdgpu.scaled_mfma 16x16x128 (%102[2] * %313) * (%105[3] * %316) + %284 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %341 = amdgpu.scaled_mfma 16x16x128 (%102[2] * %313) * (%108[2] * %317) + %285 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %342 = amdgpu.scaled_mfma 16x16x128 (%102[2] * %313) * (%108[3] * %318) + %286 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %343 = amdgpu.scaled_mfma 16x16x128 (%102[2] * %313) * (%111[2] * %319) + %287 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %344 = amdgpu.scaled_mfma 16x16x128 (%102[2] * %313) * (%111[3] * %320) + %288 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %345 = amdgpu.scaled_mfma 16x16x128 (%102[2] * %313) * (%114[2] * %321) + %289 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %346 = amdgpu.scaled_mfma 16x16x128 (%102[2] * %313) * (%114[3] * %322) + %290 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %347 = amdgpu.scaled_mfma 16x16x128 (%102[3] * %314) * (%105[2] * %315) + %291 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %348 = amdgpu.scaled_mfma 16x16x128 (%102[3] * %314) * (%105[3] * %316) + %292 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %349 = amdgpu.scaled_mfma 16x16x128 (%102[3] * %314) * (%108[2] * %317) + %293 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %350 = amdgpu.scaled_mfma 16x16x128 (%102[3] * %314) * (%108[3] * %318) + %294 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %351 = amdgpu.scaled_mfma 16x16x128 (%102[3] * %314) * (%111[2] * %319) + %295 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %352 = amdgpu.scaled_mfma 16x16x128 (%102[3] * %314) * (%111[3] * %320) + %296 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %353 = amdgpu.scaled_mfma 16x16x128 (%102[3] * %314) * (%114[2] * %321) + %297 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %354 = amdgpu.scaled_mfma 16x16x128 (%102[3] * %314) * (%114[3] * %322) + %298 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(0)
        rocdl.s.barrier
        %355 = affine.apply #map51()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%355], %alloc_1[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %356 = affine.apply #map52()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%356], %alloc_1[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %357 = affine.apply #map53()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%357], %alloc_1[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %358 = affine.apply #map54()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%358], %alloc_1[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %359 = affine.apply #map51()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%359], %alloc[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %360 = affine.apply #map52()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%360], %alloc[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %361 = affine.apply #map53()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%361], %alloc[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %362 = affine.apply #map54()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%362], %alloc[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        rocdl.sched.barrier 0
        %363 = vector.load %reinterpret_cast_10[%61] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %364 = vector.load %reinterpret_cast_10[%63] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %365 = vector.load %reinterpret_cast_10[%65] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %366 = vector.load %reinterpret_cast_10[%67] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %367 = vector.load %reinterpret_cast_11[%69] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %368 = vector.load %reinterpret_cast_11[%71] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %369 = vector.load %reinterpret_cast_11[%73] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %370 = vector.load %reinterpret_cast_11[%75] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %371 = vector.load %reinterpret_cast_11[%77] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %372 = vector.load %reinterpret_cast_11[%79] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %373 = vector.load %reinterpret_cast_11[%81] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %374 = vector.load %reinterpret_cast_11[%83] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %375 = vector.bitcast %363 : vector<16xi8> to vector<32xf4E2M1FN>
        %376 = vector.bitcast %364 : vector<16xi8> to vector<32xf4E2M1FN>
        %377 = vector.bitcast %365 : vector<16xi8> to vector<32xf4E2M1FN>
        %378 = vector.bitcast %366 : vector<16xi8> to vector<32xf4E2M1FN>
        %379 = vector.bitcast %367 : vector<16xi8> to vector<32xf4E2M1FN>
        %380 = vector.bitcast %368 : vector<16xi8> to vector<32xf4E2M1FN>
        %381 = vector.bitcast %369 : vector<16xi8> to vector<32xf4E2M1FN>
        %382 = vector.bitcast %370 : vector<16xi8> to vector<32xf4E2M1FN>
        %383 = vector.bitcast %371 : vector<16xi8> to vector<32xf4E2M1FN>
        %384 = vector.bitcast %372 : vector<16xi8> to vector<32xf4E2M1FN>
        %385 = vector.bitcast %373 : vector<16xi8> to vector<32xf4E2M1FN>
        %386 = vector.bitcast %374 : vector<16xi8> to vector<32xf4E2M1FN>
        %387 = affine.apply #map55()[%block_id_x, %thread_id_x]
        %388 = vector.load %27[%387] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %389 = vector.bitcast %388 : vector<4xi8> to vector<4xf8E8M0FNU>
        %390 = affine.apply #map56()[%block_id_x, %thread_id_x]
        %391 = vector.load %27[%390] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %392 = vector.bitcast %391 : vector<4xi8> to vector<4xf8E8M0FNU>
        %393 = affine.apply #map57()[%block_id_y, %thread_id_y, %thread_id_x]
        %394 = vector.load %34[%393] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %395 = vector.bitcast %394 : vector<4xi8> to vector<4xf8E8M0FNU>
        %396 = affine.apply #map58()[%block_id_y, %thread_id_y, %thread_id_x]
        %397 = vector.load %34[%396] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %398 = vector.bitcast %397 : vector<4xi8> to vector<4xf8E8M0FNU>
        %399 = affine.apply #map59()[%block_id_y, %thread_id_y, %thread_id_x]
        %400 = vector.load %34[%399] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %401 = vector.bitcast %400 : vector<4xi8> to vector<4xf8E8M0FNU>
        %402 = affine.apply #map60()[%block_id_y, %thread_id_y, %thread_id_x]
        %403 = vector.load %34[%402] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %404 = vector.bitcast %403 : vector<4xi8> to vector<4xf8E8M0FNU>
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %405 = amdgpu.scaled_mfma 16x16x128 (%251[0] * %375) * (%257[0] * %379) + %323 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %406 = amdgpu.scaled_mfma 16x16x128 (%251[0] * %375) * (%257[1] * %380) + %324 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %407 = amdgpu.scaled_mfma 16x16x128 (%251[0] * %375) * (%260[0] * %381) + %325 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %408 = amdgpu.scaled_mfma 16x16x128 (%251[0] * %375) * (%260[1] * %382) + %326 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %409 = amdgpu.scaled_mfma 16x16x128 (%251[0] * %375) * (%263[0] * %383) + %327 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %410 = amdgpu.scaled_mfma 16x16x128 (%251[0] * %375) * (%263[1] * %384) + %328 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %411 = amdgpu.scaled_mfma 16x16x128 (%251[0] * %375) * (%266[0] * %385) + %329 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %412 = amdgpu.scaled_mfma 16x16x128 (%251[0] * %375) * (%266[1] * %386) + %330 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %413 = amdgpu.scaled_mfma 16x16x128 (%251[1] * %376) * (%257[0] * %379) + %331 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %414 = amdgpu.scaled_mfma 16x16x128 (%251[1] * %376) * (%257[1] * %380) + %332 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %415 = amdgpu.scaled_mfma 16x16x128 (%251[1] * %376) * (%260[0] * %381) + %333 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %416 = amdgpu.scaled_mfma 16x16x128 (%251[1] * %376) * (%260[1] * %382) + %334 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %417 = amdgpu.scaled_mfma 16x16x128 (%251[1] * %376) * (%263[0] * %383) + %335 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %418 = amdgpu.scaled_mfma 16x16x128 (%251[1] * %376) * (%263[1] * %384) + %336 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %419 = amdgpu.scaled_mfma 16x16x128 (%251[1] * %376) * (%266[0] * %385) + %337 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %420 = amdgpu.scaled_mfma 16x16x128 (%251[1] * %376) * (%266[1] * %386) + %338 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %421 = amdgpu.scaled_mfma 16x16x128 (%254[0] * %377) * (%257[0] * %379) + %339 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %422 = amdgpu.scaled_mfma 16x16x128 (%254[0] * %377) * (%257[1] * %380) + %340 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %423 = amdgpu.scaled_mfma 16x16x128 (%254[0] * %377) * (%260[0] * %381) + %341 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %424 = amdgpu.scaled_mfma 16x16x128 (%254[0] * %377) * (%260[1] * %382) + %342 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %425 = amdgpu.scaled_mfma 16x16x128 (%254[0] * %377) * (%263[0] * %383) + %343 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %426 = amdgpu.scaled_mfma 16x16x128 (%254[0] * %377) * (%263[1] * %384) + %344 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %427 = amdgpu.scaled_mfma 16x16x128 (%254[0] * %377) * (%266[0] * %385) + %345 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %428 = amdgpu.scaled_mfma 16x16x128 (%254[0] * %377) * (%266[1] * %386) + %346 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %429 = amdgpu.scaled_mfma 16x16x128 (%254[1] * %378) * (%257[0] * %379) + %347 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %430 = amdgpu.scaled_mfma 16x16x128 (%254[1] * %378) * (%257[1] * %380) + %348 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %431 = amdgpu.scaled_mfma 16x16x128 (%254[1] * %378) * (%260[0] * %381) + %349 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %432 = amdgpu.scaled_mfma 16x16x128 (%254[1] * %378) * (%260[1] * %382) + %350 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %433 = amdgpu.scaled_mfma 16x16x128 (%254[1] * %378) * (%263[0] * %383) + %351 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %434 = amdgpu.scaled_mfma 16x16x128 (%254[1] * %378) * (%263[1] * %384) + %352 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %435 = amdgpu.scaled_mfma 16x16x128 (%254[1] * %378) * (%266[0] * %385) + %353 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %436 = amdgpu.scaled_mfma 16x16x128 (%254[1] * %378) * (%266[1] * %386) + %354 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        %437 = vector.load %reinterpret_cast_10[%149] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %438 = vector.load %reinterpret_cast_10[%151] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %439 = vector.load %reinterpret_cast_10[%153] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %440 = vector.load %reinterpret_cast_10[%155] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %441 = vector.load %reinterpret_cast_11[%157] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %442 = vector.load %reinterpret_cast_11[%159] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %443 = vector.load %reinterpret_cast_11[%161] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %444 = vector.load %reinterpret_cast_11[%163] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %445 = vector.load %reinterpret_cast_11[%165] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %446 = vector.load %reinterpret_cast_11[%167] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %447 = vector.load %reinterpret_cast_11[%169] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %448 = vector.load %reinterpret_cast_11[%171] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %449 = vector.bitcast %437 : vector<16xi8> to vector<32xf4E2M1FN>
        %450 = vector.bitcast %438 : vector<16xi8> to vector<32xf4E2M1FN>
        %451 = vector.bitcast %439 : vector<16xi8> to vector<32xf4E2M1FN>
        %452 = vector.bitcast %440 : vector<16xi8> to vector<32xf4E2M1FN>
        %453 = vector.bitcast %441 : vector<16xi8> to vector<32xf4E2M1FN>
        %454 = vector.bitcast %442 : vector<16xi8> to vector<32xf4E2M1FN>
        %455 = vector.bitcast %443 : vector<16xi8> to vector<32xf4E2M1FN>
        %456 = vector.bitcast %444 : vector<16xi8> to vector<32xf4E2M1FN>
        %457 = vector.bitcast %445 : vector<16xi8> to vector<32xf4E2M1FN>
        %458 = vector.bitcast %446 : vector<16xi8> to vector<32xf4E2M1FN>
        %459 = vector.bitcast %447 : vector<16xi8> to vector<32xf4E2M1FN>
        %460 = vector.bitcast %448 : vector<16xi8> to vector<32xf4E2M1FN>
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(6)
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %461 = amdgpu.scaled_mfma 16x16x128 (%251[2] * %449) * (%257[2] * %453) + %405 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %462 = amdgpu.scaled_mfma 16x16x128 (%251[2] * %449) * (%257[3] * %454) + %406 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %463 = amdgpu.scaled_mfma 16x16x128 (%251[2] * %449) * (%260[2] * %455) + %407 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %464 = amdgpu.scaled_mfma 16x16x128 (%251[2] * %449) * (%260[3] * %456) + %408 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %465 = amdgpu.scaled_mfma 16x16x128 (%251[2] * %449) * (%263[2] * %457) + %409 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %466 = amdgpu.scaled_mfma 16x16x128 (%251[2] * %449) * (%263[3] * %458) + %410 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %467 = amdgpu.scaled_mfma 16x16x128 (%251[2] * %449) * (%266[2] * %459) + %411 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %468 = amdgpu.scaled_mfma 16x16x128 (%251[2] * %449) * (%266[3] * %460) + %412 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %469 = amdgpu.scaled_mfma 16x16x128 (%251[3] * %450) * (%257[2] * %453) + %413 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %470 = amdgpu.scaled_mfma 16x16x128 (%251[3] * %450) * (%257[3] * %454) + %414 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %471 = amdgpu.scaled_mfma 16x16x128 (%251[3] * %450) * (%260[2] * %455) + %415 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %472 = amdgpu.scaled_mfma 16x16x128 (%251[3] * %450) * (%260[3] * %456) + %416 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %473 = amdgpu.scaled_mfma 16x16x128 (%251[3] * %450) * (%263[2] * %457) + %417 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %474 = amdgpu.scaled_mfma 16x16x128 (%251[3] * %450) * (%263[3] * %458) + %418 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %475 = amdgpu.scaled_mfma 16x16x128 (%251[3] * %450) * (%266[2] * %459) + %419 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %476 = amdgpu.scaled_mfma 16x16x128 (%251[3] * %450) * (%266[3] * %460) + %420 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %477 = amdgpu.scaled_mfma 16x16x128 (%254[2] * %451) * (%257[2] * %453) + %421 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %478 = amdgpu.scaled_mfma 16x16x128 (%254[2] * %451) * (%257[3] * %454) + %422 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %479 = amdgpu.scaled_mfma 16x16x128 (%254[2] * %451) * (%260[2] * %455) + %423 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %480 = amdgpu.scaled_mfma 16x16x128 (%254[2] * %451) * (%260[3] * %456) + %424 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %481 = amdgpu.scaled_mfma 16x16x128 (%254[2] * %451) * (%263[2] * %457) + %425 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %482 = amdgpu.scaled_mfma 16x16x128 (%254[2] * %451) * (%263[3] * %458) + %426 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %483 = amdgpu.scaled_mfma 16x16x128 (%254[2] * %451) * (%266[2] * %459) + %427 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %484 = amdgpu.scaled_mfma 16x16x128 (%254[2] * %451) * (%266[3] * %460) + %428 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %485 = amdgpu.scaled_mfma 16x16x128 (%254[3] * %452) * (%257[2] * %453) + %429 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %486 = amdgpu.scaled_mfma 16x16x128 (%254[3] * %452) * (%257[3] * %454) + %430 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %487 = amdgpu.scaled_mfma 16x16x128 (%254[3] * %452) * (%260[2] * %455) + %431 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %488 = amdgpu.scaled_mfma 16x16x128 (%254[3] * %452) * (%260[3] * %456) + %432 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %489 = amdgpu.scaled_mfma 16x16x128 (%254[3] * %452) * (%263[2] * %457) + %433 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %490 = amdgpu.scaled_mfma 16x16x128 (%254[3] * %452) * (%263[3] * %458) + %434 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %491 = amdgpu.scaled_mfma 16x16x128 (%254[3] * %452) * (%266[2] * %459) + %435 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %492 = amdgpu.scaled_mfma 16x16x128 (%254[3] * %452) * (%266[3] * %460) + %436 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        scf.if %50 {
          rocdl.s.barrier
        }
        amdgpu.lds_barrier
        %493 = vector.load %reinterpret_cast_13[%69] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %494 = vector.load %reinterpret_cast_13[%157] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %495 = vector.load %reinterpret_cast_13[%71] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %496 = vector.load %reinterpret_cast_13[%159] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %497 = vector.load %reinterpret_cast_13[%73] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %498 = vector.load %reinterpret_cast_13[%161] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %499 = vector.load %reinterpret_cast_13[%75] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %500 = vector.load %reinterpret_cast_13[%163] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %501 = vector.load %reinterpret_cast_13[%77] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %502 = vector.load %reinterpret_cast_13[%165] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %503 = vector.load %reinterpret_cast_13[%79] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %504 = vector.load %reinterpret_cast_13[%167] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %505 = vector.load %reinterpret_cast_13[%81] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %506 = vector.load %reinterpret_cast_13[%169] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %507 = vector.load %reinterpret_cast_13[%83] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %508 = vector.load %reinterpret_cast_13[%171] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %509 = vector.load %reinterpret_cast_12[%61] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %510 = vector.load %reinterpret_cast_12[%149] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %511 = vector.load %reinterpret_cast_12[%63] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %512 = vector.load %reinterpret_cast_12[%151] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %513 = vector.load %reinterpret_cast_12[%65] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %514 = vector.load %reinterpret_cast_12[%153] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %515 = vector.load %reinterpret_cast_12[%67] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %516 = vector.load %reinterpret_cast_12[%155] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %517 = vector.bitcast %509 : vector<16xi8> to vector<32xf4E2M1FN>
        %518 = vector.bitcast %510 : vector<16xi8> to vector<32xf4E2M1FN>
        %519 = vector.bitcast %511 : vector<16xi8> to vector<32xf4E2M1FN>
        %520 = vector.bitcast %512 : vector<16xi8> to vector<32xf4E2M1FN>
        %521 = vector.bitcast %513 : vector<16xi8> to vector<32xf4E2M1FN>
        %522 = vector.bitcast %514 : vector<16xi8> to vector<32xf4E2M1FN>
        %523 = vector.bitcast %515 : vector<16xi8> to vector<32xf4E2M1FN>
        %524 = vector.bitcast %516 : vector<16xi8> to vector<32xf4E2M1FN>
        %525 = vector.bitcast %493 : vector<16xi8> to vector<32xf4E2M1FN>
        %526 = vector.bitcast %494 : vector<16xi8> to vector<32xf4E2M1FN>
        %527 = vector.bitcast %495 : vector<16xi8> to vector<32xf4E2M1FN>
        %528 = vector.bitcast %496 : vector<16xi8> to vector<32xf4E2M1FN>
        %529 = vector.bitcast %497 : vector<16xi8> to vector<32xf4E2M1FN>
        %530 = vector.bitcast %498 : vector<16xi8> to vector<32xf4E2M1FN>
        %531 = vector.bitcast %499 : vector<16xi8> to vector<32xf4E2M1FN>
        %532 = vector.bitcast %500 : vector<16xi8> to vector<32xf4E2M1FN>
        %533 = vector.bitcast %501 : vector<16xi8> to vector<32xf4E2M1FN>
        %534 = vector.bitcast %502 : vector<16xi8> to vector<32xf4E2M1FN>
        %535 = vector.bitcast %503 : vector<16xi8> to vector<32xf4E2M1FN>
        %536 = vector.bitcast %504 : vector<16xi8> to vector<32xf4E2M1FN>
        %537 = vector.bitcast %505 : vector<16xi8> to vector<32xf4E2M1FN>
        %538 = vector.bitcast %506 : vector<16xi8> to vector<32xf4E2M1FN>
        %539 = vector.bitcast %507 : vector<16xi8> to vector<32xf4E2M1FN>
        %540 = vector.bitcast %508 : vector<16xi8> to vector<32xf4E2M1FN>
        %541 = amdgpu.scaled_mfma 16x16x128 (%389[0] * %517) * (%395[0] * %525) + %461 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %542 = amdgpu.scaled_mfma 16x16x128 (%389[2] * %518) * (%395[2] * %526) + %541 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %543 = amdgpu.scaled_mfma 16x16x128 (%389[0] * %517) * (%395[1] * %527) + %462 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %544 = amdgpu.scaled_mfma 16x16x128 (%389[2] * %518) * (%395[3] * %528) + %543 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %545 = amdgpu.scaled_mfma 16x16x128 (%389[0] * %517) * (%398[0] * %529) + %463 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %546 = amdgpu.scaled_mfma 16x16x128 (%389[2] * %518) * (%398[2] * %530) + %545 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %547 = amdgpu.scaled_mfma 16x16x128 (%389[0] * %517) * (%398[1] * %531) + %464 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %548 = amdgpu.scaled_mfma 16x16x128 (%389[2] * %518) * (%398[3] * %532) + %547 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %549 = amdgpu.scaled_mfma 16x16x128 (%389[0] * %517) * (%401[0] * %533) + %465 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %550 = amdgpu.scaled_mfma 16x16x128 (%389[2] * %518) * (%401[2] * %534) + %549 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %551 = amdgpu.scaled_mfma 16x16x128 (%389[0] * %517) * (%401[1] * %535) + %466 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %552 = amdgpu.scaled_mfma 16x16x128 (%389[2] * %518) * (%401[3] * %536) + %551 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %553 = amdgpu.scaled_mfma 16x16x128 (%389[0] * %517) * (%404[0] * %537) + %467 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %554 = amdgpu.scaled_mfma 16x16x128 (%389[2] * %518) * (%404[2] * %538) + %553 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %555 = amdgpu.scaled_mfma 16x16x128 (%389[0] * %517) * (%404[1] * %539) + %468 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %556 = amdgpu.scaled_mfma 16x16x128 (%389[2] * %518) * (%404[3] * %540) + %555 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %557 = amdgpu.scaled_mfma 16x16x128 (%389[1] * %519) * (%395[0] * %525) + %469 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %558 = amdgpu.scaled_mfma 16x16x128 (%389[3] * %520) * (%395[2] * %526) + %557 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %559 = amdgpu.scaled_mfma 16x16x128 (%389[1] * %519) * (%395[1] * %527) + %470 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %560 = amdgpu.scaled_mfma 16x16x128 (%389[3] * %520) * (%395[3] * %528) + %559 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %561 = amdgpu.scaled_mfma 16x16x128 (%389[1] * %519) * (%398[0] * %529) + %471 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %562 = amdgpu.scaled_mfma 16x16x128 (%389[3] * %520) * (%398[2] * %530) + %561 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %563 = amdgpu.scaled_mfma 16x16x128 (%389[1] * %519) * (%398[1] * %531) + %472 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %564 = amdgpu.scaled_mfma 16x16x128 (%389[3] * %520) * (%398[3] * %532) + %563 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %565 = amdgpu.scaled_mfma 16x16x128 (%389[1] * %519) * (%401[0] * %533) + %473 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %566 = amdgpu.scaled_mfma 16x16x128 (%389[3] * %520) * (%401[2] * %534) + %565 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %567 = amdgpu.scaled_mfma 16x16x128 (%389[1] * %519) * (%401[1] * %535) + %474 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %568 = amdgpu.scaled_mfma 16x16x128 (%389[3] * %520) * (%401[3] * %536) + %567 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %569 = amdgpu.scaled_mfma 16x16x128 (%389[1] * %519) * (%404[0] * %537) + %475 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %570 = amdgpu.scaled_mfma 16x16x128 (%389[3] * %520) * (%404[2] * %538) + %569 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %571 = amdgpu.scaled_mfma 16x16x128 (%389[1] * %519) * (%404[1] * %539) + %476 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %572 = amdgpu.scaled_mfma 16x16x128 (%389[3] * %520) * (%404[3] * %540) + %571 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %573 = amdgpu.scaled_mfma 16x16x128 (%392[0] * %521) * (%395[0] * %525) + %477 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %574 = amdgpu.scaled_mfma 16x16x128 (%392[2] * %522) * (%395[2] * %526) + %573 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %575 = amdgpu.scaled_mfma 16x16x128 (%392[0] * %521) * (%395[1] * %527) + %478 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %576 = amdgpu.scaled_mfma 16x16x128 (%392[2] * %522) * (%395[3] * %528) + %575 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %577 = amdgpu.scaled_mfma 16x16x128 (%392[0] * %521) * (%398[0] * %529) + %479 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %578 = amdgpu.scaled_mfma 16x16x128 (%392[2] * %522) * (%398[2] * %530) + %577 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %579 = amdgpu.scaled_mfma 16x16x128 (%392[0] * %521) * (%398[1] * %531) + %480 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %580 = amdgpu.scaled_mfma 16x16x128 (%392[2] * %522) * (%398[3] * %532) + %579 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %581 = amdgpu.scaled_mfma 16x16x128 (%392[0] * %521) * (%401[0] * %533) + %481 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %582 = amdgpu.scaled_mfma 16x16x128 (%392[2] * %522) * (%401[2] * %534) + %581 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %583 = amdgpu.scaled_mfma 16x16x128 (%392[0] * %521) * (%401[1] * %535) + %482 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %584 = amdgpu.scaled_mfma 16x16x128 (%392[2] * %522) * (%401[3] * %536) + %583 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %585 = amdgpu.scaled_mfma 16x16x128 (%392[0] * %521) * (%404[0] * %537) + %483 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %586 = amdgpu.scaled_mfma 16x16x128 (%392[2] * %522) * (%404[2] * %538) + %585 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %587 = amdgpu.scaled_mfma 16x16x128 (%392[0] * %521) * (%404[1] * %539) + %484 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %588 = amdgpu.scaled_mfma 16x16x128 (%392[2] * %522) * (%404[3] * %540) + %587 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %589 = amdgpu.scaled_mfma 16x16x128 (%392[1] * %523) * (%395[0] * %525) + %485 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %590 = amdgpu.scaled_mfma 16x16x128 (%392[3] * %524) * (%395[2] * %526) + %589 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %591 = amdgpu.scaled_mfma 16x16x128 (%392[1] * %523) * (%395[1] * %527) + %486 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %592 = amdgpu.scaled_mfma 16x16x128 (%392[3] * %524) * (%395[3] * %528) + %591 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %593 = amdgpu.scaled_mfma 16x16x128 (%392[1] * %523) * (%398[0] * %529) + %487 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %594 = amdgpu.scaled_mfma 16x16x128 (%392[3] * %524) * (%398[2] * %530) + %593 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %595 = amdgpu.scaled_mfma 16x16x128 (%392[1] * %523) * (%398[1] * %531) + %488 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %596 = amdgpu.scaled_mfma 16x16x128 (%392[3] * %524) * (%398[3] * %532) + %595 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %597 = amdgpu.scaled_mfma 16x16x128 (%392[1] * %523) * (%401[0] * %533) + %489 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %598 = amdgpu.scaled_mfma 16x16x128 (%392[3] * %524) * (%401[2] * %534) + %597 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %599 = amdgpu.scaled_mfma 16x16x128 (%392[1] * %523) * (%401[1] * %535) + %490 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %600 = amdgpu.scaled_mfma 16x16x128 (%392[3] * %524) * (%401[3] * %536) + %599 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %601 = amdgpu.scaled_mfma 16x16x128 (%392[1] * %523) * (%404[0] * %537) + %491 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %602 = amdgpu.scaled_mfma 16x16x128 (%392[3] * %524) * (%404[2] * %538) + %601 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %603 = amdgpu.scaled_mfma 16x16x128 (%392[1] * %523) * (%404[1] * %539) + %492 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %604 = amdgpu.scaled_mfma 16x16x128 (%392[3] * %524) * (%404[3] * %540) + %603 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        // ── Optimized epilogue: XOR-1 shuffle → vector<2xbf16> stores ─────
        %ep_block_x = affine.apply #map61()[%block_id_x]
        %ep_block_y = affine.apply #map61()[%block_id_y]
        %ep_base_buf, %ep_offset, %ep_sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<8192x8192xbf16, strided<[?, 1]>> -> memref<bf16>, index, index, index, index, index
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

        %ep_rb0_row0 = affine.apply #map62()[%thread_id_x]
        %ep_rb0_row1 = affine.apply #map64()[%thread_id_x]
        %ep_rb0_row2 = affine.apply #map65()[%thread_id_x]
        %ep_rb0_row3 = affine.apply #map66()[%thread_id_x]
        %ep_rb0_store_row_a = arith.select %ep_is_even, %ep_rb0_row0, %ep_rb0_row2 : index
        %ep_rb0_store_row_b = arith.select %ep_is_even, %ep_rb0_row1, %ep_rb0_row3 : index
        %ep_rb0_store_off_a = arith.muli %ep_rb0_store_row_a, %strides#0 overflow<nsw> : index
        %ep_rb0_store_off_b = arith.muli %ep_rb0_store_row_b, %strides#0 overflow<nsw> : index

        %ep_rb1_row0 = affine.apply #map74()[%thread_id_x]
        %ep_rb1_row1 = affine.apply #map75()[%thread_id_x]
        %ep_rb1_row2 = affine.apply #map76()[%thread_id_x]
        %ep_rb1_row3 = affine.apply #map77()[%thread_id_x]
        %ep_rb1_store_row_a = arith.select %ep_is_even, %ep_rb1_row0, %ep_rb1_row2 : index
        %ep_rb1_store_row_b = arith.select %ep_is_even, %ep_rb1_row1, %ep_rb1_row3 : index
        %ep_rb1_store_off_a = arith.muli %ep_rb1_store_row_a, %strides#0 overflow<nsw> : index
        %ep_rb1_store_off_b = arith.muli %ep_rb1_store_row_b, %strides#0 overflow<nsw> : index

        %ep_rb2_row0 = affine.apply #map78()[%thread_id_x]
        %ep_rb2_row1 = affine.apply #map79()[%thread_id_x]
        %ep_rb2_row2 = affine.apply #map80()[%thread_id_x]
        %ep_rb2_row3 = affine.apply #map81()[%thread_id_x]
        %ep_rb2_store_row_a = arith.select %ep_is_even, %ep_rb2_row0, %ep_rb2_row2 : index
        %ep_rb2_store_row_b = arith.select %ep_is_even, %ep_rb2_row1, %ep_rb2_row3 : index
        %ep_rb2_store_off_a = arith.muli %ep_rb2_store_row_a, %strides#0 overflow<nsw> : index
        %ep_rb2_store_off_b = arith.muli %ep_rb2_store_row_b, %strides#0 overflow<nsw> : index

        %ep_rb3_row0 = affine.apply #map82()[%thread_id_x]
        %ep_rb3_row1 = affine.apply #map83()[%thread_id_x]
        %ep_rb3_row2 = affine.apply #map84()[%thread_id_x]
        %ep_rb3_row3 = affine.apply #map85()[%thread_id_x]
        %ep_rb3_store_row_a = arith.select %ep_is_even, %ep_rb3_row0, %ep_rb3_row2 : index
        %ep_rb3_store_row_b = arith.select %ep_is_even, %ep_rb3_row1, %ep_rb3_row3 : index
        %ep_rb3_store_off_a = arith.muli %ep_rb3_store_row_a, %strides#0 overflow<nsw> : index
        %ep_rb3_store_off_b = arith.muli %ep_rb3_store_row_b, %strides#0 overflow<nsw> : index

        %ep_col0 = affine.apply #map63()[%thread_id_x, %thread_id_y]
        %ep_col0_adj = arith.subi %ep_col0, %ep_lane_parity : index
        %ep_col1 = affine.apply #map67()[%thread_id_x, %thread_id_y]
        %ep_col1_adj = arith.subi %ep_col1, %ep_lane_parity : index
        %ep_col2 = affine.apply #map68()[%thread_id_x, %thread_id_y]
        %ep_col2_adj = arith.subi %ep_col2, %ep_lane_parity : index
        %ep_col3 = affine.apply #map69()[%thread_id_x, %thread_id_y]
        %ep_col3_adj = arith.subi %ep_col3, %ep_lane_parity : index
        %ep_col4 = affine.apply #map70()[%thread_id_x, %thread_id_y]
        %ep_col4_adj = arith.subi %ep_col4, %ep_lane_parity : index
        %ep_col5 = affine.apply #map71()[%thread_id_x, %thread_id_y]
        %ep_col5_adj = arith.subi %ep_col5, %ep_lane_parity : index
        %ep_col6 = affine.apply #map72()[%thread_id_x, %thread_id_y]
        %ep_col6_adj = arith.subi %ep_col6, %ep_lane_parity : index
        %ep_col7 = affine.apply #map73()[%thread_id_x, %thread_id_y]
        %ep_col7_adj = arith.subi %ep_col7, %ep_lane_parity : index

        // MFMA 0: %542 → rb0, col0
        %ep_v542_row0 = vector.extract %542[0] : f32 from vector<4xf32>
        %ep_v542_row1 = vector.extract %542[1] : f32 from vector<4xf32>
        %ep_v542_row2 = vector.extract %542[2] : f32 from vector<4xf32>
        %ep_v542_row3 = vector.extract %542[3] : f32 from vector<4xf32>
        %ep_v542_row0_nbr, %ep_v542_row0_valid = gpu.shuffle xor %ep_v542_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v542_row1_nbr, %ep_v542_row1_valid = gpu.shuffle xor %ep_v542_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v542_row2_nbr, %ep_v542_row2_valid = gpu.shuffle xor %ep_v542_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v542_row3_nbr, %ep_v542_row3_valid = gpu.shuffle xor %ep_v542_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v542_row0_lo = arith.select %ep_is_even, %ep_v542_row0,     %ep_v542_row0_nbr : f32
        %ep_v542_row0_hi = arith.select %ep_is_even, %ep_v542_row0_nbr, %ep_v542_row0     : f32
        %ep_v542_row1_lo = arith.select %ep_is_even, %ep_v542_row1,     %ep_v542_row1_nbr : f32
        %ep_v542_row1_hi = arith.select %ep_is_even, %ep_v542_row1_nbr, %ep_v542_row1     : f32
        %ep_v542_row2_lo = arith.select %ep_is_even, %ep_v542_row2,     %ep_v542_row2_nbr : f32
        %ep_v542_row2_hi = arith.select %ep_is_even, %ep_v542_row2_nbr, %ep_v542_row2     : f32
        %ep_v542_row3_lo = arith.select %ep_is_even, %ep_v542_row3,     %ep_v542_row3_nbr : f32
        %ep_v542_row3_hi = arith.select %ep_is_even, %ep_v542_row3_nbr, %ep_v542_row3     : f32
        %ep_v542_store_a_lo = arith.select %ep_is_even, %ep_v542_row0_lo, %ep_v542_row2_lo : f32
        %ep_v542_store_a_hi = arith.select %ep_is_even, %ep_v542_row0_hi, %ep_v542_row2_hi : f32
        %ep_v542_store_b_lo = arith.select %ep_is_even, %ep_v542_row1_lo, %ep_v542_row3_lo : f32
        %ep_v542_store_b_hi = arith.select %ep_is_even, %ep_v542_row1_hi, %ep_v542_row3_hi : f32
        %ep_v542_pair_a_0 = vector.broadcast %ep_v542_store_a_lo : f32 to vector<2xf32>
        %ep_v542_pair_a   = vector.insert %ep_v542_store_a_hi, %ep_v542_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v542_store_a  = arith.truncf %ep_v542_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v542_pair_b_0 = vector.broadcast %ep_v542_store_b_lo : f32 to vector<2xf32>
        %ep_v542_pair_b   = vector.insert %ep_v542_store_b_hi, %ep_v542_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v542_store_b  = arith.truncf %ep_v542_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v542_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col0_adj overflow<nsw> : index
        %ep_v542_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col0_adj overflow<nsw> : index
        vector.store %ep_v542_store_a, %ep_buffer[%ep_v542_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v542_store_b, %ep_buffer[%ep_v542_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 1: %544 → rb0, col1
        %ep_v544_row0 = vector.extract %544[0] : f32 from vector<4xf32>
        %ep_v544_row1 = vector.extract %544[1] : f32 from vector<4xf32>
        %ep_v544_row2 = vector.extract %544[2] : f32 from vector<4xf32>
        %ep_v544_row3 = vector.extract %544[3] : f32 from vector<4xf32>
        %ep_v544_row0_nbr, %ep_v544_row0_valid = gpu.shuffle xor %ep_v544_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v544_row1_nbr, %ep_v544_row1_valid = gpu.shuffle xor %ep_v544_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v544_row2_nbr, %ep_v544_row2_valid = gpu.shuffle xor %ep_v544_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v544_row3_nbr, %ep_v544_row3_valid = gpu.shuffle xor %ep_v544_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v544_row0_lo = arith.select %ep_is_even, %ep_v544_row0,     %ep_v544_row0_nbr : f32
        %ep_v544_row0_hi = arith.select %ep_is_even, %ep_v544_row0_nbr, %ep_v544_row0     : f32
        %ep_v544_row1_lo = arith.select %ep_is_even, %ep_v544_row1,     %ep_v544_row1_nbr : f32
        %ep_v544_row1_hi = arith.select %ep_is_even, %ep_v544_row1_nbr, %ep_v544_row1     : f32
        %ep_v544_row2_lo = arith.select %ep_is_even, %ep_v544_row2,     %ep_v544_row2_nbr : f32
        %ep_v544_row2_hi = arith.select %ep_is_even, %ep_v544_row2_nbr, %ep_v544_row2     : f32
        %ep_v544_row3_lo = arith.select %ep_is_even, %ep_v544_row3,     %ep_v544_row3_nbr : f32
        %ep_v544_row3_hi = arith.select %ep_is_even, %ep_v544_row3_nbr, %ep_v544_row3     : f32
        %ep_v544_store_a_lo = arith.select %ep_is_even, %ep_v544_row0_lo, %ep_v544_row2_lo : f32
        %ep_v544_store_a_hi = arith.select %ep_is_even, %ep_v544_row0_hi, %ep_v544_row2_hi : f32
        %ep_v544_store_b_lo = arith.select %ep_is_even, %ep_v544_row1_lo, %ep_v544_row3_lo : f32
        %ep_v544_store_b_hi = arith.select %ep_is_even, %ep_v544_row1_hi, %ep_v544_row3_hi : f32
        %ep_v544_pair_a_0 = vector.broadcast %ep_v544_store_a_lo : f32 to vector<2xf32>
        %ep_v544_pair_a   = vector.insert %ep_v544_store_a_hi, %ep_v544_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v544_store_a  = arith.truncf %ep_v544_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v544_pair_b_0 = vector.broadcast %ep_v544_store_b_lo : f32 to vector<2xf32>
        %ep_v544_pair_b   = vector.insert %ep_v544_store_b_hi, %ep_v544_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v544_store_b  = arith.truncf %ep_v544_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v544_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col1_adj overflow<nsw> : index
        %ep_v544_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col1_adj overflow<nsw> : index
        vector.store %ep_v544_store_a, %ep_buffer[%ep_v544_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v544_store_b, %ep_buffer[%ep_v544_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 2: %546 → rb0, col2
        %ep_v546_row0 = vector.extract %546[0] : f32 from vector<4xf32>
        %ep_v546_row1 = vector.extract %546[1] : f32 from vector<4xf32>
        %ep_v546_row2 = vector.extract %546[2] : f32 from vector<4xf32>
        %ep_v546_row3 = vector.extract %546[3] : f32 from vector<4xf32>
        %ep_v546_row0_nbr, %ep_v546_row0_valid = gpu.shuffle xor %ep_v546_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v546_row1_nbr, %ep_v546_row1_valid = gpu.shuffle xor %ep_v546_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v546_row2_nbr, %ep_v546_row2_valid = gpu.shuffle xor %ep_v546_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v546_row3_nbr, %ep_v546_row3_valid = gpu.shuffle xor %ep_v546_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v546_row0_lo = arith.select %ep_is_even, %ep_v546_row0,     %ep_v546_row0_nbr : f32
        %ep_v546_row0_hi = arith.select %ep_is_even, %ep_v546_row0_nbr, %ep_v546_row0     : f32
        %ep_v546_row1_lo = arith.select %ep_is_even, %ep_v546_row1,     %ep_v546_row1_nbr : f32
        %ep_v546_row1_hi = arith.select %ep_is_even, %ep_v546_row1_nbr, %ep_v546_row1     : f32
        %ep_v546_row2_lo = arith.select %ep_is_even, %ep_v546_row2,     %ep_v546_row2_nbr : f32
        %ep_v546_row2_hi = arith.select %ep_is_even, %ep_v546_row2_nbr, %ep_v546_row2     : f32
        %ep_v546_row3_lo = arith.select %ep_is_even, %ep_v546_row3,     %ep_v546_row3_nbr : f32
        %ep_v546_row3_hi = arith.select %ep_is_even, %ep_v546_row3_nbr, %ep_v546_row3     : f32
        %ep_v546_store_a_lo = arith.select %ep_is_even, %ep_v546_row0_lo, %ep_v546_row2_lo : f32
        %ep_v546_store_a_hi = arith.select %ep_is_even, %ep_v546_row0_hi, %ep_v546_row2_hi : f32
        %ep_v546_store_b_lo = arith.select %ep_is_even, %ep_v546_row1_lo, %ep_v546_row3_lo : f32
        %ep_v546_store_b_hi = arith.select %ep_is_even, %ep_v546_row1_hi, %ep_v546_row3_hi : f32
        %ep_v546_pair_a_0 = vector.broadcast %ep_v546_store_a_lo : f32 to vector<2xf32>
        %ep_v546_pair_a   = vector.insert %ep_v546_store_a_hi, %ep_v546_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v546_store_a  = arith.truncf %ep_v546_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v546_pair_b_0 = vector.broadcast %ep_v546_store_b_lo : f32 to vector<2xf32>
        %ep_v546_pair_b   = vector.insert %ep_v546_store_b_hi, %ep_v546_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v546_store_b  = arith.truncf %ep_v546_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v546_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col2_adj overflow<nsw> : index
        %ep_v546_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col2_adj overflow<nsw> : index
        vector.store %ep_v546_store_a, %ep_buffer[%ep_v546_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v546_store_b, %ep_buffer[%ep_v546_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 3: %548 → rb0, col3
        %ep_v548_row0 = vector.extract %548[0] : f32 from vector<4xf32>
        %ep_v548_row1 = vector.extract %548[1] : f32 from vector<4xf32>
        %ep_v548_row2 = vector.extract %548[2] : f32 from vector<4xf32>
        %ep_v548_row3 = vector.extract %548[3] : f32 from vector<4xf32>
        %ep_v548_row0_nbr, %ep_v548_row0_valid = gpu.shuffle xor %ep_v548_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v548_row1_nbr, %ep_v548_row1_valid = gpu.shuffle xor %ep_v548_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v548_row2_nbr, %ep_v548_row2_valid = gpu.shuffle xor %ep_v548_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v548_row3_nbr, %ep_v548_row3_valid = gpu.shuffle xor %ep_v548_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v548_row0_lo = arith.select %ep_is_even, %ep_v548_row0,     %ep_v548_row0_nbr : f32
        %ep_v548_row0_hi = arith.select %ep_is_even, %ep_v548_row0_nbr, %ep_v548_row0     : f32
        %ep_v548_row1_lo = arith.select %ep_is_even, %ep_v548_row1,     %ep_v548_row1_nbr : f32
        %ep_v548_row1_hi = arith.select %ep_is_even, %ep_v548_row1_nbr, %ep_v548_row1     : f32
        %ep_v548_row2_lo = arith.select %ep_is_even, %ep_v548_row2,     %ep_v548_row2_nbr : f32
        %ep_v548_row2_hi = arith.select %ep_is_even, %ep_v548_row2_nbr, %ep_v548_row2     : f32
        %ep_v548_row3_lo = arith.select %ep_is_even, %ep_v548_row3,     %ep_v548_row3_nbr : f32
        %ep_v548_row3_hi = arith.select %ep_is_even, %ep_v548_row3_nbr, %ep_v548_row3     : f32
        %ep_v548_store_a_lo = arith.select %ep_is_even, %ep_v548_row0_lo, %ep_v548_row2_lo : f32
        %ep_v548_store_a_hi = arith.select %ep_is_even, %ep_v548_row0_hi, %ep_v548_row2_hi : f32
        %ep_v548_store_b_lo = arith.select %ep_is_even, %ep_v548_row1_lo, %ep_v548_row3_lo : f32
        %ep_v548_store_b_hi = arith.select %ep_is_even, %ep_v548_row1_hi, %ep_v548_row3_hi : f32
        %ep_v548_pair_a_0 = vector.broadcast %ep_v548_store_a_lo : f32 to vector<2xf32>
        %ep_v548_pair_a   = vector.insert %ep_v548_store_a_hi, %ep_v548_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v548_store_a  = arith.truncf %ep_v548_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v548_pair_b_0 = vector.broadcast %ep_v548_store_b_lo : f32 to vector<2xf32>
        %ep_v548_pair_b   = vector.insert %ep_v548_store_b_hi, %ep_v548_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v548_store_b  = arith.truncf %ep_v548_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v548_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col3_adj overflow<nsw> : index
        %ep_v548_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col3_adj overflow<nsw> : index
        vector.store %ep_v548_store_a, %ep_buffer[%ep_v548_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v548_store_b, %ep_buffer[%ep_v548_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 4: %550 → rb0, col4
        %ep_v550_row0 = vector.extract %550[0] : f32 from vector<4xf32>
        %ep_v550_row1 = vector.extract %550[1] : f32 from vector<4xf32>
        %ep_v550_row2 = vector.extract %550[2] : f32 from vector<4xf32>
        %ep_v550_row3 = vector.extract %550[3] : f32 from vector<4xf32>
        %ep_v550_row0_nbr, %ep_v550_row0_valid = gpu.shuffle xor %ep_v550_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v550_row1_nbr, %ep_v550_row1_valid = gpu.shuffle xor %ep_v550_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v550_row2_nbr, %ep_v550_row2_valid = gpu.shuffle xor %ep_v550_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v550_row3_nbr, %ep_v550_row3_valid = gpu.shuffle xor %ep_v550_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v550_row0_lo = arith.select %ep_is_even, %ep_v550_row0,     %ep_v550_row0_nbr : f32
        %ep_v550_row0_hi = arith.select %ep_is_even, %ep_v550_row0_nbr, %ep_v550_row0     : f32
        %ep_v550_row1_lo = arith.select %ep_is_even, %ep_v550_row1,     %ep_v550_row1_nbr : f32
        %ep_v550_row1_hi = arith.select %ep_is_even, %ep_v550_row1_nbr, %ep_v550_row1     : f32
        %ep_v550_row2_lo = arith.select %ep_is_even, %ep_v550_row2,     %ep_v550_row2_nbr : f32
        %ep_v550_row2_hi = arith.select %ep_is_even, %ep_v550_row2_nbr, %ep_v550_row2     : f32
        %ep_v550_row3_lo = arith.select %ep_is_even, %ep_v550_row3,     %ep_v550_row3_nbr : f32
        %ep_v550_row3_hi = arith.select %ep_is_even, %ep_v550_row3_nbr, %ep_v550_row3     : f32
        %ep_v550_store_a_lo = arith.select %ep_is_even, %ep_v550_row0_lo, %ep_v550_row2_lo : f32
        %ep_v550_store_a_hi = arith.select %ep_is_even, %ep_v550_row0_hi, %ep_v550_row2_hi : f32
        %ep_v550_store_b_lo = arith.select %ep_is_even, %ep_v550_row1_lo, %ep_v550_row3_lo : f32
        %ep_v550_store_b_hi = arith.select %ep_is_even, %ep_v550_row1_hi, %ep_v550_row3_hi : f32
        %ep_v550_pair_a_0 = vector.broadcast %ep_v550_store_a_lo : f32 to vector<2xf32>
        %ep_v550_pair_a   = vector.insert %ep_v550_store_a_hi, %ep_v550_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v550_store_a  = arith.truncf %ep_v550_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v550_pair_b_0 = vector.broadcast %ep_v550_store_b_lo : f32 to vector<2xf32>
        %ep_v550_pair_b   = vector.insert %ep_v550_store_b_hi, %ep_v550_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v550_store_b  = arith.truncf %ep_v550_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v550_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col4_adj overflow<nsw> : index
        %ep_v550_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col4_adj overflow<nsw> : index
        vector.store %ep_v550_store_a, %ep_buffer[%ep_v550_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v550_store_b, %ep_buffer[%ep_v550_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 5: %552 → rb0, col5
        %ep_v552_row0 = vector.extract %552[0] : f32 from vector<4xf32>
        %ep_v552_row1 = vector.extract %552[1] : f32 from vector<4xf32>
        %ep_v552_row2 = vector.extract %552[2] : f32 from vector<4xf32>
        %ep_v552_row3 = vector.extract %552[3] : f32 from vector<4xf32>
        %ep_v552_row0_nbr, %ep_v552_row0_valid = gpu.shuffle xor %ep_v552_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v552_row1_nbr, %ep_v552_row1_valid = gpu.shuffle xor %ep_v552_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v552_row2_nbr, %ep_v552_row2_valid = gpu.shuffle xor %ep_v552_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v552_row3_nbr, %ep_v552_row3_valid = gpu.shuffle xor %ep_v552_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v552_row0_lo = arith.select %ep_is_even, %ep_v552_row0,     %ep_v552_row0_nbr : f32
        %ep_v552_row0_hi = arith.select %ep_is_even, %ep_v552_row0_nbr, %ep_v552_row0     : f32
        %ep_v552_row1_lo = arith.select %ep_is_even, %ep_v552_row1,     %ep_v552_row1_nbr : f32
        %ep_v552_row1_hi = arith.select %ep_is_even, %ep_v552_row1_nbr, %ep_v552_row1     : f32
        %ep_v552_row2_lo = arith.select %ep_is_even, %ep_v552_row2,     %ep_v552_row2_nbr : f32
        %ep_v552_row2_hi = arith.select %ep_is_even, %ep_v552_row2_nbr, %ep_v552_row2     : f32
        %ep_v552_row3_lo = arith.select %ep_is_even, %ep_v552_row3,     %ep_v552_row3_nbr : f32
        %ep_v552_row3_hi = arith.select %ep_is_even, %ep_v552_row3_nbr, %ep_v552_row3     : f32
        %ep_v552_store_a_lo = arith.select %ep_is_even, %ep_v552_row0_lo, %ep_v552_row2_lo : f32
        %ep_v552_store_a_hi = arith.select %ep_is_even, %ep_v552_row0_hi, %ep_v552_row2_hi : f32
        %ep_v552_store_b_lo = arith.select %ep_is_even, %ep_v552_row1_lo, %ep_v552_row3_lo : f32
        %ep_v552_store_b_hi = arith.select %ep_is_even, %ep_v552_row1_hi, %ep_v552_row3_hi : f32
        %ep_v552_pair_a_0 = vector.broadcast %ep_v552_store_a_lo : f32 to vector<2xf32>
        %ep_v552_pair_a   = vector.insert %ep_v552_store_a_hi, %ep_v552_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v552_store_a  = arith.truncf %ep_v552_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v552_pair_b_0 = vector.broadcast %ep_v552_store_b_lo : f32 to vector<2xf32>
        %ep_v552_pair_b   = vector.insert %ep_v552_store_b_hi, %ep_v552_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v552_store_b  = arith.truncf %ep_v552_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v552_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col5_adj overflow<nsw> : index
        %ep_v552_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col5_adj overflow<nsw> : index
        vector.store %ep_v552_store_a, %ep_buffer[%ep_v552_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v552_store_b, %ep_buffer[%ep_v552_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 6: %554 → rb0, col6
        %ep_v554_row0 = vector.extract %554[0] : f32 from vector<4xf32>
        %ep_v554_row1 = vector.extract %554[1] : f32 from vector<4xf32>
        %ep_v554_row2 = vector.extract %554[2] : f32 from vector<4xf32>
        %ep_v554_row3 = vector.extract %554[3] : f32 from vector<4xf32>
        %ep_v554_row0_nbr, %ep_v554_row0_valid = gpu.shuffle xor %ep_v554_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v554_row1_nbr, %ep_v554_row1_valid = gpu.shuffle xor %ep_v554_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v554_row2_nbr, %ep_v554_row2_valid = gpu.shuffle xor %ep_v554_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v554_row3_nbr, %ep_v554_row3_valid = gpu.shuffle xor %ep_v554_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v554_row0_lo = arith.select %ep_is_even, %ep_v554_row0,     %ep_v554_row0_nbr : f32
        %ep_v554_row0_hi = arith.select %ep_is_even, %ep_v554_row0_nbr, %ep_v554_row0     : f32
        %ep_v554_row1_lo = arith.select %ep_is_even, %ep_v554_row1,     %ep_v554_row1_nbr : f32
        %ep_v554_row1_hi = arith.select %ep_is_even, %ep_v554_row1_nbr, %ep_v554_row1     : f32
        %ep_v554_row2_lo = arith.select %ep_is_even, %ep_v554_row2,     %ep_v554_row2_nbr : f32
        %ep_v554_row2_hi = arith.select %ep_is_even, %ep_v554_row2_nbr, %ep_v554_row2     : f32
        %ep_v554_row3_lo = arith.select %ep_is_even, %ep_v554_row3,     %ep_v554_row3_nbr : f32
        %ep_v554_row3_hi = arith.select %ep_is_even, %ep_v554_row3_nbr, %ep_v554_row3     : f32
        %ep_v554_store_a_lo = arith.select %ep_is_even, %ep_v554_row0_lo, %ep_v554_row2_lo : f32
        %ep_v554_store_a_hi = arith.select %ep_is_even, %ep_v554_row0_hi, %ep_v554_row2_hi : f32
        %ep_v554_store_b_lo = arith.select %ep_is_even, %ep_v554_row1_lo, %ep_v554_row3_lo : f32
        %ep_v554_store_b_hi = arith.select %ep_is_even, %ep_v554_row1_hi, %ep_v554_row3_hi : f32
        %ep_v554_pair_a_0 = vector.broadcast %ep_v554_store_a_lo : f32 to vector<2xf32>
        %ep_v554_pair_a   = vector.insert %ep_v554_store_a_hi, %ep_v554_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v554_store_a  = arith.truncf %ep_v554_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v554_pair_b_0 = vector.broadcast %ep_v554_store_b_lo : f32 to vector<2xf32>
        %ep_v554_pair_b   = vector.insert %ep_v554_store_b_hi, %ep_v554_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v554_store_b  = arith.truncf %ep_v554_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v554_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col6_adj overflow<nsw> : index
        %ep_v554_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col6_adj overflow<nsw> : index
        vector.store %ep_v554_store_a, %ep_buffer[%ep_v554_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v554_store_b, %ep_buffer[%ep_v554_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 7: %556 → rb0, col7
        %ep_v556_row0 = vector.extract %556[0] : f32 from vector<4xf32>
        %ep_v556_row1 = vector.extract %556[1] : f32 from vector<4xf32>
        %ep_v556_row2 = vector.extract %556[2] : f32 from vector<4xf32>
        %ep_v556_row3 = vector.extract %556[3] : f32 from vector<4xf32>
        %ep_v556_row0_nbr, %ep_v556_row0_valid = gpu.shuffle xor %ep_v556_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v556_row1_nbr, %ep_v556_row1_valid = gpu.shuffle xor %ep_v556_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v556_row2_nbr, %ep_v556_row2_valid = gpu.shuffle xor %ep_v556_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v556_row3_nbr, %ep_v556_row3_valid = gpu.shuffle xor %ep_v556_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v556_row0_lo = arith.select %ep_is_even, %ep_v556_row0,     %ep_v556_row0_nbr : f32
        %ep_v556_row0_hi = arith.select %ep_is_even, %ep_v556_row0_nbr, %ep_v556_row0     : f32
        %ep_v556_row1_lo = arith.select %ep_is_even, %ep_v556_row1,     %ep_v556_row1_nbr : f32
        %ep_v556_row1_hi = arith.select %ep_is_even, %ep_v556_row1_nbr, %ep_v556_row1     : f32
        %ep_v556_row2_lo = arith.select %ep_is_even, %ep_v556_row2,     %ep_v556_row2_nbr : f32
        %ep_v556_row2_hi = arith.select %ep_is_even, %ep_v556_row2_nbr, %ep_v556_row2     : f32
        %ep_v556_row3_lo = arith.select %ep_is_even, %ep_v556_row3,     %ep_v556_row3_nbr : f32
        %ep_v556_row3_hi = arith.select %ep_is_even, %ep_v556_row3_nbr, %ep_v556_row3     : f32
        %ep_v556_store_a_lo = arith.select %ep_is_even, %ep_v556_row0_lo, %ep_v556_row2_lo : f32
        %ep_v556_store_a_hi = arith.select %ep_is_even, %ep_v556_row0_hi, %ep_v556_row2_hi : f32
        %ep_v556_store_b_lo = arith.select %ep_is_even, %ep_v556_row1_lo, %ep_v556_row3_lo : f32
        %ep_v556_store_b_hi = arith.select %ep_is_even, %ep_v556_row1_hi, %ep_v556_row3_hi : f32
        %ep_v556_pair_a_0 = vector.broadcast %ep_v556_store_a_lo : f32 to vector<2xf32>
        %ep_v556_pair_a   = vector.insert %ep_v556_store_a_hi, %ep_v556_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v556_store_a  = arith.truncf %ep_v556_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v556_pair_b_0 = vector.broadcast %ep_v556_store_b_lo : f32 to vector<2xf32>
        %ep_v556_pair_b   = vector.insert %ep_v556_store_b_hi, %ep_v556_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v556_store_b  = arith.truncf %ep_v556_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v556_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col7_adj overflow<nsw> : index
        %ep_v556_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col7_adj overflow<nsw> : index
        vector.store %ep_v556_store_a, %ep_buffer[%ep_v556_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v556_store_b, %ep_buffer[%ep_v556_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 8: %558 → rb1, col0
        %ep_v558_row0 = vector.extract %558[0] : f32 from vector<4xf32>
        %ep_v558_row1 = vector.extract %558[1] : f32 from vector<4xf32>
        %ep_v558_row2 = vector.extract %558[2] : f32 from vector<4xf32>
        %ep_v558_row3 = vector.extract %558[3] : f32 from vector<4xf32>
        %ep_v558_row0_nbr, %ep_v558_row0_valid = gpu.shuffle xor %ep_v558_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v558_row1_nbr, %ep_v558_row1_valid = gpu.shuffle xor %ep_v558_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v558_row2_nbr, %ep_v558_row2_valid = gpu.shuffle xor %ep_v558_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v558_row3_nbr, %ep_v558_row3_valid = gpu.shuffle xor %ep_v558_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v558_row0_lo = arith.select %ep_is_even, %ep_v558_row0,     %ep_v558_row0_nbr : f32
        %ep_v558_row0_hi = arith.select %ep_is_even, %ep_v558_row0_nbr, %ep_v558_row0     : f32
        %ep_v558_row1_lo = arith.select %ep_is_even, %ep_v558_row1,     %ep_v558_row1_nbr : f32
        %ep_v558_row1_hi = arith.select %ep_is_even, %ep_v558_row1_nbr, %ep_v558_row1     : f32
        %ep_v558_row2_lo = arith.select %ep_is_even, %ep_v558_row2,     %ep_v558_row2_nbr : f32
        %ep_v558_row2_hi = arith.select %ep_is_even, %ep_v558_row2_nbr, %ep_v558_row2     : f32
        %ep_v558_row3_lo = arith.select %ep_is_even, %ep_v558_row3,     %ep_v558_row3_nbr : f32
        %ep_v558_row3_hi = arith.select %ep_is_even, %ep_v558_row3_nbr, %ep_v558_row3     : f32
        %ep_v558_store_a_lo = arith.select %ep_is_even, %ep_v558_row0_lo, %ep_v558_row2_lo : f32
        %ep_v558_store_a_hi = arith.select %ep_is_even, %ep_v558_row0_hi, %ep_v558_row2_hi : f32
        %ep_v558_store_b_lo = arith.select %ep_is_even, %ep_v558_row1_lo, %ep_v558_row3_lo : f32
        %ep_v558_store_b_hi = arith.select %ep_is_even, %ep_v558_row1_hi, %ep_v558_row3_hi : f32
        %ep_v558_pair_a_0 = vector.broadcast %ep_v558_store_a_lo : f32 to vector<2xf32>
        %ep_v558_pair_a   = vector.insert %ep_v558_store_a_hi, %ep_v558_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v558_store_a  = arith.truncf %ep_v558_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v558_pair_b_0 = vector.broadcast %ep_v558_store_b_lo : f32 to vector<2xf32>
        %ep_v558_pair_b   = vector.insert %ep_v558_store_b_hi, %ep_v558_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v558_store_b  = arith.truncf %ep_v558_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v558_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col0_adj overflow<nsw> : index
        %ep_v558_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col0_adj overflow<nsw> : index
        vector.store %ep_v558_store_a, %ep_buffer[%ep_v558_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v558_store_b, %ep_buffer[%ep_v558_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 9: %560 → rb1, col1
        %ep_v560_row0 = vector.extract %560[0] : f32 from vector<4xf32>
        %ep_v560_row1 = vector.extract %560[1] : f32 from vector<4xf32>
        %ep_v560_row2 = vector.extract %560[2] : f32 from vector<4xf32>
        %ep_v560_row3 = vector.extract %560[3] : f32 from vector<4xf32>
        %ep_v560_row0_nbr, %ep_v560_row0_valid = gpu.shuffle xor %ep_v560_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v560_row1_nbr, %ep_v560_row1_valid = gpu.shuffle xor %ep_v560_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v560_row2_nbr, %ep_v560_row2_valid = gpu.shuffle xor %ep_v560_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v560_row3_nbr, %ep_v560_row3_valid = gpu.shuffle xor %ep_v560_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v560_row0_lo = arith.select %ep_is_even, %ep_v560_row0,     %ep_v560_row0_nbr : f32
        %ep_v560_row0_hi = arith.select %ep_is_even, %ep_v560_row0_nbr, %ep_v560_row0     : f32
        %ep_v560_row1_lo = arith.select %ep_is_even, %ep_v560_row1,     %ep_v560_row1_nbr : f32
        %ep_v560_row1_hi = arith.select %ep_is_even, %ep_v560_row1_nbr, %ep_v560_row1     : f32
        %ep_v560_row2_lo = arith.select %ep_is_even, %ep_v560_row2,     %ep_v560_row2_nbr : f32
        %ep_v560_row2_hi = arith.select %ep_is_even, %ep_v560_row2_nbr, %ep_v560_row2     : f32
        %ep_v560_row3_lo = arith.select %ep_is_even, %ep_v560_row3,     %ep_v560_row3_nbr : f32
        %ep_v560_row3_hi = arith.select %ep_is_even, %ep_v560_row3_nbr, %ep_v560_row3     : f32
        %ep_v560_store_a_lo = arith.select %ep_is_even, %ep_v560_row0_lo, %ep_v560_row2_lo : f32
        %ep_v560_store_a_hi = arith.select %ep_is_even, %ep_v560_row0_hi, %ep_v560_row2_hi : f32
        %ep_v560_store_b_lo = arith.select %ep_is_even, %ep_v560_row1_lo, %ep_v560_row3_lo : f32
        %ep_v560_store_b_hi = arith.select %ep_is_even, %ep_v560_row1_hi, %ep_v560_row3_hi : f32
        %ep_v560_pair_a_0 = vector.broadcast %ep_v560_store_a_lo : f32 to vector<2xf32>
        %ep_v560_pair_a   = vector.insert %ep_v560_store_a_hi, %ep_v560_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v560_store_a  = arith.truncf %ep_v560_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v560_pair_b_0 = vector.broadcast %ep_v560_store_b_lo : f32 to vector<2xf32>
        %ep_v560_pair_b   = vector.insert %ep_v560_store_b_hi, %ep_v560_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v560_store_b  = arith.truncf %ep_v560_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v560_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col1_adj overflow<nsw> : index
        %ep_v560_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col1_adj overflow<nsw> : index
        vector.store %ep_v560_store_a, %ep_buffer[%ep_v560_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v560_store_b, %ep_buffer[%ep_v560_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 10: %562 → rb1, col2
        %ep_v562_row0 = vector.extract %562[0] : f32 from vector<4xf32>
        %ep_v562_row1 = vector.extract %562[1] : f32 from vector<4xf32>
        %ep_v562_row2 = vector.extract %562[2] : f32 from vector<4xf32>
        %ep_v562_row3 = vector.extract %562[3] : f32 from vector<4xf32>
        %ep_v562_row0_nbr, %ep_v562_row0_valid = gpu.shuffle xor %ep_v562_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v562_row1_nbr, %ep_v562_row1_valid = gpu.shuffle xor %ep_v562_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v562_row2_nbr, %ep_v562_row2_valid = gpu.shuffle xor %ep_v562_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v562_row3_nbr, %ep_v562_row3_valid = gpu.shuffle xor %ep_v562_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v562_row0_lo = arith.select %ep_is_even, %ep_v562_row0,     %ep_v562_row0_nbr : f32
        %ep_v562_row0_hi = arith.select %ep_is_even, %ep_v562_row0_nbr, %ep_v562_row0     : f32
        %ep_v562_row1_lo = arith.select %ep_is_even, %ep_v562_row1,     %ep_v562_row1_nbr : f32
        %ep_v562_row1_hi = arith.select %ep_is_even, %ep_v562_row1_nbr, %ep_v562_row1     : f32
        %ep_v562_row2_lo = arith.select %ep_is_even, %ep_v562_row2,     %ep_v562_row2_nbr : f32
        %ep_v562_row2_hi = arith.select %ep_is_even, %ep_v562_row2_nbr, %ep_v562_row2     : f32
        %ep_v562_row3_lo = arith.select %ep_is_even, %ep_v562_row3,     %ep_v562_row3_nbr : f32
        %ep_v562_row3_hi = arith.select %ep_is_even, %ep_v562_row3_nbr, %ep_v562_row3     : f32
        %ep_v562_store_a_lo = arith.select %ep_is_even, %ep_v562_row0_lo, %ep_v562_row2_lo : f32
        %ep_v562_store_a_hi = arith.select %ep_is_even, %ep_v562_row0_hi, %ep_v562_row2_hi : f32
        %ep_v562_store_b_lo = arith.select %ep_is_even, %ep_v562_row1_lo, %ep_v562_row3_lo : f32
        %ep_v562_store_b_hi = arith.select %ep_is_even, %ep_v562_row1_hi, %ep_v562_row3_hi : f32
        %ep_v562_pair_a_0 = vector.broadcast %ep_v562_store_a_lo : f32 to vector<2xf32>
        %ep_v562_pair_a   = vector.insert %ep_v562_store_a_hi, %ep_v562_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v562_store_a  = arith.truncf %ep_v562_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v562_pair_b_0 = vector.broadcast %ep_v562_store_b_lo : f32 to vector<2xf32>
        %ep_v562_pair_b   = vector.insert %ep_v562_store_b_hi, %ep_v562_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v562_store_b  = arith.truncf %ep_v562_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v562_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col2_adj overflow<nsw> : index
        %ep_v562_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col2_adj overflow<nsw> : index
        vector.store %ep_v562_store_a, %ep_buffer[%ep_v562_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v562_store_b, %ep_buffer[%ep_v562_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 11: %564 → rb1, col3
        %ep_v564_row0 = vector.extract %564[0] : f32 from vector<4xf32>
        %ep_v564_row1 = vector.extract %564[1] : f32 from vector<4xf32>
        %ep_v564_row2 = vector.extract %564[2] : f32 from vector<4xf32>
        %ep_v564_row3 = vector.extract %564[3] : f32 from vector<4xf32>
        %ep_v564_row0_nbr, %ep_v564_row0_valid = gpu.shuffle xor %ep_v564_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v564_row1_nbr, %ep_v564_row1_valid = gpu.shuffle xor %ep_v564_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v564_row2_nbr, %ep_v564_row2_valid = gpu.shuffle xor %ep_v564_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v564_row3_nbr, %ep_v564_row3_valid = gpu.shuffle xor %ep_v564_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v564_row0_lo = arith.select %ep_is_even, %ep_v564_row0,     %ep_v564_row0_nbr : f32
        %ep_v564_row0_hi = arith.select %ep_is_even, %ep_v564_row0_nbr, %ep_v564_row0     : f32
        %ep_v564_row1_lo = arith.select %ep_is_even, %ep_v564_row1,     %ep_v564_row1_nbr : f32
        %ep_v564_row1_hi = arith.select %ep_is_even, %ep_v564_row1_nbr, %ep_v564_row1     : f32
        %ep_v564_row2_lo = arith.select %ep_is_even, %ep_v564_row2,     %ep_v564_row2_nbr : f32
        %ep_v564_row2_hi = arith.select %ep_is_even, %ep_v564_row2_nbr, %ep_v564_row2     : f32
        %ep_v564_row3_lo = arith.select %ep_is_even, %ep_v564_row3,     %ep_v564_row3_nbr : f32
        %ep_v564_row3_hi = arith.select %ep_is_even, %ep_v564_row3_nbr, %ep_v564_row3     : f32
        %ep_v564_store_a_lo = arith.select %ep_is_even, %ep_v564_row0_lo, %ep_v564_row2_lo : f32
        %ep_v564_store_a_hi = arith.select %ep_is_even, %ep_v564_row0_hi, %ep_v564_row2_hi : f32
        %ep_v564_store_b_lo = arith.select %ep_is_even, %ep_v564_row1_lo, %ep_v564_row3_lo : f32
        %ep_v564_store_b_hi = arith.select %ep_is_even, %ep_v564_row1_hi, %ep_v564_row3_hi : f32
        %ep_v564_pair_a_0 = vector.broadcast %ep_v564_store_a_lo : f32 to vector<2xf32>
        %ep_v564_pair_a   = vector.insert %ep_v564_store_a_hi, %ep_v564_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v564_store_a  = arith.truncf %ep_v564_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v564_pair_b_0 = vector.broadcast %ep_v564_store_b_lo : f32 to vector<2xf32>
        %ep_v564_pair_b   = vector.insert %ep_v564_store_b_hi, %ep_v564_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v564_store_b  = arith.truncf %ep_v564_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v564_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col3_adj overflow<nsw> : index
        %ep_v564_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col3_adj overflow<nsw> : index
        vector.store %ep_v564_store_a, %ep_buffer[%ep_v564_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v564_store_b, %ep_buffer[%ep_v564_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 12: %566 → rb1, col4
        %ep_v566_row0 = vector.extract %566[0] : f32 from vector<4xf32>
        %ep_v566_row1 = vector.extract %566[1] : f32 from vector<4xf32>
        %ep_v566_row2 = vector.extract %566[2] : f32 from vector<4xf32>
        %ep_v566_row3 = vector.extract %566[3] : f32 from vector<4xf32>
        %ep_v566_row0_nbr, %ep_v566_row0_valid = gpu.shuffle xor %ep_v566_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v566_row1_nbr, %ep_v566_row1_valid = gpu.shuffle xor %ep_v566_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v566_row2_nbr, %ep_v566_row2_valid = gpu.shuffle xor %ep_v566_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v566_row3_nbr, %ep_v566_row3_valid = gpu.shuffle xor %ep_v566_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v566_row0_lo = arith.select %ep_is_even, %ep_v566_row0,     %ep_v566_row0_nbr : f32
        %ep_v566_row0_hi = arith.select %ep_is_even, %ep_v566_row0_nbr, %ep_v566_row0     : f32
        %ep_v566_row1_lo = arith.select %ep_is_even, %ep_v566_row1,     %ep_v566_row1_nbr : f32
        %ep_v566_row1_hi = arith.select %ep_is_even, %ep_v566_row1_nbr, %ep_v566_row1     : f32
        %ep_v566_row2_lo = arith.select %ep_is_even, %ep_v566_row2,     %ep_v566_row2_nbr : f32
        %ep_v566_row2_hi = arith.select %ep_is_even, %ep_v566_row2_nbr, %ep_v566_row2     : f32
        %ep_v566_row3_lo = arith.select %ep_is_even, %ep_v566_row3,     %ep_v566_row3_nbr : f32
        %ep_v566_row3_hi = arith.select %ep_is_even, %ep_v566_row3_nbr, %ep_v566_row3     : f32
        %ep_v566_store_a_lo = arith.select %ep_is_even, %ep_v566_row0_lo, %ep_v566_row2_lo : f32
        %ep_v566_store_a_hi = arith.select %ep_is_even, %ep_v566_row0_hi, %ep_v566_row2_hi : f32
        %ep_v566_store_b_lo = arith.select %ep_is_even, %ep_v566_row1_lo, %ep_v566_row3_lo : f32
        %ep_v566_store_b_hi = arith.select %ep_is_even, %ep_v566_row1_hi, %ep_v566_row3_hi : f32
        %ep_v566_pair_a_0 = vector.broadcast %ep_v566_store_a_lo : f32 to vector<2xf32>
        %ep_v566_pair_a   = vector.insert %ep_v566_store_a_hi, %ep_v566_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v566_store_a  = arith.truncf %ep_v566_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v566_pair_b_0 = vector.broadcast %ep_v566_store_b_lo : f32 to vector<2xf32>
        %ep_v566_pair_b   = vector.insert %ep_v566_store_b_hi, %ep_v566_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v566_store_b  = arith.truncf %ep_v566_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v566_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col4_adj overflow<nsw> : index
        %ep_v566_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col4_adj overflow<nsw> : index
        vector.store %ep_v566_store_a, %ep_buffer[%ep_v566_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v566_store_b, %ep_buffer[%ep_v566_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 13: %568 → rb1, col5
        %ep_v568_row0 = vector.extract %568[0] : f32 from vector<4xf32>
        %ep_v568_row1 = vector.extract %568[1] : f32 from vector<4xf32>
        %ep_v568_row2 = vector.extract %568[2] : f32 from vector<4xf32>
        %ep_v568_row3 = vector.extract %568[3] : f32 from vector<4xf32>
        %ep_v568_row0_nbr, %ep_v568_row0_valid = gpu.shuffle xor %ep_v568_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v568_row1_nbr, %ep_v568_row1_valid = gpu.shuffle xor %ep_v568_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v568_row2_nbr, %ep_v568_row2_valid = gpu.shuffle xor %ep_v568_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v568_row3_nbr, %ep_v568_row3_valid = gpu.shuffle xor %ep_v568_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v568_row0_lo = arith.select %ep_is_even, %ep_v568_row0,     %ep_v568_row0_nbr : f32
        %ep_v568_row0_hi = arith.select %ep_is_even, %ep_v568_row0_nbr, %ep_v568_row0     : f32
        %ep_v568_row1_lo = arith.select %ep_is_even, %ep_v568_row1,     %ep_v568_row1_nbr : f32
        %ep_v568_row1_hi = arith.select %ep_is_even, %ep_v568_row1_nbr, %ep_v568_row1     : f32
        %ep_v568_row2_lo = arith.select %ep_is_even, %ep_v568_row2,     %ep_v568_row2_nbr : f32
        %ep_v568_row2_hi = arith.select %ep_is_even, %ep_v568_row2_nbr, %ep_v568_row2     : f32
        %ep_v568_row3_lo = arith.select %ep_is_even, %ep_v568_row3,     %ep_v568_row3_nbr : f32
        %ep_v568_row3_hi = arith.select %ep_is_even, %ep_v568_row3_nbr, %ep_v568_row3     : f32
        %ep_v568_store_a_lo = arith.select %ep_is_even, %ep_v568_row0_lo, %ep_v568_row2_lo : f32
        %ep_v568_store_a_hi = arith.select %ep_is_even, %ep_v568_row0_hi, %ep_v568_row2_hi : f32
        %ep_v568_store_b_lo = arith.select %ep_is_even, %ep_v568_row1_lo, %ep_v568_row3_lo : f32
        %ep_v568_store_b_hi = arith.select %ep_is_even, %ep_v568_row1_hi, %ep_v568_row3_hi : f32
        %ep_v568_pair_a_0 = vector.broadcast %ep_v568_store_a_lo : f32 to vector<2xf32>
        %ep_v568_pair_a   = vector.insert %ep_v568_store_a_hi, %ep_v568_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v568_store_a  = arith.truncf %ep_v568_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v568_pair_b_0 = vector.broadcast %ep_v568_store_b_lo : f32 to vector<2xf32>
        %ep_v568_pair_b   = vector.insert %ep_v568_store_b_hi, %ep_v568_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v568_store_b  = arith.truncf %ep_v568_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v568_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col5_adj overflow<nsw> : index
        %ep_v568_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col5_adj overflow<nsw> : index
        vector.store %ep_v568_store_a, %ep_buffer[%ep_v568_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v568_store_b, %ep_buffer[%ep_v568_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 14: %570 → rb1, col6
        %ep_v570_row0 = vector.extract %570[0] : f32 from vector<4xf32>
        %ep_v570_row1 = vector.extract %570[1] : f32 from vector<4xf32>
        %ep_v570_row2 = vector.extract %570[2] : f32 from vector<4xf32>
        %ep_v570_row3 = vector.extract %570[3] : f32 from vector<4xf32>
        %ep_v570_row0_nbr, %ep_v570_row0_valid = gpu.shuffle xor %ep_v570_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v570_row1_nbr, %ep_v570_row1_valid = gpu.shuffle xor %ep_v570_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v570_row2_nbr, %ep_v570_row2_valid = gpu.shuffle xor %ep_v570_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v570_row3_nbr, %ep_v570_row3_valid = gpu.shuffle xor %ep_v570_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v570_row0_lo = arith.select %ep_is_even, %ep_v570_row0,     %ep_v570_row0_nbr : f32
        %ep_v570_row0_hi = arith.select %ep_is_even, %ep_v570_row0_nbr, %ep_v570_row0     : f32
        %ep_v570_row1_lo = arith.select %ep_is_even, %ep_v570_row1,     %ep_v570_row1_nbr : f32
        %ep_v570_row1_hi = arith.select %ep_is_even, %ep_v570_row1_nbr, %ep_v570_row1     : f32
        %ep_v570_row2_lo = arith.select %ep_is_even, %ep_v570_row2,     %ep_v570_row2_nbr : f32
        %ep_v570_row2_hi = arith.select %ep_is_even, %ep_v570_row2_nbr, %ep_v570_row2     : f32
        %ep_v570_row3_lo = arith.select %ep_is_even, %ep_v570_row3,     %ep_v570_row3_nbr : f32
        %ep_v570_row3_hi = arith.select %ep_is_even, %ep_v570_row3_nbr, %ep_v570_row3     : f32
        %ep_v570_store_a_lo = arith.select %ep_is_even, %ep_v570_row0_lo, %ep_v570_row2_lo : f32
        %ep_v570_store_a_hi = arith.select %ep_is_even, %ep_v570_row0_hi, %ep_v570_row2_hi : f32
        %ep_v570_store_b_lo = arith.select %ep_is_even, %ep_v570_row1_lo, %ep_v570_row3_lo : f32
        %ep_v570_store_b_hi = arith.select %ep_is_even, %ep_v570_row1_hi, %ep_v570_row3_hi : f32
        %ep_v570_pair_a_0 = vector.broadcast %ep_v570_store_a_lo : f32 to vector<2xf32>
        %ep_v570_pair_a   = vector.insert %ep_v570_store_a_hi, %ep_v570_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v570_store_a  = arith.truncf %ep_v570_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v570_pair_b_0 = vector.broadcast %ep_v570_store_b_lo : f32 to vector<2xf32>
        %ep_v570_pair_b   = vector.insert %ep_v570_store_b_hi, %ep_v570_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v570_store_b  = arith.truncf %ep_v570_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v570_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col6_adj overflow<nsw> : index
        %ep_v570_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col6_adj overflow<nsw> : index
        vector.store %ep_v570_store_a, %ep_buffer[%ep_v570_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v570_store_b, %ep_buffer[%ep_v570_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 15: %572 → rb1, col7
        %ep_v572_row0 = vector.extract %572[0] : f32 from vector<4xf32>
        %ep_v572_row1 = vector.extract %572[1] : f32 from vector<4xf32>
        %ep_v572_row2 = vector.extract %572[2] : f32 from vector<4xf32>
        %ep_v572_row3 = vector.extract %572[3] : f32 from vector<4xf32>
        %ep_v572_row0_nbr, %ep_v572_row0_valid = gpu.shuffle xor %ep_v572_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v572_row1_nbr, %ep_v572_row1_valid = gpu.shuffle xor %ep_v572_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v572_row2_nbr, %ep_v572_row2_valid = gpu.shuffle xor %ep_v572_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v572_row3_nbr, %ep_v572_row3_valid = gpu.shuffle xor %ep_v572_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v572_row0_lo = arith.select %ep_is_even, %ep_v572_row0,     %ep_v572_row0_nbr : f32
        %ep_v572_row0_hi = arith.select %ep_is_even, %ep_v572_row0_nbr, %ep_v572_row0     : f32
        %ep_v572_row1_lo = arith.select %ep_is_even, %ep_v572_row1,     %ep_v572_row1_nbr : f32
        %ep_v572_row1_hi = arith.select %ep_is_even, %ep_v572_row1_nbr, %ep_v572_row1     : f32
        %ep_v572_row2_lo = arith.select %ep_is_even, %ep_v572_row2,     %ep_v572_row2_nbr : f32
        %ep_v572_row2_hi = arith.select %ep_is_even, %ep_v572_row2_nbr, %ep_v572_row2     : f32
        %ep_v572_row3_lo = arith.select %ep_is_even, %ep_v572_row3,     %ep_v572_row3_nbr : f32
        %ep_v572_row3_hi = arith.select %ep_is_even, %ep_v572_row3_nbr, %ep_v572_row3     : f32
        %ep_v572_store_a_lo = arith.select %ep_is_even, %ep_v572_row0_lo, %ep_v572_row2_lo : f32
        %ep_v572_store_a_hi = arith.select %ep_is_even, %ep_v572_row0_hi, %ep_v572_row2_hi : f32
        %ep_v572_store_b_lo = arith.select %ep_is_even, %ep_v572_row1_lo, %ep_v572_row3_lo : f32
        %ep_v572_store_b_hi = arith.select %ep_is_even, %ep_v572_row1_hi, %ep_v572_row3_hi : f32
        %ep_v572_pair_a_0 = vector.broadcast %ep_v572_store_a_lo : f32 to vector<2xf32>
        %ep_v572_pair_a   = vector.insert %ep_v572_store_a_hi, %ep_v572_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v572_store_a  = arith.truncf %ep_v572_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v572_pair_b_0 = vector.broadcast %ep_v572_store_b_lo : f32 to vector<2xf32>
        %ep_v572_pair_b   = vector.insert %ep_v572_store_b_hi, %ep_v572_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v572_store_b  = arith.truncf %ep_v572_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v572_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col7_adj overflow<nsw> : index
        %ep_v572_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col7_adj overflow<nsw> : index
        vector.store %ep_v572_store_a, %ep_buffer[%ep_v572_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v572_store_b, %ep_buffer[%ep_v572_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 16: %574 → rb2, col0
        %ep_v574_row0 = vector.extract %574[0] : f32 from vector<4xf32>
        %ep_v574_row1 = vector.extract %574[1] : f32 from vector<4xf32>
        %ep_v574_row2 = vector.extract %574[2] : f32 from vector<4xf32>
        %ep_v574_row3 = vector.extract %574[3] : f32 from vector<4xf32>
        %ep_v574_row0_nbr, %ep_v574_row0_valid = gpu.shuffle xor %ep_v574_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v574_row1_nbr, %ep_v574_row1_valid = gpu.shuffle xor %ep_v574_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v574_row2_nbr, %ep_v574_row2_valid = gpu.shuffle xor %ep_v574_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v574_row3_nbr, %ep_v574_row3_valid = gpu.shuffle xor %ep_v574_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v574_row0_lo = arith.select %ep_is_even, %ep_v574_row0,     %ep_v574_row0_nbr : f32
        %ep_v574_row0_hi = arith.select %ep_is_even, %ep_v574_row0_nbr, %ep_v574_row0     : f32
        %ep_v574_row1_lo = arith.select %ep_is_even, %ep_v574_row1,     %ep_v574_row1_nbr : f32
        %ep_v574_row1_hi = arith.select %ep_is_even, %ep_v574_row1_nbr, %ep_v574_row1     : f32
        %ep_v574_row2_lo = arith.select %ep_is_even, %ep_v574_row2,     %ep_v574_row2_nbr : f32
        %ep_v574_row2_hi = arith.select %ep_is_even, %ep_v574_row2_nbr, %ep_v574_row2     : f32
        %ep_v574_row3_lo = arith.select %ep_is_even, %ep_v574_row3,     %ep_v574_row3_nbr : f32
        %ep_v574_row3_hi = arith.select %ep_is_even, %ep_v574_row3_nbr, %ep_v574_row3     : f32
        %ep_v574_store_a_lo = arith.select %ep_is_even, %ep_v574_row0_lo, %ep_v574_row2_lo : f32
        %ep_v574_store_a_hi = arith.select %ep_is_even, %ep_v574_row0_hi, %ep_v574_row2_hi : f32
        %ep_v574_store_b_lo = arith.select %ep_is_even, %ep_v574_row1_lo, %ep_v574_row3_lo : f32
        %ep_v574_store_b_hi = arith.select %ep_is_even, %ep_v574_row1_hi, %ep_v574_row3_hi : f32
        %ep_v574_pair_a_0 = vector.broadcast %ep_v574_store_a_lo : f32 to vector<2xf32>
        %ep_v574_pair_a   = vector.insert %ep_v574_store_a_hi, %ep_v574_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v574_store_a  = arith.truncf %ep_v574_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v574_pair_b_0 = vector.broadcast %ep_v574_store_b_lo : f32 to vector<2xf32>
        %ep_v574_pair_b   = vector.insert %ep_v574_store_b_hi, %ep_v574_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v574_store_b  = arith.truncf %ep_v574_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v574_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col0_adj overflow<nsw> : index
        %ep_v574_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col0_adj overflow<nsw> : index
        vector.store %ep_v574_store_a, %ep_buffer[%ep_v574_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v574_store_b, %ep_buffer[%ep_v574_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 17: %576 → rb2, col1
        %ep_v576_row0 = vector.extract %576[0] : f32 from vector<4xf32>
        %ep_v576_row1 = vector.extract %576[1] : f32 from vector<4xf32>
        %ep_v576_row2 = vector.extract %576[2] : f32 from vector<4xf32>
        %ep_v576_row3 = vector.extract %576[3] : f32 from vector<4xf32>
        %ep_v576_row0_nbr, %ep_v576_row0_valid = gpu.shuffle xor %ep_v576_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v576_row1_nbr, %ep_v576_row1_valid = gpu.shuffle xor %ep_v576_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v576_row2_nbr, %ep_v576_row2_valid = gpu.shuffle xor %ep_v576_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v576_row3_nbr, %ep_v576_row3_valid = gpu.shuffle xor %ep_v576_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v576_row0_lo = arith.select %ep_is_even, %ep_v576_row0,     %ep_v576_row0_nbr : f32
        %ep_v576_row0_hi = arith.select %ep_is_even, %ep_v576_row0_nbr, %ep_v576_row0     : f32
        %ep_v576_row1_lo = arith.select %ep_is_even, %ep_v576_row1,     %ep_v576_row1_nbr : f32
        %ep_v576_row1_hi = arith.select %ep_is_even, %ep_v576_row1_nbr, %ep_v576_row1     : f32
        %ep_v576_row2_lo = arith.select %ep_is_even, %ep_v576_row2,     %ep_v576_row2_nbr : f32
        %ep_v576_row2_hi = arith.select %ep_is_even, %ep_v576_row2_nbr, %ep_v576_row2     : f32
        %ep_v576_row3_lo = arith.select %ep_is_even, %ep_v576_row3,     %ep_v576_row3_nbr : f32
        %ep_v576_row3_hi = arith.select %ep_is_even, %ep_v576_row3_nbr, %ep_v576_row3     : f32
        %ep_v576_store_a_lo = arith.select %ep_is_even, %ep_v576_row0_lo, %ep_v576_row2_lo : f32
        %ep_v576_store_a_hi = arith.select %ep_is_even, %ep_v576_row0_hi, %ep_v576_row2_hi : f32
        %ep_v576_store_b_lo = arith.select %ep_is_even, %ep_v576_row1_lo, %ep_v576_row3_lo : f32
        %ep_v576_store_b_hi = arith.select %ep_is_even, %ep_v576_row1_hi, %ep_v576_row3_hi : f32
        %ep_v576_pair_a_0 = vector.broadcast %ep_v576_store_a_lo : f32 to vector<2xf32>
        %ep_v576_pair_a   = vector.insert %ep_v576_store_a_hi, %ep_v576_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v576_store_a  = arith.truncf %ep_v576_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v576_pair_b_0 = vector.broadcast %ep_v576_store_b_lo : f32 to vector<2xf32>
        %ep_v576_pair_b   = vector.insert %ep_v576_store_b_hi, %ep_v576_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v576_store_b  = arith.truncf %ep_v576_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v576_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col1_adj overflow<nsw> : index
        %ep_v576_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col1_adj overflow<nsw> : index
        vector.store %ep_v576_store_a, %ep_buffer[%ep_v576_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v576_store_b, %ep_buffer[%ep_v576_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 18: %578 → rb2, col2
        %ep_v578_row0 = vector.extract %578[0] : f32 from vector<4xf32>
        %ep_v578_row1 = vector.extract %578[1] : f32 from vector<4xf32>
        %ep_v578_row2 = vector.extract %578[2] : f32 from vector<4xf32>
        %ep_v578_row3 = vector.extract %578[3] : f32 from vector<4xf32>
        %ep_v578_row0_nbr, %ep_v578_row0_valid = gpu.shuffle xor %ep_v578_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v578_row1_nbr, %ep_v578_row1_valid = gpu.shuffle xor %ep_v578_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v578_row2_nbr, %ep_v578_row2_valid = gpu.shuffle xor %ep_v578_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v578_row3_nbr, %ep_v578_row3_valid = gpu.shuffle xor %ep_v578_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v578_row0_lo = arith.select %ep_is_even, %ep_v578_row0,     %ep_v578_row0_nbr : f32
        %ep_v578_row0_hi = arith.select %ep_is_even, %ep_v578_row0_nbr, %ep_v578_row0     : f32
        %ep_v578_row1_lo = arith.select %ep_is_even, %ep_v578_row1,     %ep_v578_row1_nbr : f32
        %ep_v578_row1_hi = arith.select %ep_is_even, %ep_v578_row1_nbr, %ep_v578_row1     : f32
        %ep_v578_row2_lo = arith.select %ep_is_even, %ep_v578_row2,     %ep_v578_row2_nbr : f32
        %ep_v578_row2_hi = arith.select %ep_is_even, %ep_v578_row2_nbr, %ep_v578_row2     : f32
        %ep_v578_row3_lo = arith.select %ep_is_even, %ep_v578_row3,     %ep_v578_row3_nbr : f32
        %ep_v578_row3_hi = arith.select %ep_is_even, %ep_v578_row3_nbr, %ep_v578_row3     : f32
        %ep_v578_store_a_lo = arith.select %ep_is_even, %ep_v578_row0_lo, %ep_v578_row2_lo : f32
        %ep_v578_store_a_hi = arith.select %ep_is_even, %ep_v578_row0_hi, %ep_v578_row2_hi : f32
        %ep_v578_store_b_lo = arith.select %ep_is_even, %ep_v578_row1_lo, %ep_v578_row3_lo : f32
        %ep_v578_store_b_hi = arith.select %ep_is_even, %ep_v578_row1_hi, %ep_v578_row3_hi : f32
        %ep_v578_pair_a_0 = vector.broadcast %ep_v578_store_a_lo : f32 to vector<2xf32>
        %ep_v578_pair_a   = vector.insert %ep_v578_store_a_hi, %ep_v578_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v578_store_a  = arith.truncf %ep_v578_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v578_pair_b_0 = vector.broadcast %ep_v578_store_b_lo : f32 to vector<2xf32>
        %ep_v578_pair_b   = vector.insert %ep_v578_store_b_hi, %ep_v578_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v578_store_b  = arith.truncf %ep_v578_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v578_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col2_adj overflow<nsw> : index
        %ep_v578_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col2_adj overflow<nsw> : index
        vector.store %ep_v578_store_a, %ep_buffer[%ep_v578_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v578_store_b, %ep_buffer[%ep_v578_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 19: %580 → rb2, col3
        %ep_v580_row0 = vector.extract %580[0] : f32 from vector<4xf32>
        %ep_v580_row1 = vector.extract %580[1] : f32 from vector<4xf32>
        %ep_v580_row2 = vector.extract %580[2] : f32 from vector<4xf32>
        %ep_v580_row3 = vector.extract %580[3] : f32 from vector<4xf32>
        %ep_v580_row0_nbr, %ep_v580_row0_valid = gpu.shuffle xor %ep_v580_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v580_row1_nbr, %ep_v580_row1_valid = gpu.shuffle xor %ep_v580_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v580_row2_nbr, %ep_v580_row2_valid = gpu.shuffle xor %ep_v580_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v580_row3_nbr, %ep_v580_row3_valid = gpu.shuffle xor %ep_v580_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v580_row0_lo = arith.select %ep_is_even, %ep_v580_row0,     %ep_v580_row0_nbr : f32
        %ep_v580_row0_hi = arith.select %ep_is_even, %ep_v580_row0_nbr, %ep_v580_row0     : f32
        %ep_v580_row1_lo = arith.select %ep_is_even, %ep_v580_row1,     %ep_v580_row1_nbr : f32
        %ep_v580_row1_hi = arith.select %ep_is_even, %ep_v580_row1_nbr, %ep_v580_row1     : f32
        %ep_v580_row2_lo = arith.select %ep_is_even, %ep_v580_row2,     %ep_v580_row2_nbr : f32
        %ep_v580_row2_hi = arith.select %ep_is_even, %ep_v580_row2_nbr, %ep_v580_row2     : f32
        %ep_v580_row3_lo = arith.select %ep_is_even, %ep_v580_row3,     %ep_v580_row3_nbr : f32
        %ep_v580_row3_hi = arith.select %ep_is_even, %ep_v580_row3_nbr, %ep_v580_row3     : f32
        %ep_v580_store_a_lo = arith.select %ep_is_even, %ep_v580_row0_lo, %ep_v580_row2_lo : f32
        %ep_v580_store_a_hi = arith.select %ep_is_even, %ep_v580_row0_hi, %ep_v580_row2_hi : f32
        %ep_v580_store_b_lo = arith.select %ep_is_even, %ep_v580_row1_lo, %ep_v580_row3_lo : f32
        %ep_v580_store_b_hi = arith.select %ep_is_even, %ep_v580_row1_hi, %ep_v580_row3_hi : f32
        %ep_v580_pair_a_0 = vector.broadcast %ep_v580_store_a_lo : f32 to vector<2xf32>
        %ep_v580_pair_a   = vector.insert %ep_v580_store_a_hi, %ep_v580_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v580_store_a  = arith.truncf %ep_v580_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v580_pair_b_0 = vector.broadcast %ep_v580_store_b_lo : f32 to vector<2xf32>
        %ep_v580_pair_b   = vector.insert %ep_v580_store_b_hi, %ep_v580_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v580_store_b  = arith.truncf %ep_v580_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v580_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col3_adj overflow<nsw> : index
        %ep_v580_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col3_adj overflow<nsw> : index
        vector.store %ep_v580_store_a, %ep_buffer[%ep_v580_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v580_store_b, %ep_buffer[%ep_v580_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 20: %582 → rb2, col4
        %ep_v582_row0 = vector.extract %582[0] : f32 from vector<4xf32>
        %ep_v582_row1 = vector.extract %582[1] : f32 from vector<4xf32>
        %ep_v582_row2 = vector.extract %582[2] : f32 from vector<4xf32>
        %ep_v582_row3 = vector.extract %582[3] : f32 from vector<4xf32>
        %ep_v582_row0_nbr, %ep_v582_row0_valid = gpu.shuffle xor %ep_v582_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v582_row1_nbr, %ep_v582_row1_valid = gpu.shuffle xor %ep_v582_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v582_row2_nbr, %ep_v582_row2_valid = gpu.shuffle xor %ep_v582_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v582_row3_nbr, %ep_v582_row3_valid = gpu.shuffle xor %ep_v582_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v582_row0_lo = arith.select %ep_is_even, %ep_v582_row0,     %ep_v582_row0_nbr : f32
        %ep_v582_row0_hi = arith.select %ep_is_even, %ep_v582_row0_nbr, %ep_v582_row0     : f32
        %ep_v582_row1_lo = arith.select %ep_is_even, %ep_v582_row1,     %ep_v582_row1_nbr : f32
        %ep_v582_row1_hi = arith.select %ep_is_even, %ep_v582_row1_nbr, %ep_v582_row1     : f32
        %ep_v582_row2_lo = arith.select %ep_is_even, %ep_v582_row2,     %ep_v582_row2_nbr : f32
        %ep_v582_row2_hi = arith.select %ep_is_even, %ep_v582_row2_nbr, %ep_v582_row2     : f32
        %ep_v582_row3_lo = arith.select %ep_is_even, %ep_v582_row3,     %ep_v582_row3_nbr : f32
        %ep_v582_row3_hi = arith.select %ep_is_even, %ep_v582_row3_nbr, %ep_v582_row3     : f32
        %ep_v582_store_a_lo = arith.select %ep_is_even, %ep_v582_row0_lo, %ep_v582_row2_lo : f32
        %ep_v582_store_a_hi = arith.select %ep_is_even, %ep_v582_row0_hi, %ep_v582_row2_hi : f32
        %ep_v582_store_b_lo = arith.select %ep_is_even, %ep_v582_row1_lo, %ep_v582_row3_lo : f32
        %ep_v582_store_b_hi = arith.select %ep_is_even, %ep_v582_row1_hi, %ep_v582_row3_hi : f32
        %ep_v582_pair_a_0 = vector.broadcast %ep_v582_store_a_lo : f32 to vector<2xf32>
        %ep_v582_pair_a   = vector.insert %ep_v582_store_a_hi, %ep_v582_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v582_store_a  = arith.truncf %ep_v582_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v582_pair_b_0 = vector.broadcast %ep_v582_store_b_lo : f32 to vector<2xf32>
        %ep_v582_pair_b   = vector.insert %ep_v582_store_b_hi, %ep_v582_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v582_store_b  = arith.truncf %ep_v582_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v582_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col4_adj overflow<nsw> : index
        %ep_v582_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col4_adj overflow<nsw> : index
        vector.store %ep_v582_store_a, %ep_buffer[%ep_v582_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v582_store_b, %ep_buffer[%ep_v582_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 21: %584 → rb2, col5
        %ep_v584_row0 = vector.extract %584[0] : f32 from vector<4xf32>
        %ep_v584_row1 = vector.extract %584[1] : f32 from vector<4xf32>
        %ep_v584_row2 = vector.extract %584[2] : f32 from vector<4xf32>
        %ep_v584_row3 = vector.extract %584[3] : f32 from vector<4xf32>
        %ep_v584_row0_nbr, %ep_v584_row0_valid = gpu.shuffle xor %ep_v584_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v584_row1_nbr, %ep_v584_row1_valid = gpu.shuffle xor %ep_v584_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v584_row2_nbr, %ep_v584_row2_valid = gpu.shuffle xor %ep_v584_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v584_row3_nbr, %ep_v584_row3_valid = gpu.shuffle xor %ep_v584_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v584_row0_lo = arith.select %ep_is_even, %ep_v584_row0,     %ep_v584_row0_nbr : f32
        %ep_v584_row0_hi = arith.select %ep_is_even, %ep_v584_row0_nbr, %ep_v584_row0     : f32
        %ep_v584_row1_lo = arith.select %ep_is_even, %ep_v584_row1,     %ep_v584_row1_nbr : f32
        %ep_v584_row1_hi = arith.select %ep_is_even, %ep_v584_row1_nbr, %ep_v584_row1     : f32
        %ep_v584_row2_lo = arith.select %ep_is_even, %ep_v584_row2,     %ep_v584_row2_nbr : f32
        %ep_v584_row2_hi = arith.select %ep_is_even, %ep_v584_row2_nbr, %ep_v584_row2     : f32
        %ep_v584_row3_lo = arith.select %ep_is_even, %ep_v584_row3,     %ep_v584_row3_nbr : f32
        %ep_v584_row3_hi = arith.select %ep_is_even, %ep_v584_row3_nbr, %ep_v584_row3     : f32
        %ep_v584_store_a_lo = arith.select %ep_is_even, %ep_v584_row0_lo, %ep_v584_row2_lo : f32
        %ep_v584_store_a_hi = arith.select %ep_is_even, %ep_v584_row0_hi, %ep_v584_row2_hi : f32
        %ep_v584_store_b_lo = arith.select %ep_is_even, %ep_v584_row1_lo, %ep_v584_row3_lo : f32
        %ep_v584_store_b_hi = arith.select %ep_is_even, %ep_v584_row1_hi, %ep_v584_row3_hi : f32
        %ep_v584_pair_a_0 = vector.broadcast %ep_v584_store_a_lo : f32 to vector<2xf32>
        %ep_v584_pair_a   = vector.insert %ep_v584_store_a_hi, %ep_v584_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v584_store_a  = arith.truncf %ep_v584_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v584_pair_b_0 = vector.broadcast %ep_v584_store_b_lo : f32 to vector<2xf32>
        %ep_v584_pair_b   = vector.insert %ep_v584_store_b_hi, %ep_v584_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v584_store_b  = arith.truncf %ep_v584_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v584_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col5_adj overflow<nsw> : index
        %ep_v584_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col5_adj overflow<nsw> : index
        vector.store %ep_v584_store_a, %ep_buffer[%ep_v584_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v584_store_b, %ep_buffer[%ep_v584_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 22: %586 → rb2, col6
        %ep_v586_row0 = vector.extract %586[0] : f32 from vector<4xf32>
        %ep_v586_row1 = vector.extract %586[1] : f32 from vector<4xf32>
        %ep_v586_row2 = vector.extract %586[2] : f32 from vector<4xf32>
        %ep_v586_row3 = vector.extract %586[3] : f32 from vector<4xf32>
        %ep_v586_row0_nbr, %ep_v586_row0_valid = gpu.shuffle xor %ep_v586_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v586_row1_nbr, %ep_v586_row1_valid = gpu.shuffle xor %ep_v586_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v586_row2_nbr, %ep_v586_row2_valid = gpu.shuffle xor %ep_v586_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v586_row3_nbr, %ep_v586_row3_valid = gpu.shuffle xor %ep_v586_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v586_row0_lo = arith.select %ep_is_even, %ep_v586_row0,     %ep_v586_row0_nbr : f32
        %ep_v586_row0_hi = arith.select %ep_is_even, %ep_v586_row0_nbr, %ep_v586_row0     : f32
        %ep_v586_row1_lo = arith.select %ep_is_even, %ep_v586_row1,     %ep_v586_row1_nbr : f32
        %ep_v586_row1_hi = arith.select %ep_is_even, %ep_v586_row1_nbr, %ep_v586_row1     : f32
        %ep_v586_row2_lo = arith.select %ep_is_even, %ep_v586_row2,     %ep_v586_row2_nbr : f32
        %ep_v586_row2_hi = arith.select %ep_is_even, %ep_v586_row2_nbr, %ep_v586_row2     : f32
        %ep_v586_row3_lo = arith.select %ep_is_even, %ep_v586_row3,     %ep_v586_row3_nbr : f32
        %ep_v586_row3_hi = arith.select %ep_is_even, %ep_v586_row3_nbr, %ep_v586_row3     : f32
        %ep_v586_store_a_lo = arith.select %ep_is_even, %ep_v586_row0_lo, %ep_v586_row2_lo : f32
        %ep_v586_store_a_hi = arith.select %ep_is_even, %ep_v586_row0_hi, %ep_v586_row2_hi : f32
        %ep_v586_store_b_lo = arith.select %ep_is_even, %ep_v586_row1_lo, %ep_v586_row3_lo : f32
        %ep_v586_store_b_hi = arith.select %ep_is_even, %ep_v586_row1_hi, %ep_v586_row3_hi : f32
        %ep_v586_pair_a_0 = vector.broadcast %ep_v586_store_a_lo : f32 to vector<2xf32>
        %ep_v586_pair_a   = vector.insert %ep_v586_store_a_hi, %ep_v586_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v586_store_a  = arith.truncf %ep_v586_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v586_pair_b_0 = vector.broadcast %ep_v586_store_b_lo : f32 to vector<2xf32>
        %ep_v586_pair_b   = vector.insert %ep_v586_store_b_hi, %ep_v586_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v586_store_b  = arith.truncf %ep_v586_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v586_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col6_adj overflow<nsw> : index
        %ep_v586_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col6_adj overflow<nsw> : index
        vector.store %ep_v586_store_a, %ep_buffer[%ep_v586_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v586_store_b, %ep_buffer[%ep_v586_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 23: %588 → rb2, col7
        %ep_v588_row0 = vector.extract %588[0] : f32 from vector<4xf32>
        %ep_v588_row1 = vector.extract %588[1] : f32 from vector<4xf32>
        %ep_v588_row2 = vector.extract %588[2] : f32 from vector<4xf32>
        %ep_v588_row3 = vector.extract %588[3] : f32 from vector<4xf32>
        %ep_v588_row0_nbr, %ep_v588_row0_valid = gpu.shuffle xor %ep_v588_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v588_row1_nbr, %ep_v588_row1_valid = gpu.shuffle xor %ep_v588_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v588_row2_nbr, %ep_v588_row2_valid = gpu.shuffle xor %ep_v588_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v588_row3_nbr, %ep_v588_row3_valid = gpu.shuffle xor %ep_v588_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v588_row0_lo = arith.select %ep_is_even, %ep_v588_row0,     %ep_v588_row0_nbr : f32
        %ep_v588_row0_hi = arith.select %ep_is_even, %ep_v588_row0_nbr, %ep_v588_row0     : f32
        %ep_v588_row1_lo = arith.select %ep_is_even, %ep_v588_row1,     %ep_v588_row1_nbr : f32
        %ep_v588_row1_hi = arith.select %ep_is_even, %ep_v588_row1_nbr, %ep_v588_row1     : f32
        %ep_v588_row2_lo = arith.select %ep_is_even, %ep_v588_row2,     %ep_v588_row2_nbr : f32
        %ep_v588_row2_hi = arith.select %ep_is_even, %ep_v588_row2_nbr, %ep_v588_row2     : f32
        %ep_v588_row3_lo = arith.select %ep_is_even, %ep_v588_row3,     %ep_v588_row3_nbr : f32
        %ep_v588_row3_hi = arith.select %ep_is_even, %ep_v588_row3_nbr, %ep_v588_row3     : f32
        %ep_v588_store_a_lo = arith.select %ep_is_even, %ep_v588_row0_lo, %ep_v588_row2_lo : f32
        %ep_v588_store_a_hi = arith.select %ep_is_even, %ep_v588_row0_hi, %ep_v588_row2_hi : f32
        %ep_v588_store_b_lo = arith.select %ep_is_even, %ep_v588_row1_lo, %ep_v588_row3_lo : f32
        %ep_v588_store_b_hi = arith.select %ep_is_even, %ep_v588_row1_hi, %ep_v588_row3_hi : f32
        %ep_v588_pair_a_0 = vector.broadcast %ep_v588_store_a_lo : f32 to vector<2xf32>
        %ep_v588_pair_a   = vector.insert %ep_v588_store_a_hi, %ep_v588_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v588_store_a  = arith.truncf %ep_v588_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v588_pair_b_0 = vector.broadcast %ep_v588_store_b_lo : f32 to vector<2xf32>
        %ep_v588_pair_b   = vector.insert %ep_v588_store_b_hi, %ep_v588_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v588_store_b  = arith.truncf %ep_v588_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v588_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col7_adj overflow<nsw> : index
        %ep_v588_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col7_adj overflow<nsw> : index
        vector.store %ep_v588_store_a, %ep_buffer[%ep_v588_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v588_store_b, %ep_buffer[%ep_v588_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 24: %590 → rb3, col0
        %ep_v590_row0 = vector.extract %590[0] : f32 from vector<4xf32>
        %ep_v590_row1 = vector.extract %590[1] : f32 from vector<4xf32>
        %ep_v590_row2 = vector.extract %590[2] : f32 from vector<4xf32>
        %ep_v590_row3 = vector.extract %590[3] : f32 from vector<4xf32>
        %ep_v590_row0_nbr, %ep_v590_row0_valid = gpu.shuffle xor %ep_v590_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v590_row1_nbr, %ep_v590_row1_valid = gpu.shuffle xor %ep_v590_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v590_row2_nbr, %ep_v590_row2_valid = gpu.shuffle xor %ep_v590_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v590_row3_nbr, %ep_v590_row3_valid = gpu.shuffle xor %ep_v590_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v590_row0_lo = arith.select %ep_is_even, %ep_v590_row0,     %ep_v590_row0_nbr : f32
        %ep_v590_row0_hi = arith.select %ep_is_even, %ep_v590_row0_nbr, %ep_v590_row0     : f32
        %ep_v590_row1_lo = arith.select %ep_is_even, %ep_v590_row1,     %ep_v590_row1_nbr : f32
        %ep_v590_row1_hi = arith.select %ep_is_even, %ep_v590_row1_nbr, %ep_v590_row1     : f32
        %ep_v590_row2_lo = arith.select %ep_is_even, %ep_v590_row2,     %ep_v590_row2_nbr : f32
        %ep_v590_row2_hi = arith.select %ep_is_even, %ep_v590_row2_nbr, %ep_v590_row2     : f32
        %ep_v590_row3_lo = arith.select %ep_is_even, %ep_v590_row3,     %ep_v590_row3_nbr : f32
        %ep_v590_row3_hi = arith.select %ep_is_even, %ep_v590_row3_nbr, %ep_v590_row3     : f32
        %ep_v590_store_a_lo = arith.select %ep_is_even, %ep_v590_row0_lo, %ep_v590_row2_lo : f32
        %ep_v590_store_a_hi = arith.select %ep_is_even, %ep_v590_row0_hi, %ep_v590_row2_hi : f32
        %ep_v590_store_b_lo = arith.select %ep_is_even, %ep_v590_row1_lo, %ep_v590_row3_lo : f32
        %ep_v590_store_b_hi = arith.select %ep_is_even, %ep_v590_row1_hi, %ep_v590_row3_hi : f32
        %ep_v590_pair_a_0 = vector.broadcast %ep_v590_store_a_lo : f32 to vector<2xf32>
        %ep_v590_pair_a   = vector.insert %ep_v590_store_a_hi, %ep_v590_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v590_store_a  = arith.truncf %ep_v590_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v590_pair_b_0 = vector.broadcast %ep_v590_store_b_lo : f32 to vector<2xf32>
        %ep_v590_pair_b   = vector.insert %ep_v590_store_b_hi, %ep_v590_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v590_store_b  = arith.truncf %ep_v590_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v590_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col0_adj overflow<nsw> : index
        %ep_v590_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col0_adj overflow<nsw> : index
        vector.store %ep_v590_store_a, %ep_buffer[%ep_v590_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v590_store_b, %ep_buffer[%ep_v590_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 25: %592 → rb3, col1
        %ep_v592_row0 = vector.extract %592[0] : f32 from vector<4xf32>
        %ep_v592_row1 = vector.extract %592[1] : f32 from vector<4xf32>
        %ep_v592_row2 = vector.extract %592[2] : f32 from vector<4xf32>
        %ep_v592_row3 = vector.extract %592[3] : f32 from vector<4xf32>
        %ep_v592_row0_nbr, %ep_v592_row0_valid = gpu.shuffle xor %ep_v592_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v592_row1_nbr, %ep_v592_row1_valid = gpu.shuffle xor %ep_v592_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v592_row2_nbr, %ep_v592_row2_valid = gpu.shuffle xor %ep_v592_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v592_row3_nbr, %ep_v592_row3_valid = gpu.shuffle xor %ep_v592_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v592_row0_lo = arith.select %ep_is_even, %ep_v592_row0,     %ep_v592_row0_nbr : f32
        %ep_v592_row0_hi = arith.select %ep_is_even, %ep_v592_row0_nbr, %ep_v592_row0     : f32
        %ep_v592_row1_lo = arith.select %ep_is_even, %ep_v592_row1,     %ep_v592_row1_nbr : f32
        %ep_v592_row1_hi = arith.select %ep_is_even, %ep_v592_row1_nbr, %ep_v592_row1     : f32
        %ep_v592_row2_lo = arith.select %ep_is_even, %ep_v592_row2,     %ep_v592_row2_nbr : f32
        %ep_v592_row2_hi = arith.select %ep_is_even, %ep_v592_row2_nbr, %ep_v592_row2     : f32
        %ep_v592_row3_lo = arith.select %ep_is_even, %ep_v592_row3,     %ep_v592_row3_nbr : f32
        %ep_v592_row3_hi = arith.select %ep_is_even, %ep_v592_row3_nbr, %ep_v592_row3     : f32
        %ep_v592_store_a_lo = arith.select %ep_is_even, %ep_v592_row0_lo, %ep_v592_row2_lo : f32
        %ep_v592_store_a_hi = arith.select %ep_is_even, %ep_v592_row0_hi, %ep_v592_row2_hi : f32
        %ep_v592_store_b_lo = arith.select %ep_is_even, %ep_v592_row1_lo, %ep_v592_row3_lo : f32
        %ep_v592_store_b_hi = arith.select %ep_is_even, %ep_v592_row1_hi, %ep_v592_row3_hi : f32
        %ep_v592_pair_a_0 = vector.broadcast %ep_v592_store_a_lo : f32 to vector<2xf32>
        %ep_v592_pair_a   = vector.insert %ep_v592_store_a_hi, %ep_v592_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v592_store_a  = arith.truncf %ep_v592_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v592_pair_b_0 = vector.broadcast %ep_v592_store_b_lo : f32 to vector<2xf32>
        %ep_v592_pair_b   = vector.insert %ep_v592_store_b_hi, %ep_v592_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v592_store_b  = arith.truncf %ep_v592_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v592_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col1_adj overflow<nsw> : index
        %ep_v592_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col1_adj overflow<nsw> : index
        vector.store %ep_v592_store_a, %ep_buffer[%ep_v592_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v592_store_b, %ep_buffer[%ep_v592_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 26: %594 → rb3, col2
        %ep_v594_row0 = vector.extract %594[0] : f32 from vector<4xf32>
        %ep_v594_row1 = vector.extract %594[1] : f32 from vector<4xf32>
        %ep_v594_row2 = vector.extract %594[2] : f32 from vector<4xf32>
        %ep_v594_row3 = vector.extract %594[3] : f32 from vector<4xf32>
        %ep_v594_row0_nbr, %ep_v594_row0_valid = gpu.shuffle xor %ep_v594_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v594_row1_nbr, %ep_v594_row1_valid = gpu.shuffle xor %ep_v594_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v594_row2_nbr, %ep_v594_row2_valid = gpu.shuffle xor %ep_v594_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v594_row3_nbr, %ep_v594_row3_valid = gpu.shuffle xor %ep_v594_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v594_row0_lo = arith.select %ep_is_even, %ep_v594_row0,     %ep_v594_row0_nbr : f32
        %ep_v594_row0_hi = arith.select %ep_is_even, %ep_v594_row0_nbr, %ep_v594_row0     : f32
        %ep_v594_row1_lo = arith.select %ep_is_even, %ep_v594_row1,     %ep_v594_row1_nbr : f32
        %ep_v594_row1_hi = arith.select %ep_is_even, %ep_v594_row1_nbr, %ep_v594_row1     : f32
        %ep_v594_row2_lo = arith.select %ep_is_even, %ep_v594_row2,     %ep_v594_row2_nbr : f32
        %ep_v594_row2_hi = arith.select %ep_is_even, %ep_v594_row2_nbr, %ep_v594_row2     : f32
        %ep_v594_row3_lo = arith.select %ep_is_even, %ep_v594_row3,     %ep_v594_row3_nbr : f32
        %ep_v594_row3_hi = arith.select %ep_is_even, %ep_v594_row3_nbr, %ep_v594_row3     : f32
        %ep_v594_store_a_lo = arith.select %ep_is_even, %ep_v594_row0_lo, %ep_v594_row2_lo : f32
        %ep_v594_store_a_hi = arith.select %ep_is_even, %ep_v594_row0_hi, %ep_v594_row2_hi : f32
        %ep_v594_store_b_lo = arith.select %ep_is_even, %ep_v594_row1_lo, %ep_v594_row3_lo : f32
        %ep_v594_store_b_hi = arith.select %ep_is_even, %ep_v594_row1_hi, %ep_v594_row3_hi : f32
        %ep_v594_pair_a_0 = vector.broadcast %ep_v594_store_a_lo : f32 to vector<2xf32>
        %ep_v594_pair_a   = vector.insert %ep_v594_store_a_hi, %ep_v594_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v594_store_a  = arith.truncf %ep_v594_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v594_pair_b_0 = vector.broadcast %ep_v594_store_b_lo : f32 to vector<2xf32>
        %ep_v594_pair_b   = vector.insert %ep_v594_store_b_hi, %ep_v594_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v594_store_b  = arith.truncf %ep_v594_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v594_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col2_adj overflow<nsw> : index
        %ep_v594_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col2_adj overflow<nsw> : index
        vector.store %ep_v594_store_a, %ep_buffer[%ep_v594_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v594_store_b, %ep_buffer[%ep_v594_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 27: %596 → rb3, col3
        %ep_v596_row0 = vector.extract %596[0] : f32 from vector<4xf32>
        %ep_v596_row1 = vector.extract %596[1] : f32 from vector<4xf32>
        %ep_v596_row2 = vector.extract %596[2] : f32 from vector<4xf32>
        %ep_v596_row3 = vector.extract %596[3] : f32 from vector<4xf32>
        %ep_v596_row0_nbr, %ep_v596_row0_valid = gpu.shuffle xor %ep_v596_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v596_row1_nbr, %ep_v596_row1_valid = gpu.shuffle xor %ep_v596_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v596_row2_nbr, %ep_v596_row2_valid = gpu.shuffle xor %ep_v596_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v596_row3_nbr, %ep_v596_row3_valid = gpu.shuffle xor %ep_v596_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v596_row0_lo = arith.select %ep_is_even, %ep_v596_row0,     %ep_v596_row0_nbr : f32
        %ep_v596_row0_hi = arith.select %ep_is_even, %ep_v596_row0_nbr, %ep_v596_row0     : f32
        %ep_v596_row1_lo = arith.select %ep_is_even, %ep_v596_row1,     %ep_v596_row1_nbr : f32
        %ep_v596_row1_hi = arith.select %ep_is_even, %ep_v596_row1_nbr, %ep_v596_row1     : f32
        %ep_v596_row2_lo = arith.select %ep_is_even, %ep_v596_row2,     %ep_v596_row2_nbr : f32
        %ep_v596_row2_hi = arith.select %ep_is_even, %ep_v596_row2_nbr, %ep_v596_row2     : f32
        %ep_v596_row3_lo = arith.select %ep_is_even, %ep_v596_row3,     %ep_v596_row3_nbr : f32
        %ep_v596_row3_hi = arith.select %ep_is_even, %ep_v596_row3_nbr, %ep_v596_row3     : f32
        %ep_v596_store_a_lo = arith.select %ep_is_even, %ep_v596_row0_lo, %ep_v596_row2_lo : f32
        %ep_v596_store_a_hi = arith.select %ep_is_even, %ep_v596_row0_hi, %ep_v596_row2_hi : f32
        %ep_v596_store_b_lo = arith.select %ep_is_even, %ep_v596_row1_lo, %ep_v596_row3_lo : f32
        %ep_v596_store_b_hi = arith.select %ep_is_even, %ep_v596_row1_hi, %ep_v596_row3_hi : f32
        %ep_v596_pair_a_0 = vector.broadcast %ep_v596_store_a_lo : f32 to vector<2xf32>
        %ep_v596_pair_a   = vector.insert %ep_v596_store_a_hi, %ep_v596_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v596_store_a  = arith.truncf %ep_v596_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v596_pair_b_0 = vector.broadcast %ep_v596_store_b_lo : f32 to vector<2xf32>
        %ep_v596_pair_b   = vector.insert %ep_v596_store_b_hi, %ep_v596_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v596_store_b  = arith.truncf %ep_v596_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v596_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col3_adj overflow<nsw> : index
        %ep_v596_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col3_adj overflow<nsw> : index
        vector.store %ep_v596_store_a, %ep_buffer[%ep_v596_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v596_store_b, %ep_buffer[%ep_v596_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 28: %598 → rb3, col4
        %ep_v598_row0 = vector.extract %598[0] : f32 from vector<4xf32>
        %ep_v598_row1 = vector.extract %598[1] : f32 from vector<4xf32>
        %ep_v598_row2 = vector.extract %598[2] : f32 from vector<4xf32>
        %ep_v598_row3 = vector.extract %598[3] : f32 from vector<4xf32>
        %ep_v598_row0_nbr, %ep_v598_row0_valid = gpu.shuffle xor %ep_v598_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v598_row1_nbr, %ep_v598_row1_valid = gpu.shuffle xor %ep_v598_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v598_row2_nbr, %ep_v598_row2_valid = gpu.shuffle xor %ep_v598_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v598_row3_nbr, %ep_v598_row3_valid = gpu.shuffle xor %ep_v598_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v598_row0_lo = arith.select %ep_is_even, %ep_v598_row0,     %ep_v598_row0_nbr : f32
        %ep_v598_row0_hi = arith.select %ep_is_even, %ep_v598_row0_nbr, %ep_v598_row0     : f32
        %ep_v598_row1_lo = arith.select %ep_is_even, %ep_v598_row1,     %ep_v598_row1_nbr : f32
        %ep_v598_row1_hi = arith.select %ep_is_even, %ep_v598_row1_nbr, %ep_v598_row1     : f32
        %ep_v598_row2_lo = arith.select %ep_is_even, %ep_v598_row2,     %ep_v598_row2_nbr : f32
        %ep_v598_row2_hi = arith.select %ep_is_even, %ep_v598_row2_nbr, %ep_v598_row2     : f32
        %ep_v598_row3_lo = arith.select %ep_is_even, %ep_v598_row3,     %ep_v598_row3_nbr : f32
        %ep_v598_row3_hi = arith.select %ep_is_even, %ep_v598_row3_nbr, %ep_v598_row3     : f32
        %ep_v598_store_a_lo = arith.select %ep_is_even, %ep_v598_row0_lo, %ep_v598_row2_lo : f32
        %ep_v598_store_a_hi = arith.select %ep_is_even, %ep_v598_row0_hi, %ep_v598_row2_hi : f32
        %ep_v598_store_b_lo = arith.select %ep_is_even, %ep_v598_row1_lo, %ep_v598_row3_lo : f32
        %ep_v598_store_b_hi = arith.select %ep_is_even, %ep_v598_row1_hi, %ep_v598_row3_hi : f32
        %ep_v598_pair_a_0 = vector.broadcast %ep_v598_store_a_lo : f32 to vector<2xf32>
        %ep_v598_pair_a   = vector.insert %ep_v598_store_a_hi, %ep_v598_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v598_store_a  = arith.truncf %ep_v598_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v598_pair_b_0 = vector.broadcast %ep_v598_store_b_lo : f32 to vector<2xf32>
        %ep_v598_pair_b   = vector.insert %ep_v598_store_b_hi, %ep_v598_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v598_store_b  = arith.truncf %ep_v598_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v598_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col4_adj overflow<nsw> : index
        %ep_v598_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col4_adj overflow<nsw> : index
        vector.store %ep_v598_store_a, %ep_buffer[%ep_v598_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v598_store_b, %ep_buffer[%ep_v598_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 29: %600 → rb3, col5
        %ep_v600_row0 = vector.extract %600[0] : f32 from vector<4xf32>
        %ep_v600_row1 = vector.extract %600[1] : f32 from vector<4xf32>
        %ep_v600_row2 = vector.extract %600[2] : f32 from vector<4xf32>
        %ep_v600_row3 = vector.extract %600[3] : f32 from vector<4xf32>
        %ep_v600_row0_nbr, %ep_v600_row0_valid = gpu.shuffle xor %ep_v600_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v600_row1_nbr, %ep_v600_row1_valid = gpu.shuffle xor %ep_v600_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v600_row2_nbr, %ep_v600_row2_valid = gpu.shuffle xor %ep_v600_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v600_row3_nbr, %ep_v600_row3_valid = gpu.shuffle xor %ep_v600_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v600_row0_lo = arith.select %ep_is_even, %ep_v600_row0,     %ep_v600_row0_nbr : f32
        %ep_v600_row0_hi = arith.select %ep_is_even, %ep_v600_row0_nbr, %ep_v600_row0     : f32
        %ep_v600_row1_lo = arith.select %ep_is_even, %ep_v600_row1,     %ep_v600_row1_nbr : f32
        %ep_v600_row1_hi = arith.select %ep_is_even, %ep_v600_row1_nbr, %ep_v600_row1     : f32
        %ep_v600_row2_lo = arith.select %ep_is_even, %ep_v600_row2,     %ep_v600_row2_nbr : f32
        %ep_v600_row2_hi = arith.select %ep_is_even, %ep_v600_row2_nbr, %ep_v600_row2     : f32
        %ep_v600_row3_lo = arith.select %ep_is_even, %ep_v600_row3,     %ep_v600_row3_nbr : f32
        %ep_v600_row3_hi = arith.select %ep_is_even, %ep_v600_row3_nbr, %ep_v600_row3     : f32
        %ep_v600_store_a_lo = arith.select %ep_is_even, %ep_v600_row0_lo, %ep_v600_row2_lo : f32
        %ep_v600_store_a_hi = arith.select %ep_is_even, %ep_v600_row0_hi, %ep_v600_row2_hi : f32
        %ep_v600_store_b_lo = arith.select %ep_is_even, %ep_v600_row1_lo, %ep_v600_row3_lo : f32
        %ep_v600_store_b_hi = arith.select %ep_is_even, %ep_v600_row1_hi, %ep_v600_row3_hi : f32
        %ep_v600_pair_a_0 = vector.broadcast %ep_v600_store_a_lo : f32 to vector<2xf32>
        %ep_v600_pair_a   = vector.insert %ep_v600_store_a_hi, %ep_v600_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v600_store_a  = arith.truncf %ep_v600_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v600_pair_b_0 = vector.broadcast %ep_v600_store_b_lo : f32 to vector<2xf32>
        %ep_v600_pair_b   = vector.insert %ep_v600_store_b_hi, %ep_v600_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v600_store_b  = arith.truncf %ep_v600_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v600_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col5_adj overflow<nsw> : index
        %ep_v600_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col5_adj overflow<nsw> : index
        vector.store %ep_v600_store_a, %ep_buffer[%ep_v600_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v600_store_b, %ep_buffer[%ep_v600_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 30: %602 → rb3, col6
        %ep_v602_row0 = vector.extract %602[0] : f32 from vector<4xf32>
        %ep_v602_row1 = vector.extract %602[1] : f32 from vector<4xf32>
        %ep_v602_row2 = vector.extract %602[2] : f32 from vector<4xf32>
        %ep_v602_row3 = vector.extract %602[3] : f32 from vector<4xf32>
        %ep_v602_row0_nbr, %ep_v602_row0_valid = gpu.shuffle xor %ep_v602_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v602_row1_nbr, %ep_v602_row1_valid = gpu.shuffle xor %ep_v602_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v602_row2_nbr, %ep_v602_row2_valid = gpu.shuffle xor %ep_v602_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v602_row3_nbr, %ep_v602_row3_valid = gpu.shuffle xor %ep_v602_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v602_row0_lo = arith.select %ep_is_even, %ep_v602_row0,     %ep_v602_row0_nbr : f32
        %ep_v602_row0_hi = arith.select %ep_is_even, %ep_v602_row0_nbr, %ep_v602_row0     : f32
        %ep_v602_row1_lo = arith.select %ep_is_even, %ep_v602_row1,     %ep_v602_row1_nbr : f32
        %ep_v602_row1_hi = arith.select %ep_is_even, %ep_v602_row1_nbr, %ep_v602_row1     : f32
        %ep_v602_row2_lo = arith.select %ep_is_even, %ep_v602_row2,     %ep_v602_row2_nbr : f32
        %ep_v602_row2_hi = arith.select %ep_is_even, %ep_v602_row2_nbr, %ep_v602_row2     : f32
        %ep_v602_row3_lo = arith.select %ep_is_even, %ep_v602_row3,     %ep_v602_row3_nbr : f32
        %ep_v602_row3_hi = arith.select %ep_is_even, %ep_v602_row3_nbr, %ep_v602_row3     : f32
        %ep_v602_store_a_lo = arith.select %ep_is_even, %ep_v602_row0_lo, %ep_v602_row2_lo : f32
        %ep_v602_store_a_hi = arith.select %ep_is_even, %ep_v602_row0_hi, %ep_v602_row2_hi : f32
        %ep_v602_store_b_lo = arith.select %ep_is_even, %ep_v602_row1_lo, %ep_v602_row3_lo : f32
        %ep_v602_store_b_hi = arith.select %ep_is_even, %ep_v602_row1_hi, %ep_v602_row3_hi : f32
        %ep_v602_pair_a_0 = vector.broadcast %ep_v602_store_a_lo : f32 to vector<2xf32>
        %ep_v602_pair_a   = vector.insert %ep_v602_store_a_hi, %ep_v602_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v602_store_a  = arith.truncf %ep_v602_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v602_pair_b_0 = vector.broadcast %ep_v602_store_b_lo : f32 to vector<2xf32>
        %ep_v602_pair_b   = vector.insert %ep_v602_store_b_hi, %ep_v602_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v602_store_b  = arith.truncf %ep_v602_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v602_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col6_adj overflow<nsw> : index
        %ep_v602_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col6_adj overflow<nsw> : index
        vector.store %ep_v602_store_a, %ep_buffer[%ep_v602_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v602_store_b, %ep_buffer[%ep_v602_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA 31: %604 → rb3, col7
        %ep_v604_row0 = vector.extract %604[0] : f32 from vector<4xf32>
        %ep_v604_row1 = vector.extract %604[1] : f32 from vector<4xf32>
        %ep_v604_row2 = vector.extract %604[2] : f32 from vector<4xf32>
        %ep_v604_row3 = vector.extract %604[3] : f32 from vector<4xf32>
        %ep_v604_row0_nbr, %ep_v604_row0_valid = gpu.shuffle xor %ep_v604_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v604_row1_nbr, %ep_v604_row1_valid = gpu.shuffle xor %ep_v604_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v604_row2_nbr, %ep_v604_row2_valid = gpu.shuffle xor %ep_v604_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v604_row3_nbr, %ep_v604_row3_valid = gpu.shuffle xor %ep_v604_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
        %ep_v604_row0_lo = arith.select %ep_is_even, %ep_v604_row0,     %ep_v604_row0_nbr : f32
        %ep_v604_row0_hi = arith.select %ep_is_even, %ep_v604_row0_nbr, %ep_v604_row0     : f32
        %ep_v604_row1_lo = arith.select %ep_is_even, %ep_v604_row1,     %ep_v604_row1_nbr : f32
        %ep_v604_row1_hi = arith.select %ep_is_even, %ep_v604_row1_nbr, %ep_v604_row1     : f32
        %ep_v604_row2_lo = arith.select %ep_is_even, %ep_v604_row2,     %ep_v604_row2_nbr : f32
        %ep_v604_row2_hi = arith.select %ep_is_even, %ep_v604_row2_nbr, %ep_v604_row2     : f32
        %ep_v604_row3_lo = arith.select %ep_is_even, %ep_v604_row3,     %ep_v604_row3_nbr : f32
        %ep_v604_row3_hi = arith.select %ep_is_even, %ep_v604_row3_nbr, %ep_v604_row3     : f32
        %ep_v604_store_a_lo = arith.select %ep_is_even, %ep_v604_row0_lo, %ep_v604_row2_lo : f32
        %ep_v604_store_a_hi = arith.select %ep_is_even, %ep_v604_row0_hi, %ep_v604_row2_hi : f32
        %ep_v604_store_b_lo = arith.select %ep_is_even, %ep_v604_row1_lo, %ep_v604_row3_lo : f32
        %ep_v604_store_b_hi = arith.select %ep_is_even, %ep_v604_row1_hi, %ep_v604_row3_hi : f32
        %ep_v604_pair_a_0 = vector.broadcast %ep_v604_store_a_lo : f32 to vector<2xf32>
        %ep_v604_pair_a   = vector.insert %ep_v604_store_a_hi, %ep_v604_pair_a_0 [1] : f32 into vector<2xf32>
        %ep_v604_store_a  = arith.truncf %ep_v604_pair_a : vector<2xf32> to vector<2xbf16>
        %ep_v604_pair_b_0 = vector.broadcast %ep_v604_store_b_lo : f32 to vector<2xf32>
        %ep_v604_pair_b   = vector.insert %ep_v604_store_b_hi, %ep_v604_pair_b_0 [1] : f32 into vector<2xf32>
        %ep_v604_store_b  = arith.truncf %ep_v604_pair_b : vector<2xf32> to vector<2xbf16>
        %ep_v604_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col7_adj overflow<nsw> : index
        %ep_v604_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col7_adj overflow<nsw> : index
        vector.store %ep_v604_store_a, %ep_buffer[%ep_v604_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %ep_v604_store_b, %ep_buffer[%ep_v604_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: !hal.fence, %arg11: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg10) => %arg0 : !hal.buffer_view -> tensor<8192x512xi8>
    %1 = hal.tensor.import wait(%arg10) => %arg1 : !hal.buffer_view -> tensor<8192x32xi8>
    %2 = hal.tensor.import wait(%arg10) => %arg2 : !hal.buffer_view -> tensor<8192x512xi8>
    %3 = hal.tensor.import wait(%arg10) => %arg3 : !hal.buffer_view -> tensor<8192x32xi8>
    %4 = hal.tensor.import wait(%arg10) => %arg4 : !hal.buffer_view -> tensor<8192x8192xbf16>
    %5 = flow.dispatch @gemm::@gemm[%arg5, %arg6, %arg7, %arg8, %arg9](%0, %1, %2, %3, %4, %arg5, %arg6, %arg7, %arg8, %arg9) : (tensor<8192x512xi8>, tensor<8192x32xi8>, tensor<8192x512xi8>, tensor<8192x32xi8>, tensor<8192x8192xbf16>, index, index, index, index, index) -> %4
    %6 = hal.tensor.barrier join(%5 : tensor<8192x8192xbf16>) => %arg11 : !hal.fence
    %7 = hal.tensor.export %6 : tensor<8192x8192xbf16> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
