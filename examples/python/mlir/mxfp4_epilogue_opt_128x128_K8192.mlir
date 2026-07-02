#map = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 16) * 128)>
#map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
#map2 = affine_map<()[s0] -> (s0 mod 8)>
#map3 = affine_map<()[s0, s1, s2, s3] -> (s0 * 524288 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8) floordiv 128) * 524288)>
#map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 16) * 128 + 64)>
#map5 = affine_map<()[s0, s1, s2, s3] -> (s0 * 524288 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 128) * 524288 + 262144)>
#map6 = affine_map<()[s0, s1] -> (s0 * 32768 + s1 * 4 + (s1 floordiv 64) * 8192 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64)>
#map7 = affine_map<()[s0, s1, s2] -> (s0 * 32768 + s1 * 16384 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64)>
#map8 = affine_map<()[s0, s1, s2] -> (s0 * 32768 + s1 * 16384 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 8192)>
#map9 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
#map10 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
#map11 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 4096 - (s0 floordiv 16) * 2048)>
#map12 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 4096 - (s0 floordiv 16) * 2048 + 2048)>
#map13 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048)>
#map14 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 2048)>
#map15 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 4096)>
#map16 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 6144)>
#map17 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
#map18 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 524288 + s1 * 131072 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8) floordiv 128) * 524288 + 128)>
#map19 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 524288 + s1 * 131072 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 128) * 524288 + 262272)>
#map20 = affine_map<()[s0, s1, s2] -> (s0 * 32768 + s1 * 4 + s2 * 256 + (s1 floordiv 64) * 8192 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 256)>
#map21 = affine_map<()[s0, s1, s2, s3] -> (s0 * 32768 + s1 * 16384 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 - (s3 floordiv 16) * 64 + 256)>
#map22 = affine_map<()[s0, s1, s2, s3] -> (s0 * 32768 + s1 * 16384 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 - (s3 floordiv 16) * 64 + 8448)>
#map23 = affine_map<()[s0, s1, s2, s3] -> (s0 * 524288 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8) floordiv 128) * 524288 + 3968)>
#map24 = affine_map<()[s0, s1, s2, s3] -> (s0 * 524288 + s1 * 131072 + s3 * 16 + (s2 floordiv 8) * 4096 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 128) * 524288 + 266112)>
#map25 = affine_map<()[s0, s1] -> (s0 * 32768 + s1 * 4 + (s1 floordiv 64) * 8192 + ((s1 mod 64) floordiv 16) * 64 - (s1 floordiv 16) * 64 + 7936)>
#map26 = affine_map<()[s0, s1, s2] -> (s0 * 32768 + s1 * 16384 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 7936)>
#map27 = affine_map<()[s0, s1, s2] -> (s0 * 32768 + s1 * 16384 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 - (s2 floordiv 16) * 64 + 16128)>
#map28 = affine_map<()[s0] -> (s0 * 128)>
#map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4)>
#map30 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16)>
#map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map34 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 16)>
#map35 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 32)>
#map36 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 48)>
#map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (index, index, index) {
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      stream.return %c16, %c16, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index) attributes {translation_info = #translation} {
        %c4_i32 = arith.constant 4 : i32
        %c256_i14 = arith.constant 256 : i14
        %c4096_i14 = arith.constant 4096 : i14
        %c2147483645_i64 = arith.constant 2147483645 : i64
        %c30 = arith.constant 30 : index
        %c524288_i64 = arith.constant 524288 : i64
        %c2 = arith.constant 2 : index
        %c8388608_i64 = arith.constant 8388608 : i64
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
        %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<bf16>
        %block_id_x = gpu.block_id x upper_bound 16
        %block_id_y = gpu.block_id y upper_bound 16
        %thread_id_x = gpu.thread_id x upper_bound 256
        %thread_id_y = gpu.thread_id y upper_bound 2
        %reinterpret_cast = memref.reinterpret_cast %4 to offset: [0], sizes: [2048, 2048], strides: [%arg9, 1] : memref<bf16> to memref<2048x2048xbf16, strided<[?, 1]>>
        %alloc = memref.alloc() : memref<128x128xi8, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<128x128xi8, #gpu.address_space<workgroup>>
        %alloc_1 = memref.alloc() : memref<128x128xi8, #gpu.address_space<workgroup>>
        %alloc_2 = memref.alloc() : memref<128x128xi8, #gpu.address_space<workgroup>>
        %5 = affine.apply #map()[%thread_id_x, %thread_id_y]
        %6 = gpu.subgroup_broadcast %5, first_active_lane : index
        %7 = gpu.subgroup_broadcast %c0, first_active_lane : index
        %reinterpret_cast_3 = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast = memref.cast %reinterpret_cast_3 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %8 = affine.apply #map1()[%thread_id_x]
        %9 = affine.apply #map2()[%thread_id_x]
        %10 = arith.xori %9, %8 : index
        %11 = affine.apply #map3()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        %12 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c8388608_i64) cacheSwizzleStride(%c4096_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %12[%11], %alloc_2[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %13 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %14 = gpu.subgroup_broadcast %13, first_active_lane : index
        %15 = affine.apply #map5()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%15], %alloc_2[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_4 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_5 = memref.cast %reinterpret_cast_4 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %16 = affine.apply #map3()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        %17 = amdgpu.fat_raw_buffer_cast %cast_5 validBytes(%c8388608_i64) cacheSwizzleStride(%c4096_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %17[%16], %alloc_0[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %18 = affine.apply #map5()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %17[%18], %alloc_0[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_6 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_7 = memref.cast %reinterpret_cast_6 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %19 = amdgpu.fat_raw_buffer_cast %cast_7 validBytes(%c524288_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %20 = affine.apply #map6()[%block_id_x, %thread_id_x]
        %21 = vector.load %19[%20] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %reinterpret_cast_8 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_9 = memref.cast %reinterpret_cast_8 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %22 = amdgpu.fat_raw_buffer_cast %cast_9 validBytes(%c524288_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %23 = affine.apply #map7()[%block_id_y, %thread_id_y, %thread_id_x]
        %24 = vector.load %22[%23] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %25 = affine.apply #map8()[%block_id_y, %thread_id_y, %thread_id_x]
        %26 = vector.load %22[%25] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        amdgpu.memory_counter_wait load(3)
        rocdl.s.barrier
        %27 = affine.apply #map9()[%thread_id_x, %thread_id_y]
        %28 = arith.index_cast %27 : index to i32
        %29 = arith.cmpi sge, %28, %c4_i32 : i32
        %30 = arith.cmpi slt, %28, %c4_i32 : i32
        scf.if %29 {
          rocdl.s.barrier
        }
        %reinterpret_cast_10 = memref.reinterpret_cast %alloc_2 to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
        %31 = affine.apply #map10()[%thread_id_x]
        %32 = arith.xori %31, %9 : index
        %33 = affine.apply #map11()[%thread_id_x, %32]
        %34 = affine.apply #map12()[%thread_id_x, %32]
        %reinterpret_cast_11 = memref.reinterpret_cast %alloc_0 to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
        %35 = affine.apply #map13()[%thread_id_y, %thread_id_x, %32]
        %36 = affine.apply #map14()[%thread_id_y, %thread_id_x, %32]
        %37 = affine.apply #map15()[%thread_id_y, %thread_id_x, %32]
        %38 = affine.apply #map16()[%thread_id_y, %thread_id_x, %32]
        %39 = affine.apply #map17()[%thread_id_x]
        %40 = arith.xori %39, %9 : index
        %41 = affine.apply #map11()[%thread_id_x, %40]
        %42 = affine.apply #map12()[%thread_id_x, %40]
        %43 = affine.apply #map13()[%thread_id_y, %thread_id_x, %40]
        %44 = affine.apply #map14()[%thread_id_y, %thread_id_x, %40]
        %45 = affine.apply #map15()[%thread_id_y, %thread_id_x, %40]
        %46 = affine.apply #map16()[%thread_id_y, %thread_id_x, %40]
        %reinterpret_cast_12 = memref.reinterpret_cast %alloc_1 to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_13 = memref.reinterpret_cast %alloc to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
        %47:11 = scf.for %arg10 = %c0 to %c30 step %c2 iter_args(%arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %21, %arg20 = %24, %arg21 = %26) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>) {
          %242 = vector.bitcast %arg21 : vector<4xi8> to vector<4xf8E8M0FNU>
          %243 = vector.bitcast %arg20 : vector<4xi8> to vector<4xf8E8M0FNU>
          %244 = vector.bitcast %arg19 : vector<4xi8> to vector<4xf8E8M0FNU>
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(0)
          rocdl.s.barrier
          %245 = affine.apply #map18()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%245], %alloc_1[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          %246 = affine.apply #map19()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%246], %alloc_1[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          %247 = affine.apply #map18()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %17[%247], %alloc[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          %248 = affine.apply #map19()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %17[%248], %alloc[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          rocdl.sched.barrier 0
          %249 = vector.load %reinterpret_cast_10[%33] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %250 = vector.load %reinterpret_cast_10[%34] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %251 = vector.load %reinterpret_cast_11[%35] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %252 = vector.load %reinterpret_cast_11[%36] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %253 = vector.load %reinterpret_cast_11[%37] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %254 = vector.load %reinterpret_cast_11[%38] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %255 = vector.bitcast %249 : vector<16xi8> to vector<32xf4E2M1FN>
          %256 = vector.bitcast %250 : vector<16xi8> to vector<32xf4E2M1FN>
          %257 = vector.bitcast %251 : vector<16xi8> to vector<32xf4E2M1FN>
          %258 = vector.bitcast %252 : vector<16xi8> to vector<32xf4E2M1FN>
          %259 = vector.bitcast %253 : vector<16xi8> to vector<32xf4E2M1FN>
          %260 = vector.bitcast %254 : vector<16xi8> to vector<32xf4E2M1FN>
          %261 = affine.apply #map20()[%block_id_x, %thread_id_x, %arg10]
          %262 = vector.load %19[%261] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %263 = vector.bitcast %262 : vector<4xi8> to vector<4xf8E8M0FNU>
          %264 = affine.apply #map21()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %265 = vector.load %22[%264] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %266 = vector.bitcast %265 : vector<4xi8> to vector<4xf8E8M0FNU>
          %267 = affine.apply #map22()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %268 = vector.load %22[%267] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %269 = vector.bitcast %268 : vector<4xi8> to vector<4xf8E8M0FNU>
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %270 = amdgpu.scaled_mfma 16x16x128 (%244[0] * %255) * (%243[0] * %257) + %arg11 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %271 = amdgpu.scaled_mfma 16x16x128 (%244[0] * %255) * (%243[1] * %258) + %arg12 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %272 = amdgpu.scaled_mfma 16x16x128 (%244[0] * %255) * (%242[0] * %259) + %arg13 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %273 = amdgpu.scaled_mfma 16x16x128 (%244[0] * %255) * (%242[1] * %260) + %arg14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %274 = amdgpu.scaled_mfma 16x16x128 (%244[1] * %256) * (%243[0] * %257) + %arg15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %275 = amdgpu.scaled_mfma 16x16x128 (%244[1] * %256) * (%243[1] * %258) + %arg16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %276 = amdgpu.scaled_mfma 16x16x128 (%244[1] * %256) * (%242[0] * %259) + %arg17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %277 = amdgpu.scaled_mfma 16x16x128 (%244[1] * %256) * (%242[1] * %260) + %arg18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.sched.barrier 0
          %278 = vector.load %reinterpret_cast_10[%41] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %279 = vector.load %reinterpret_cast_10[%42] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %280 = vector.load %reinterpret_cast_11[%43] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %281 = vector.load %reinterpret_cast_11[%44] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %282 = vector.load %reinterpret_cast_11[%45] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %283 = vector.load %reinterpret_cast_11[%46] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %284 = vector.bitcast %278 : vector<16xi8> to vector<32xf4E2M1FN>
          %285 = vector.bitcast %279 : vector<16xi8> to vector<32xf4E2M1FN>
          %286 = vector.bitcast %280 : vector<16xi8> to vector<32xf4E2M1FN>
          %287 = vector.bitcast %281 : vector<16xi8> to vector<32xf4E2M1FN>
          %288 = vector.bitcast %282 : vector<16xi8> to vector<32xf4E2M1FN>
          %289 = vector.bitcast %283 : vector<16xi8> to vector<32xf4E2M1FN>
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(3)
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %290 = amdgpu.scaled_mfma 16x16x128 (%244[2] * %284) * (%243[2] * %286) + %270 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %291 = amdgpu.scaled_mfma 16x16x128 (%244[2] * %284) * (%243[3] * %287) + %271 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %292 = amdgpu.scaled_mfma 16x16x128 (%244[2] * %284) * (%242[2] * %288) + %272 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %293 = amdgpu.scaled_mfma 16x16x128 (%244[2] * %284) * (%242[3] * %289) + %273 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %294 = amdgpu.scaled_mfma 16x16x128 (%244[3] * %285) * (%243[2] * %286) + %274 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %295 = amdgpu.scaled_mfma 16x16x128 (%244[3] * %285) * (%243[3] * %287) + %275 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %296 = amdgpu.scaled_mfma 16x16x128 (%244[3] * %285) * (%242[2] * %288) + %276 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %297 = amdgpu.scaled_mfma 16x16x128 (%244[3] * %285) * (%242[3] * %289) + %277 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          %298 = arith.addi %arg10, %c1 : index
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(0)
          rocdl.s.barrier
          %299 = affine.apply #map18()[%block_id_x, %thread_id_y, %thread_id_x, %298, %10]
          amdgpu.gather_to_lds %12[%299], %alloc_2[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          %300 = affine.apply #map19()[%block_id_x, %thread_id_y, %thread_id_x, %298, %10]
          amdgpu.gather_to_lds %12[%300], %alloc_2[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          %301 = affine.apply #map18()[%block_id_y, %thread_id_y, %thread_id_x, %298, %10]
          amdgpu.gather_to_lds %17[%301], %alloc_0[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          %302 = affine.apply #map19()[%block_id_y, %thread_id_y, %thread_id_x, %298, %10]
          amdgpu.gather_to_lds %17[%302], %alloc_0[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          rocdl.sched.barrier 0
          %303 = vector.load %reinterpret_cast_12[%33] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %304 = vector.load %reinterpret_cast_12[%34] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %305 = vector.load %reinterpret_cast_13[%35] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %306 = vector.load %reinterpret_cast_13[%36] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %307 = vector.load %reinterpret_cast_13[%37] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %308 = vector.load %reinterpret_cast_13[%38] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %309 = vector.bitcast %303 : vector<16xi8> to vector<32xf4E2M1FN>
          %310 = vector.bitcast %304 : vector<16xi8> to vector<32xf4E2M1FN>
          %311 = vector.bitcast %305 : vector<16xi8> to vector<32xf4E2M1FN>
          %312 = vector.bitcast %306 : vector<16xi8> to vector<32xf4E2M1FN>
          %313 = vector.bitcast %307 : vector<16xi8> to vector<32xf4E2M1FN>
          %314 = vector.bitcast %308 : vector<16xi8> to vector<32xf4E2M1FN>
          %315 = affine.apply #map20()[%block_id_x, %thread_id_x, %298]
          %316 = vector.load %19[%315] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %317 = affine.apply #map21()[%block_id_y, %thread_id_y, %298, %thread_id_x]
          %318 = vector.load %22[%317] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %319 = affine.apply #map22()[%block_id_y, %thread_id_y, %298, %thread_id_x]
          %320 = vector.load %22[%319] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %321 = amdgpu.scaled_mfma 16x16x128 (%263[0] * %309) * (%266[0] * %311) + %290 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %322 = amdgpu.scaled_mfma 16x16x128 (%263[0] * %309) * (%266[1] * %312) + %291 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %323 = amdgpu.scaled_mfma 16x16x128 (%263[0] * %309) * (%269[0] * %313) + %292 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %324 = amdgpu.scaled_mfma 16x16x128 (%263[0] * %309) * (%269[1] * %314) + %293 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %325 = amdgpu.scaled_mfma 16x16x128 (%263[1] * %310) * (%266[0] * %311) + %294 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %326 = amdgpu.scaled_mfma 16x16x128 (%263[1] * %310) * (%266[1] * %312) + %295 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %327 = amdgpu.scaled_mfma 16x16x128 (%263[1] * %310) * (%269[0] * %313) + %296 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %328 = amdgpu.scaled_mfma 16x16x128 (%263[1] * %310) * (%269[1] * %314) + %297 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.sched.barrier 0
          %329 = vector.load %reinterpret_cast_12[%41] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %330 = vector.load %reinterpret_cast_12[%42] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %331 = vector.load %reinterpret_cast_13[%43] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %332 = vector.load %reinterpret_cast_13[%44] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %333 = vector.load %reinterpret_cast_13[%45] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %334 = vector.load %reinterpret_cast_13[%46] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %335 = vector.bitcast %329 : vector<16xi8> to vector<32xf4E2M1FN>
          %336 = vector.bitcast %330 : vector<16xi8> to vector<32xf4E2M1FN>
          %337 = vector.bitcast %331 : vector<16xi8> to vector<32xf4E2M1FN>
          %338 = vector.bitcast %332 : vector<16xi8> to vector<32xf4E2M1FN>
          %339 = vector.bitcast %333 : vector<16xi8> to vector<32xf4E2M1FN>
          %340 = vector.bitcast %334 : vector<16xi8> to vector<32xf4E2M1FN>
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(3)
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %341 = amdgpu.scaled_mfma 16x16x128 (%263[2] * %335) * (%266[2] * %337) + %321 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %342 = amdgpu.scaled_mfma 16x16x128 (%263[2] * %335) * (%266[3] * %338) + %322 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %343 = amdgpu.scaled_mfma 16x16x128 (%263[2] * %335) * (%269[2] * %339) + %323 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %344 = amdgpu.scaled_mfma 16x16x128 (%263[2] * %335) * (%269[3] * %340) + %324 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %345 = amdgpu.scaled_mfma 16x16x128 (%263[3] * %336) * (%266[2] * %337) + %325 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %346 = amdgpu.scaled_mfma 16x16x128 (%263[3] * %336) * (%266[3] * %338) + %326 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %347 = amdgpu.scaled_mfma 16x16x128 (%263[3] * %336) * (%269[2] * %339) + %327 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %348 = amdgpu.scaled_mfma 16x16x128 (%263[3] * %336) * (%269[3] * %340) + %328 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          scf.yield %341, %342, %343, %344, %345, %346, %347, %348, %316, %318, %320 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>
        }
        %48 = vector.bitcast %47#10 : vector<4xi8> to vector<4xf8E8M0FNU>
        %49 = vector.bitcast %47#9 : vector<4xi8> to vector<4xf8E8M0FNU>
        %50 = vector.bitcast %47#8 : vector<4xi8> to vector<4xf8E8M0FNU>
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(0)
        rocdl.s.barrier
        %51 = affine.apply #map23()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%51], %alloc_1[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %52 = affine.apply #map24()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%52], %alloc_1[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %53 = affine.apply #map23()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %17[%53], %alloc[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %54 = affine.apply #map24()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %17[%54], %alloc[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        rocdl.sched.barrier 0
        %55 = vector.load %reinterpret_cast_10[%33] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %56 = vector.load %reinterpret_cast_10[%34] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %57 = vector.load %reinterpret_cast_11[%35] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %58 = vector.load %reinterpret_cast_11[%36] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %59 = vector.load %reinterpret_cast_11[%37] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %60 = vector.load %reinterpret_cast_11[%38] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %61 = vector.bitcast %55 : vector<16xi8> to vector<32xf4E2M1FN>
        %62 = vector.bitcast %56 : vector<16xi8> to vector<32xf4E2M1FN>
        %63 = vector.bitcast %57 : vector<16xi8> to vector<32xf4E2M1FN>
        %64 = vector.bitcast %58 : vector<16xi8> to vector<32xf4E2M1FN>
        %65 = vector.bitcast %59 : vector<16xi8> to vector<32xf4E2M1FN>
        %66 = vector.bitcast %60 : vector<16xi8> to vector<32xf4E2M1FN>
        %67 = affine.apply #map25()[%block_id_x, %thread_id_x]
        %68 = vector.load %19[%67] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %69 = vector.bitcast %68 : vector<4xi8> to vector<4xf8E8M0FNU>
        %70 = affine.apply #map26()[%block_id_y, %thread_id_y, %thread_id_x]
        %71 = vector.load %22[%70] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %72 = vector.bitcast %71 : vector<4xi8> to vector<4xf8E8M0FNU>
        %73 = affine.apply #map27()[%block_id_y, %thread_id_y, %thread_id_x]
        %74 = vector.load %22[%73] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %75 = vector.bitcast %74 : vector<4xi8> to vector<4xf8E8M0FNU>
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %76 = amdgpu.scaled_mfma 16x16x128 (%50[0] * %61) * (%49[0] * %63) + %47#0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %77 = amdgpu.scaled_mfma 16x16x128 (%50[0] * %61) * (%49[1] * %64) + %47#1 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %78 = amdgpu.scaled_mfma 16x16x128 (%50[0] * %61) * (%48[0] * %65) + %47#2 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %79 = amdgpu.scaled_mfma 16x16x128 (%50[0] * %61) * (%48[1] * %66) + %47#3 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %80 = amdgpu.scaled_mfma 16x16x128 (%50[1] * %62) * (%49[0] * %63) + %47#4 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %81 = amdgpu.scaled_mfma 16x16x128 (%50[1] * %62) * (%49[1] * %64) + %47#5 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %82 = amdgpu.scaled_mfma 16x16x128 (%50[1] * %62) * (%48[0] * %65) + %47#6 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %83 = amdgpu.scaled_mfma 16x16x128 (%50[1] * %62) * (%48[1] * %66) + %47#7 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        %84 = vector.load %reinterpret_cast_10[%41] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %85 = vector.load %reinterpret_cast_10[%42] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %86 = vector.load %reinterpret_cast_11[%43] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %87 = vector.load %reinterpret_cast_11[%44] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %88 = vector.load %reinterpret_cast_11[%45] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %89 = vector.load %reinterpret_cast_11[%46] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %90 = vector.bitcast %84 : vector<16xi8> to vector<32xf4E2M1FN>
        %91 = vector.bitcast %85 : vector<16xi8> to vector<32xf4E2M1FN>
        %92 = vector.bitcast %86 : vector<16xi8> to vector<32xf4E2M1FN>
        %93 = vector.bitcast %87 : vector<16xi8> to vector<32xf4E2M1FN>
        %94 = vector.bitcast %88 : vector<16xi8> to vector<32xf4E2M1FN>
        %95 = vector.bitcast %89 : vector<16xi8> to vector<32xf4E2M1FN>
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(3)
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %96 = amdgpu.scaled_mfma 16x16x128 (%50[2] * %90) * (%49[2] * %92) + %76 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %97 = amdgpu.scaled_mfma 16x16x128 (%50[2] * %90) * (%49[3] * %93) + %77 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %98 = amdgpu.scaled_mfma 16x16x128 (%50[2] * %90) * (%48[2] * %94) + %78 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %99 = amdgpu.scaled_mfma 16x16x128 (%50[2] * %90) * (%48[3] * %95) + %79 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %100 = amdgpu.scaled_mfma 16x16x128 (%50[3] * %91) * (%49[2] * %92) + %80 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %101 = amdgpu.scaled_mfma 16x16x128 (%50[3] * %91) * (%49[3] * %93) + %81 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %102 = amdgpu.scaled_mfma 16x16x128 (%50[3] * %91) * (%48[2] * %94) + %82 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %103 = amdgpu.scaled_mfma 16x16x128 (%50[3] * %91) * (%48[3] * %95) + %83 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        scf.if %30 {
          rocdl.s.barrier
        }
        amdgpu.lds_barrier
        %104 = vector.load %reinterpret_cast_13[%35] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %105 = vector.load %reinterpret_cast_13[%43] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %106 = vector.load %reinterpret_cast_13[%36] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %107 = vector.load %reinterpret_cast_13[%44] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %108 = vector.load %reinterpret_cast_13[%37] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %109 = vector.load %reinterpret_cast_13[%45] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %110 = vector.load %reinterpret_cast_13[%38] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %111 = vector.load %reinterpret_cast_13[%46] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %112 = vector.load %reinterpret_cast_12[%33] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %113 = vector.load %reinterpret_cast_12[%41] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %114 = vector.load %reinterpret_cast_12[%34] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %115 = vector.load %reinterpret_cast_12[%42] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %116 = vector.bitcast %112 : vector<16xi8> to vector<32xf4E2M1FN>
        %117 = vector.bitcast %113 : vector<16xi8> to vector<32xf4E2M1FN>
        %118 = vector.bitcast %114 : vector<16xi8> to vector<32xf4E2M1FN>
        %119 = vector.bitcast %115 : vector<16xi8> to vector<32xf4E2M1FN>
        %120 = vector.bitcast %104 : vector<16xi8> to vector<32xf4E2M1FN>
        %121 = vector.bitcast %105 : vector<16xi8> to vector<32xf4E2M1FN>
        %122 = vector.bitcast %106 : vector<16xi8> to vector<32xf4E2M1FN>
        %123 = vector.bitcast %107 : vector<16xi8> to vector<32xf4E2M1FN>
        %124 = vector.bitcast %108 : vector<16xi8> to vector<32xf4E2M1FN>
        %125 = vector.bitcast %109 : vector<16xi8> to vector<32xf4E2M1FN>
        %126 = vector.bitcast %110 : vector<16xi8> to vector<32xf4E2M1FN>
        %127 = vector.bitcast %111 : vector<16xi8> to vector<32xf4E2M1FN>
        %128 = amdgpu.scaled_mfma 16x16x128 (%69[0] * %116) * (%72[0] * %120) + %96 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %129 = amdgpu.scaled_mfma 16x16x128 (%69[2] * %117) * (%72[2] * %121) + %128 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %130 = amdgpu.scaled_mfma 16x16x128 (%69[0] * %116) * (%72[1] * %122) + %97 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %131 = amdgpu.scaled_mfma 16x16x128 (%69[2] * %117) * (%72[3] * %123) + %130 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %132 = amdgpu.scaled_mfma 16x16x128 (%69[0] * %116) * (%75[0] * %124) + %98 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %133 = amdgpu.scaled_mfma 16x16x128 (%69[2] * %117) * (%75[2] * %125) + %132 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %134 = amdgpu.scaled_mfma 16x16x128 (%69[0] * %116) * (%75[1] * %126) + %99 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %135 = amdgpu.scaled_mfma 16x16x128 (%69[2] * %117) * (%75[3] * %127) + %134 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %136 = amdgpu.scaled_mfma 16x16x128 (%69[1] * %118) * (%72[0] * %120) + %100 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %137 = amdgpu.scaled_mfma 16x16x128 (%69[3] * %119) * (%72[2] * %121) + %136 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %138 = amdgpu.scaled_mfma 16x16x128 (%69[1] * %118) * (%72[1] * %122) + %101 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %139 = amdgpu.scaled_mfma 16x16x128 (%69[3] * %119) * (%72[3] * %123) + %138 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %140 = amdgpu.scaled_mfma 16x16x128 (%69[1] * %118) * (%75[0] * %124) + %102 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %141 = amdgpu.scaled_mfma 16x16x128 (%69[3] * %119) * (%75[2] * %125) + %140 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %142 = amdgpu.scaled_mfma 16x16x128 (%69[1] * %118) * (%75[1] * %126) + %103 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %143 = amdgpu.scaled_mfma 16x16x128 (%69[3] * %119) * (%75[3] * %127) + %142 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        // ── Optimized epilogue: XOR-1 shuffle → vector<2xbf16> stores ──────
        %ep_blkx = affine.apply #map28()[%block_id_x]
        %ep_blky = affine.apply #map28()[%block_id_y]
        %ep_bbase, %ep_boff, %ep_bsz:2, %ep_stride:2 = memref.extract_strided_metadata %reinterpret_cast : memref<2048x2048xbf16, strided<[?, 1]>> -> memref<bf16>, index, index, index, index, index
        %ep_blk_roff = arith.muli %ep_blkx, %ep_stride#0 overflow<nsw> : index
        %ep_base_off = arith.addi %ep_blk_roff, %ep_blky overflow<nsw> : index
        %ep_rcast = memref.reinterpret_cast %4 to offset: [%ep_base_off], sizes: [1073741822], strides: [1] : memref<bf16> to memref<1073741822xbf16, strided<[1], offset: ?>>
        %ep_fcast = memref.cast %ep_rcast : memref<1073741822xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
        %ep_si14 = arith.index_cast %ep_stride#0 : index to i14
        %ep_buf = amdgpu.fat_raw_buffer_cast %ep_fcast validBytes(%c2147483645_i64) cacheSwizzleStride(%ep_si14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
        %ep_xor_off = arith.constant 1 : i32
        %ep_xor_w   = arith.constant 64 : i32
        %ep_lp_one  = arith.constant 1 : index
        %ep_lp_zero = arith.constant 0 : index
        %ep_lparity = arith.andi %thread_id_x, %ep_lp_one : index
        %ep_is_even = arith.cmpi eq, %ep_lparity, %ep_lp_zero : index
        %ep_rb0_r0 = affine.apply #map29()[%thread_id_x]
        %ep_rb0_r1 = affine.apply #map31()[%thread_id_x]
        %ep_rb0_r2 = affine.apply #map32()[%thread_id_x]
        %ep_rb0_r3 = affine.apply #map33()[%thread_id_x]
        %ep_rb0_ra = arith.select %ep_is_even, %ep_rb0_r0, %ep_rb0_r2 : index
        %ep_rb0_rb = arith.select %ep_is_even, %ep_rb0_r1, %ep_rb0_r3 : index
        %ep_rb0_oa = arith.muli %ep_rb0_ra, %ep_stride#0 overflow<nsw> : index
        %ep_rb0_ob = arith.muli %ep_rb0_rb, %ep_stride#0 overflow<nsw> : index
        %ep_rb1_r0 = affine.apply #map37()[%thread_id_x]
        %ep_rb1_r1 = affine.apply #map38()[%thread_id_x]
        %ep_rb1_r2 = affine.apply #map39()[%thread_id_x]
        %ep_rb1_r3 = affine.apply #map40()[%thread_id_x]
        %ep_rb1_ra = arith.select %ep_is_even, %ep_rb1_r0, %ep_rb1_r2 : index
        %ep_rb1_rb = arith.select %ep_is_even, %ep_rb1_r1, %ep_rb1_r3 : index
        %ep_rb1_oa = arith.muli %ep_rb1_ra, %ep_stride#0 overflow<nsw> : index
        %ep_rb1_ob = arith.muli %ep_rb1_rb, %ep_stride#0 overflow<nsw> : index
        %ep_c0 = affine.apply #map30()[%thread_id_x, %thread_id_y]
        %ep_c0a = arith.subi %ep_c0, %ep_lparity : index
        %ep_c1 = affine.apply #map34()[%thread_id_x, %thread_id_y]
        %ep_c1a = arith.subi %ep_c1, %ep_lparity : index
        %ep_c2 = affine.apply #map35()[%thread_id_x, %thread_id_y]
        %ep_c2a = arith.subi %ep_c2, %ep_lparity : index
        %ep_c3 = affine.apply #map36()[%thread_id_x, %thread_id_y]
        %ep_c3a = arith.subi %ep_c3, %ep_lparity : index

        // MFMA %129 → rb0, col0
        %v129_e0 = vector.extract %129[0] : f32 from vector<4xf32>
        %v129_e1 = vector.extract %129[1] : f32 from vector<4xf32>
        %v129_e2 = vector.extract %129[2] : f32 from vector<4xf32>
        %v129_e3 = vector.extract %129[3] : f32 from vector<4xf32>
        %v129_e0n, %v129_e0v = gpu.shuffle xor %v129_e0, %ep_xor_off, %ep_xor_w : f32
        %v129_e1n, %v129_e1v = gpu.shuffle xor %v129_e1, %ep_xor_off, %ep_xor_w : f32
        %v129_e2n, %v129_e2v = gpu.shuffle xor %v129_e2, %ep_xor_off, %ep_xor_w : f32
        %v129_e3n, %v129_e3v = gpu.shuffle xor %v129_e3, %ep_xor_off, %ep_xor_w : f32
        %v129_e0lo = arith.select %ep_is_even, %v129_e0,  %v129_e0n : f32
        %v129_e0hi = arith.select %ep_is_even, %v129_e0n, %v129_e0  : f32
        %v129_e1lo = arith.select %ep_is_even, %v129_e1,  %v129_e1n : f32
        %v129_e1hi = arith.select %ep_is_even, %v129_e1n, %v129_e1  : f32
        %v129_e2lo = arith.select %ep_is_even, %v129_e2,  %v129_e2n : f32
        %v129_e2hi = arith.select %ep_is_even, %v129_e2n, %v129_e2  : f32
        %v129_e3lo = arith.select %ep_is_even, %v129_e3,  %v129_e3n : f32
        %v129_e3hi = arith.select %ep_is_even, %v129_e3n, %v129_e3  : f32
        %v129_alo = arith.select %ep_is_even, %v129_e0lo, %v129_e2lo : f32
        %v129_ahi = arith.select %ep_is_even, %v129_e0hi, %v129_e2hi : f32
        %v129_blo = arith.select %ep_is_even, %v129_e1lo, %v129_e3lo : f32
        %v129_bhi = arith.select %ep_is_even, %v129_e1hi, %v129_e3hi : f32
        %v129_va0 = vector.broadcast %v129_alo : f32 to vector<2xf32>
        %v129_va  = vector.insert %v129_ahi, %v129_va0 [1] : f32 into vector<2xf32>
        %v129_vab = arith.truncf %v129_va : vector<2xf32> to vector<2xbf16>
        %v129_vb0 = vector.broadcast %v129_blo : f32 to vector<2xf32>
        %v129_vb  = vector.insert %v129_bhi, %v129_vb0 [1] : f32 into vector<2xf32>
        %v129_vbb = arith.truncf %v129_vb : vector<2xf32> to vector<2xbf16>
        %v129_adra = arith.addi %ep_rb0_oa, %ep_c0a overflow<nsw> : index
        %v129_adrb = arith.addi %ep_rb0_ob, %ep_c0a overflow<nsw> : index
        vector.store %v129_vab, %ep_buf[%v129_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v129_vbb, %ep_buf[%v129_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %131 → rb0, col1
        %v131_e0 = vector.extract %131[0] : f32 from vector<4xf32>
        %v131_e1 = vector.extract %131[1] : f32 from vector<4xf32>
        %v131_e2 = vector.extract %131[2] : f32 from vector<4xf32>
        %v131_e3 = vector.extract %131[3] : f32 from vector<4xf32>
        %v131_e0n, %v131_e0v = gpu.shuffle xor %v131_e0, %ep_xor_off, %ep_xor_w : f32
        %v131_e1n, %v131_e1v = gpu.shuffle xor %v131_e1, %ep_xor_off, %ep_xor_w : f32
        %v131_e2n, %v131_e2v = gpu.shuffle xor %v131_e2, %ep_xor_off, %ep_xor_w : f32
        %v131_e3n, %v131_e3v = gpu.shuffle xor %v131_e3, %ep_xor_off, %ep_xor_w : f32
        %v131_e0lo = arith.select %ep_is_even, %v131_e0,  %v131_e0n : f32
        %v131_e0hi = arith.select %ep_is_even, %v131_e0n, %v131_e0  : f32
        %v131_e1lo = arith.select %ep_is_even, %v131_e1,  %v131_e1n : f32
        %v131_e1hi = arith.select %ep_is_even, %v131_e1n, %v131_e1  : f32
        %v131_e2lo = arith.select %ep_is_even, %v131_e2,  %v131_e2n : f32
        %v131_e2hi = arith.select %ep_is_even, %v131_e2n, %v131_e2  : f32
        %v131_e3lo = arith.select %ep_is_even, %v131_e3,  %v131_e3n : f32
        %v131_e3hi = arith.select %ep_is_even, %v131_e3n, %v131_e3  : f32
        %v131_alo = arith.select %ep_is_even, %v131_e0lo, %v131_e2lo : f32
        %v131_ahi = arith.select %ep_is_even, %v131_e0hi, %v131_e2hi : f32
        %v131_blo = arith.select %ep_is_even, %v131_e1lo, %v131_e3lo : f32
        %v131_bhi = arith.select %ep_is_even, %v131_e1hi, %v131_e3hi : f32
        %v131_va0 = vector.broadcast %v131_alo : f32 to vector<2xf32>
        %v131_va  = vector.insert %v131_ahi, %v131_va0 [1] : f32 into vector<2xf32>
        %v131_vab = arith.truncf %v131_va : vector<2xf32> to vector<2xbf16>
        %v131_vb0 = vector.broadcast %v131_blo : f32 to vector<2xf32>
        %v131_vb  = vector.insert %v131_bhi, %v131_vb0 [1] : f32 into vector<2xf32>
        %v131_vbb = arith.truncf %v131_vb : vector<2xf32> to vector<2xbf16>
        %v131_adra = arith.addi %ep_rb0_oa, %ep_c1a overflow<nsw> : index
        %v131_adrb = arith.addi %ep_rb0_ob, %ep_c1a overflow<nsw> : index
        vector.store %v131_vab, %ep_buf[%v131_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v131_vbb, %ep_buf[%v131_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %133 → rb0, col2
        %v133_e0 = vector.extract %133[0] : f32 from vector<4xf32>
        %v133_e1 = vector.extract %133[1] : f32 from vector<4xf32>
        %v133_e2 = vector.extract %133[2] : f32 from vector<4xf32>
        %v133_e3 = vector.extract %133[3] : f32 from vector<4xf32>
        %v133_e0n, %v133_e0v = gpu.shuffle xor %v133_e0, %ep_xor_off, %ep_xor_w : f32
        %v133_e1n, %v133_e1v = gpu.shuffle xor %v133_e1, %ep_xor_off, %ep_xor_w : f32
        %v133_e2n, %v133_e2v = gpu.shuffle xor %v133_e2, %ep_xor_off, %ep_xor_w : f32
        %v133_e3n, %v133_e3v = gpu.shuffle xor %v133_e3, %ep_xor_off, %ep_xor_w : f32
        %v133_e0lo = arith.select %ep_is_even, %v133_e0,  %v133_e0n : f32
        %v133_e0hi = arith.select %ep_is_even, %v133_e0n, %v133_e0  : f32
        %v133_e1lo = arith.select %ep_is_even, %v133_e1,  %v133_e1n : f32
        %v133_e1hi = arith.select %ep_is_even, %v133_e1n, %v133_e1  : f32
        %v133_e2lo = arith.select %ep_is_even, %v133_e2,  %v133_e2n : f32
        %v133_e2hi = arith.select %ep_is_even, %v133_e2n, %v133_e2  : f32
        %v133_e3lo = arith.select %ep_is_even, %v133_e3,  %v133_e3n : f32
        %v133_e3hi = arith.select %ep_is_even, %v133_e3n, %v133_e3  : f32
        %v133_alo = arith.select %ep_is_even, %v133_e0lo, %v133_e2lo : f32
        %v133_ahi = arith.select %ep_is_even, %v133_e0hi, %v133_e2hi : f32
        %v133_blo = arith.select %ep_is_even, %v133_e1lo, %v133_e3lo : f32
        %v133_bhi = arith.select %ep_is_even, %v133_e1hi, %v133_e3hi : f32
        %v133_va0 = vector.broadcast %v133_alo : f32 to vector<2xf32>
        %v133_va  = vector.insert %v133_ahi, %v133_va0 [1] : f32 into vector<2xf32>
        %v133_vab = arith.truncf %v133_va : vector<2xf32> to vector<2xbf16>
        %v133_vb0 = vector.broadcast %v133_blo : f32 to vector<2xf32>
        %v133_vb  = vector.insert %v133_bhi, %v133_vb0 [1] : f32 into vector<2xf32>
        %v133_vbb = arith.truncf %v133_vb : vector<2xf32> to vector<2xbf16>
        %v133_adra = arith.addi %ep_rb0_oa, %ep_c2a overflow<nsw> : index
        %v133_adrb = arith.addi %ep_rb0_ob, %ep_c2a overflow<nsw> : index
        vector.store %v133_vab, %ep_buf[%v133_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v133_vbb, %ep_buf[%v133_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %135 → rb0, col3
        %v135_e0 = vector.extract %135[0] : f32 from vector<4xf32>
        %v135_e1 = vector.extract %135[1] : f32 from vector<4xf32>
        %v135_e2 = vector.extract %135[2] : f32 from vector<4xf32>
        %v135_e3 = vector.extract %135[3] : f32 from vector<4xf32>
        %v135_e0n, %v135_e0v = gpu.shuffle xor %v135_e0, %ep_xor_off, %ep_xor_w : f32
        %v135_e1n, %v135_e1v = gpu.shuffle xor %v135_e1, %ep_xor_off, %ep_xor_w : f32
        %v135_e2n, %v135_e2v = gpu.shuffle xor %v135_e2, %ep_xor_off, %ep_xor_w : f32
        %v135_e3n, %v135_e3v = gpu.shuffle xor %v135_e3, %ep_xor_off, %ep_xor_w : f32
        %v135_e0lo = arith.select %ep_is_even, %v135_e0,  %v135_e0n : f32
        %v135_e0hi = arith.select %ep_is_even, %v135_e0n, %v135_e0  : f32
        %v135_e1lo = arith.select %ep_is_even, %v135_e1,  %v135_e1n : f32
        %v135_e1hi = arith.select %ep_is_even, %v135_e1n, %v135_e1  : f32
        %v135_e2lo = arith.select %ep_is_even, %v135_e2,  %v135_e2n : f32
        %v135_e2hi = arith.select %ep_is_even, %v135_e2n, %v135_e2  : f32
        %v135_e3lo = arith.select %ep_is_even, %v135_e3,  %v135_e3n : f32
        %v135_e3hi = arith.select %ep_is_even, %v135_e3n, %v135_e3  : f32
        %v135_alo = arith.select %ep_is_even, %v135_e0lo, %v135_e2lo : f32
        %v135_ahi = arith.select %ep_is_even, %v135_e0hi, %v135_e2hi : f32
        %v135_blo = arith.select %ep_is_even, %v135_e1lo, %v135_e3lo : f32
        %v135_bhi = arith.select %ep_is_even, %v135_e1hi, %v135_e3hi : f32
        %v135_va0 = vector.broadcast %v135_alo : f32 to vector<2xf32>
        %v135_va  = vector.insert %v135_ahi, %v135_va0 [1] : f32 into vector<2xf32>
        %v135_vab = arith.truncf %v135_va : vector<2xf32> to vector<2xbf16>
        %v135_vb0 = vector.broadcast %v135_blo : f32 to vector<2xf32>
        %v135_vb  = vector.insert %v135_bhi, %v135_vb0 [1] : f32 into vector<2xf32>
        %v135_vbb = arith.truncf %v135_vb : vector<2xf32> to vector<2xbf16>
        %v135_adra = arith.addi %ep_rb0_oa, %ep_c3a overflow<nsw> : index
        %v135_adrb = arith.addi %ep_rb0_ob, %ep_c3a overflow<nsw> : index
        vector.store %v135_vab, %ep_buf[%v135_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v135_vbb, %ep_buf[%v135_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %137 → rb1, col0
        %v137_e0 = vector.extract %137[0] : f32 from vector<4xf32>
        %v137_e1 = vector.extract %137[1] : f32 from vector<4xf32>
        %v137_e2 = vector.extract %137[2] : f32 from vector<4xf32>
        %v137_e3 = vector.extract %137[3] : f32 from vector<4xf32>
        %v137_e0n, %v137_e0v = gpu.shuffle xor %v137_e0, %ep_xor_off, %ep_xor_w : f32
        %v137_e1n, %v137_e1v = gpu.shuffle xor %v137_e1, %ep_xor_off, %ep_xor_w : f32
        %v137_e2n, %v137_e2v = gpu.shuffle xor %v137_e2, %ep_xor_off, %ep_xor_w : f32
        %v137_e3n, %v137_e3v = gpu.shuffle xor %v137_e3, %ep_xor_off, %ep_xor_w : f32
        %v137_e0lo = arith.select %ep_is_even, %v137_e0,  %v137_e0n : f32
        %v137_e0hi = arith.select %ep_is_even, %v137_e0n, %v137_e0  : f32
        %v137_e1lo = arith.select %ep_is_even, %v137_e1,  %v137_e1n : f32
        %v137_e1hi = arith.select %ep_is_even, %v137_e1n, %v137_e1  : f32
        %v137_e2lo = arith.select %ep_is_even, %v137_e2,  %v137_e2n : f32
        %v137_e2hi = arith.select %ep_is_even, %v137_e2n, %v137_e2  : f32
        %v137_e3lo = arith.select %ep_is_even, %v137_e3,  %v137_e3n : f32
        %v137_e3hi = arith.select %ep_is_even, %v137_e3n, %v137_e3  : f32
        %v137_alo = arith.select %ep_is_even, %v137_e0lo, %v137_e2lo : f32
        %v137_ahi = arith.select %ep_is_even, %v137_e0hi, %v137_e2hi : f32
        %v137_blo = arith.select %ep_is_even, %v137_e1lo, %v137_e3lo : f32
        %v137_bhi = arith.select %ep_is_even, %v137_e1hi, %v137_e3hi : f32
        %v137_va0 = vector.broadcast %v137_alo : f32 to vector<2xf32>
        %v137_va  = vector.insert %v137_ahi, %v137_va0 [1] : f32 into vector<2xf32>
        %v137_vab = arith.truncf %v137_va : vector<2xf32> to vector<2xbf16>
        %v137_vb0 = vector.broadcast %v137_blo : f32 to vector<2xf32>
        %v137_vb  = vector.insert %v137_bhi, %v137_vb0 [1] : f32 into vector<2xf32>
        %v137_vbb = arith.truncf %v137_vb : vector<2xf32> to vector<2xbf16>
        %v137_adra = arith.addi %ep_rb1_oa, %ep_c0a overflow<nsw> : index
        %v137_adrb = arith.addi %ep_rb1_ob, %ep_c0a overflow<nsw> : index
        vector.store %v137_vab, %ep_buf[%v137_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v137_vbb, %ep_buf[%v137_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %139 → rb1, col1
        %v139_e0 = vector.extract %139[0] : f32 from vector<4xf32>
        %v139_e1 = vector.extract %139[1] : f32 from vector<4xf32>
        %v139_e2 = vector.extract %139[2] : f32 from vector<4xf32>
        %v139_e3 = vector.extract %139[3] : f32 from vector<4xf32>
        %v139_e0n, %v139_e0v = gpu.shuffle xor %v139_e0, %ep_xor_off, %ep_xor_w : f32
        %v139_e1n, %v139_e1v = gpu.shuffle xor %v139_e1, %ep_xor_off, %ep_xor_w : f32
        %v139_e2n, %v139_e2v = gpu.shuffle xor %v139_e2, %ep_xor_off, %ep_xor_w : f32
        %v139_e3n, %v139_e3v = gpu.shuffle xor %v139_e3, %ep_xor_off, %ep_xor_w : f32
        %v139_e0lo = arith.select %ep_is_even, %v139_e0,  %v139_e0n : f32
        %v139_e0hi = arith.select %ep_is_even, %v139_e0n, %v139_e0  : f32
        %v139_e1lo = arith.select %ep_is_even, %v139_e1,  %v139_e1n : f32
        %v139_e1hi = arith.select %ep_is_even, %v139_e1n, %v139_e1  : f32
        %v139_e2lo = arith.select %ep_is_even, %v139_e2,  %v139_e2n : f32
        %v139_e2hi = arith.select %ep_is_even, %v139_e2n, %v139_e2  : f32
        %v139_e3lo = arith.select %ep_is_even, %v139_e3,  %v139_e3n : f32
        %v139_e3hi = arith.select %ep_is_even, %v139_e3n, %v139_e3  : f32
        %v139_alo = arith.select %ep_is_even, %v139_e0lo, %v139_e2lo : f32
        %v139_ahi = arith.select %ep_is_even, %v139_e0hi, %v139_e2hi : f32
        %v139_blo = arith.select %ep_is_even, %v139_e1lo, %v139_e3lo : f32
        %v139_bhi = arith.select %ep_is_even, %v139_e1hi, %v139_e3hi : f32
        %v139_va0 = vector.broadcast %v139_alo : f32 to vector<2xf32>
        %v139_va  = vector.insert %v139_ahi, %v139_va0 [1] : f32 into vector<2xf32>
        %v139_vab = arith.truncf %v139_va : vector<2xf32> to vector<2xbf16>
        %v139_vb0 = vector.broadcast %v139_blo : f32 to vector<2xf32>
        %v139_vb  = vector.insert %v139_bhi, %v139_vb0 [1] : f32 into vector<2xf32>
        %v139_vbb = arith.truncf %v139_vb : vector<2xf32> to vector<2xbf16>
        %v139_adra = arith.addi %ep_rb1_oa, %ep_c1a overflow<nsw> : index
        %v139_adrb = arith.addi %ep_rb1_ob, %ep_c1a overflow<nsw> : index
        vector.store %v139_vab, %ep_buf[%v139_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v139_vbb, %ep_buf[%v139_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %141 → rb1, col2
        %v141_e0 = vector.extract %141[0] : f32 from vector<4xf32>
        %v141_e1 = vector.extract %141[1] : f32 from vector<4xf32>
        %v141_e2 = vector.extract %141[2] : f32 from vector<4xf32>
        %v141_e3 = vector.extract %141[3] : f32 from vector<4xf32>
        %v141_e0n, %v141_e0v = gpu.shuffle xor %v141_e0, %ep_xor_off, %ep_xor_w : f32
        %v141_e1n, %v141_e1v = gpu.shuffle xor %v141_e1, %ep_xor_off, %ep_xor_w : f32
        %v141_e2n, %v141_e2v = gpu.shuffle xor %v141_e2, %ep_xor_off, %ep_xor_w : f32
        %v141_e3n, %v141_e3v = gpu.shuffle xor %v141_e3, %ep_xor_off, %ep_xor_w : f32
        %v141_e0lo = arith.select %ep_is_even, %v141_e0,  %v141_e0n : f32
        %v141_e0hi = arith.select %ep_is_even, %v141_e0n, %v141_e0  : f32
        %v141_e1lo = arith.select %ep_is_even, %v141_e1,  %v141_e1n : f32
        %v141_e1hi = arith.select %ep_is_even, %v141_e1n, %v141_e1  : f32
        %v141_e2lo = arith.select %ep_is_even, %v141_e2,  %v141_e2n : f32
        %v141_e2hi = arith.select %ep_is_even, %v141_e2n, %v141_e2  : f32
        %v141_e3lo = arith.select %ep_is_even, %v141_e3,  %v141_e3n : f32
        %v141_e3hi = arith.select %ep_is_even, %v141_e3n, %v141_e3  : f32
        %v141_alo = arith.select %ep_is_even, %v141_e0lo, %v141_e2lo : f32
        %v141_ahi = arith.select %ep_is_even, %v141_e0hi, %v141_e2hi : f32
        %v141_blo = arith.select %ep_is_even, %v141_e1lo, %v141_e3lo : f32
        %v141_bhi = arith.select %ep_is_even, %v141_e1hi, %v141_e3hi : f32
        %v141_va0 = vector.broadcast %v141_alo : f32 to vector<2xf32>
        %v141_va  = vector.insert %v141_ahi, %v141_va0 [1] : f32 into vector<2xf32>
        %v141_vab = arith.truncf %v141_va : vector<2xf32> to vector<2xbf16>
        %v141_vb0 = vector.broadcast %v141_blo : f32 to vector<2xf32>
        %v141_vb  = vector.insert %v141_bhi, %v141_vb0 [1] : f32 into vector<2xf32>
        %v141_vbb = arith.truncf %v141_vb : vector<2xf32> to vector<2xbf16>
        %v141_adra = arith.addi %ep_rb1_oa, %ep_c2a overflow<nsw> : index
        %v141_adrb = arith.addi %ep_rb1_ob, %ep_c2a overflow<nsw> : index
        vector.store %v141_vab, %ep_buf[%v141_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v141_vbb, %ep_buf[%v141_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %143 → rb1, col3
        %v143_e0 = vector.extract %143[0] : f32 from vector<4xf32>
        %v143_e1 = vector.extract %143[1] : f32 from vector<4xf32>
        %v143_e2 = vector.extract %143[2] : f32 from vector<4xf32>
        %v143_e3 = vector.extract %143[3] : f32 from vector<4xf32>
        %v143_e0n, %v143_e0v = gpu.shuffle xor %v143_e0, %ep_xor_off, %ep_xor_w : f32
        %v143_e1n, %v143_e1v = gpu.shuffle xor %v143_e1, %ep_xor_off, %ep_xor_w : f32
        %v143_e2n, %v143_e2v = gpu.shuffle xor %v143_e2, %ep_xor_off, %ep_xor_w : f32
        %v143_e3n, %v143_e3v = gpu.shuffle xor %v143_e3, %ep_xor_off, %ep_xor_w : f32
        %v143_e0lo = arith.select %ep_is_even, %v143_e0,  %v143_e0n : f32
        %v143_e0hi = arith.select %ep_is_even, %v143_e0n, %v143_e0  : f32
        %v143_e1lo = arith.select %ep_is_even, %v143_e1,  %v143_e1n : f32
        %v143_e1hi = arith.select %ep_is_even, %v143_e1n, %v143_e1  : f32
        %v143_e2lo = arith.select %ep_is_even, %v143_e2,  %v143_e2n : f32
        %v143_e2hi = arith.select %ep_is_even, %v143_e2n, %v143_e2  : f32
        %v143_e3lo = arith.select %ep_is_even, %v143_e3,  %v143_e3n : f32
        %v143_e3hi = arith.select %ep_is_even, %v143_e3n, %v143_e3  : f32
        %v143_alo = arith.select %ep_is_even, %v143_e0lo, %v143_e2lo : f32
        %v143_ahi = arith.select %ep_is_even, %v143_e0hi, %v143_e2hi : f32
        %v143_blo = arith.select %ep_is_even, %v143_e1lo, %v143_e3lo : f32
        %v143_bhi = arith.select %ep_is_even, %v143_e1hi, %v143_e3hi : f32
        %v143_va0 = vector.broadcast %v143_alo : f32 to vector<2xf32>
        %v143_va  = vector.insert %v143_ahi, %v143_va0 [1] : f32 into vector<2xf32>
        %v143_vab = arith.truncf %v143_va : vector<2xf32> to vector<2xbf16>
        %v143_vb0 = vector.broadcast %v143_blo : f32 to vector<2xf32>
        %v143_vb  = vector.insert %v143_bhi, %v143_vb0 [1] : f32 into vector<2xf32>
        %v143_vbb = arith.truncf %v143_vb : vector<2xf32> to vector<2xbf16>
        %v143_adra = arith.addi %ep_rb1_oa, %ep_c3a overflow<nsw> : index
        %v143_adrb = arith.addi %ep_rb1_ob, %ep_c3a overflow<nsw> : index
        vector.store %v143_vab, %ep_buf[%v143_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v143_vbb, %ep_buf[%v143_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: !hal.fence, %arg11: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg10) => %arg0 : !hal.buffer_view -> tensor<2048x4096xi8>
    %1 = hal.tensor.import wait(%arg10) => %arg1 : !hal.buffer_view -> tensor<2048x256xi8>
    %2 = hal.tensor.import wait(%arg10) => %arg2 : !hal.buffer_view -> tensor<2048x4096xi8>
    %3 = hal.tensor.import wait(%arg10) => %arg3 : !hal.buffer_view -> tensor<2048x256xi8>
    %4 = hal.tensor.import wait(%arg10) => %arg4 : !hal.buffer_view -> tensor<2048x2048xbf16>
    %5 = flow.dispatch @gemm::@gemm[%arg5, %arg6, %arg7, %arg8, %arg9](%0, %1, %2, %3, %4, %arg5, %arg6, %arg7, %arg8, %arg9) : (tensor<2048x4096xi8>, tensor<2048x256xi8>, tensor<2048x4096xi8>, tensor<2048x256xi8>, tensor<2048x2048xbf16>, index, index, index, index, index) -> %4
    %6 = hal.tensor.barrier join(%5 : tensor<2048x2048xbf16>) => %arg11 : !hal.fence
    %7 = hal.tensor.export %6 : tensor<2048x2048xbf16> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
