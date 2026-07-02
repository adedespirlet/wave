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
        // ── Transposed epilogue: permlane + bf16-shuffle opt ────────────────
        %tp_blkr = affine.apply #map28()[%block_id_y]
        %tp_blkc = affine.apply #map28()[%block_id_x]
        %tp_bbase, %tp_boff, %tp_bsz:2, %tp_strd:2 = memref.extract_strided_metadata %reinterpret_cast : memref<2048x2048xbf16, strided<[?, 1]>> -> memref<bf16>, index, index, index, index, index
        %tp_blk_roff = arith.muli %tp_blkr, %tp_strd#0 overflow<nsw> : index
        %tp_base_off = arith.addi %tp_blk_roff, %tp_blkc overflow<nsw> : index
        %tp_rcast = memref.reinterpret_cast %4 to offset: [%tp_base_off], sizes: [1073741822], strides: [1] : memref<bf16> to memref<1073741822xbf16, strided<[1], offset: ?>>
        %tp_fcast = memref.cast %tp_rcast : memref<1073741822xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
        %tp_si14 = arith.index_cast %tp_strd#0 : index to i14
        %tp_buf = amdgpu.fat_raw_buffer_cast %tp_fcast validBytes(%c2147483645_i64) cacheSwizzleStride(%tp_si14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
        %tp_c0  = arith.constant 0 : index
        %tp_c8  = arith.constant 8 : index
        %tp_c16 = arith.constant 16 : index
        %tp_c32 = arith.constant 32 : index
        %tp_c48 = arith.constant 48 : index
        %tp_c64 = arith.constant 64 : index
        %tp_tx32 = arith.remui %thread_id_x, %tp_c32 : index
        %tp_lpar = arith.divui %tp_tx32, %tp_c16 : index
        %tp_even = arith.cmpi eq, %tp_lpar, %tp_c0 : index
        %tp_mbase = affine.apply #map30()[%thread_id_x, %thread_id_y]
        %tp_mrow = arith.muli %tp_mbase, %tp_strd#0 overflow<nsw> : index
        %tp_tx_d64 = arith.divui %thread_id_x, %tp_c64 : index
        %tp_nb_w = arith.muli %tp_tx_d64, %tp_c32 : index
        %tp_tx64 = arith.remui %thread_id_x, %tp_c64 : index
        %tp_tx64d32 = arith.divui %tp_tx64, %tp_c32 : index
        %tp_nb_s = arith.muli %tp_tx64d32, %tp_c8 : index
        %tp_nb0 = arith.addi %tp_nb_w, %tp_nb_s overflow<nsw> : index
        %tp_nb1 = arith.addi %tp_nb0, %tp_c16 overflow<nsw> : index
        %tp_mt0 = arith.constant 0 : index
        %tp_mt1 = arith.muli %tp_c16, %tp_strd#0 overflow<nsw> : index
        %tp_mt2 = arith.muli %tp_c32, %tp_strd#0 overflow<nsw> : index
        %tp_mt3 = arith.muli %tp_c48, %tp_strd#0 overflow<nsw> : index
        %tp_bn0 = arith.addi %tp_mrow, %tp_nb0 overflow<nsw> : index
        %tp_bn1 = arith.addi %tp_mrow, %tp_nb1 overflow<nsw> : index
        // === Pair 0 (rb0,col0-col1): %129, %131 ===
        %v129_b16 = arith.truncf %129 : vector<4xf32> to vector<4xbf16>
        %v131_b16 = arith.truncf %131 : vector<4xf32> to vector<4xbf16>
        %tp0_a01v = vector.extract_strided_slice %v129_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp0_a23v = vector.extract_strided_slice %v129_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp0_a01r = vector.bitcast %tp0_a01v : vector<2xbf16> to vector<1xi32>
        %tp0_a23r = vector.bitcast %tp0_a23v : vector<2xbf16> to vector<1xi32>
        %tp0_a01i = vector.extract %tp0_a01r[0] : i32 from vector<1xi32>
        %tp0_a23i = vector.extract %tp0_a23r[0] : i32 from vector<1xi32>
        %tp0_a01s = amdgpu.permlane_swap %tp0_a01i 16 : i32
        %tp0_a23s = amdgpu.permlane_swap %tp0_a23i 16 : i32
        %tp0_b01v = vector.extract_strided_slice %v131_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp0_b23v = vector.extract_strided_slice %v131_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp0_b01r = vector.bitcast %tp0_b01v : vector<2xbf16> to vector<1xi32>
        %tp0_b23r = vector.bitcast %tp0_b23v : vector<2xbf16> to vector<1xi32>
        %tp0_b01i = vector.extract %tp0_b01r[0] : i32 from vector<1xi32>
        %tp0_b23i = vector.extract %tp0_b23r[0] : i32 from vector<1xi32>
        %tp0_b01s = amdgpu.permlane_swap %tp0_b01i 16 : i32
        %tp0_b23s = amdgpu.permlane_swap %tp0_b23i 16 : i32
        %tp0_a01_e = arith.select %tp_even, %tp0_a01i, %tp0_a01s : i32
        %tp0_a23_e = arith.select %tp_even, %tp0_a23i, %tp0_a23s : i32
        %tp0_a01_x = arith.select %tp_even, %tp0_a01s, %tp0_a01i : i32
        %tp0_a23_x = arith.select %tp_even, %tp0_a23s, %tp0_a23i : i32
        %tp0_hi01r = vector.broadcast %tp0_a01_e : i32 to vector<1xi32>
        %tp0_hi01  = vector.bitcast %tp0_hi01r : vector<1xi32> to vector<2xbf16>
        %tp0_hi23r = vector.broadcast %tp0_a23_e : i32 to vector<1xi32>
        %tp0_hi23  = vector.bitcast %tp0_hi23r : vector<1xi32> to vector<2xbf16>
        %tp0_hi45r = vector.broadcast %tp0_a01_x : i32 to vector<1xi32>
        %tp0_hi45  = vector.bitcast %tp0_hi45r : vector<1xi32> to vector<2xbf16>
        %tp0_hi67r = vector.broadcast %tp0_a23_x : i32 to vector<1xi32>
        %tp0_hi67  = vector.bitcast %tp0_hi67r : vector<1xi32> to vector<2xbf16>
        %tp0_hi04 = vector.shuffle %tp0_hi01, %tp0_hi23 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp0_hi48 = vector.shuffle %tp0_hi45, %tp0_hi67 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp0_hivec = vector.shuffle %tp0_hi04, %tp0_hi48 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %tp0_b01_e = arith.select %tp_even, %tp0_b01i, %tp0_b01s : i32
        %tp0_b23_e = arith.select %tp_even, %tp0_b23i, %tp0_b23s : i32
        %tp0_b01_x = arith.select %tp_even, %tp0_b01s, %tp0_b01i : i32
        %tp0_b23_x = arith.select %tp_even, %tp0_b23s, %tp0_b23i : i32
        %tp0_lo01r = vector.broadcast %tp0_b01_e : i32 to vector<1xi32>
        %tp0_lo01  = vector.bitcast %tp0_lo01r : vector<1xi32> to vector<2xbf16>
        %tp0_lo23r = vector.broadcast %tp0_b23_e : i32 to vector<1xi32>
        %tp0_lo23  = vector.bitcast %tp0_lo23r : vector<1xi32> to vector<2xbf16>
        %tp0_lo45r = vector.broadcast %tp0_b01_x : i32 to vector<1xi32>
        %tp0_lo45  = vector.bitcast %tp0_lo45r : vector<1xi32> to vector<2xbf16>
        %tp0_lo67r = vector.broadcast %tp0_b23_x : i32 to vector<1xi32>
        %tp0_lo67  = vector.bitcast %tp0_lo67r : vector<1xi32> to vector<2xbf16>
        %tp0_lo04 = vector.shuffle %tp0_lo01, %tp0_lo23 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp0_lo48 = vector.shuffle %tp0_lo45, %tp0_lo67 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp0_lovec = vector.shuffle %tp0_lo04, %tp0_lo48 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %tp0_fvec = arith.select %tp_even, %tp0_hivec, %tp0_lovec : vector<8xbf16>
        %tp0_moff = arith.select %tp_even, %tp_mt0, %tp_mt1 : index
        %tp0_addr = arith.addi %tp_bn0, %tp0_moff overflow<nsw> : index
        vector.store %tp0_fvec, %tp_buf[%tp0_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // === Pair 1 (rb0,col2-col3): %133, %135 ===
        %v133_b16 = arith.truncf %133 : vector<4xf32> to vector<4xbf16>
        %v135_b16 = arith.truncf %135 : vector<4xf32> to vector<4xbf16>
        %tp1_a01v = vector.extract_strided_slice %v133_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp1_a23v = vector.extract_strided_slice %v133_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp1_a01r = vector.bitcast %tp1_a01v : vector<2xbf16> to vector<1xi32>
        %tp1_a23r = vector.bitcast %tp1_a23v : vector<2xbf16> to vector<1xi32>
        %tp1_a01i = vector.extract %tp1_a01r[0] : i32 from vector<1xi32>
        %tp1_a23i = vector.extract %tp1_a23r[0] : i32 from vector<1xi32>
        %tp1_a01s = amdgpu.permlane_swap %tp1_a01i 16 : i32
        %tp1_a23s = amdgpu.permlane_swap %tp1_a23i 16 : i32
        %tp1_b01v = vector.extract_strided_slice %v135_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp1_b23v = vector.extract_strided_slice %v135_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp1_b01r = vector.bitcast %tp1_b01v : vector<2xbf16> to vector<1xi32>
        %tp1_b23r = vector.bitcast %tp1_b23v : vector<2xbf16> to vector<1xi32>
        %tp1_b01i = vector.extract %tp1_b01r[0] : i32 from vector<1xi32>
        %tp1_b23i = vector.extract %tp1_b23r[0] : i32 from vector<1xi32>
        %tp1_b01s = amdgpu.permlane_swap %tp1_b01i 16 : i32
        %tp1_b23s = amdgpu.permlane_swap %tp1_b23i 16 : i32
        %tp1_a01_e = arith.select %tp_even, %tp1_a01i, %tp1_a01s : i32
        %tp1_a23_e = arith.select %tp_even, %tp1_a23i, %tp1_a23s : i32
        %tp1_a01_x = arith.select %tp_even, %tp1_a01s, %tp1_a01i : i32
        %tp1_a23_x = arith.select %tp_even, %tp1_a23s, %tp1_a23i : i32
        %tp1_hi01r = vector.broadcast %tp1_a01_e : i32 to vector<1xi32>
        %tp1_hi01  = vector.bitcast %tp1_hi01r : vector<1xi32> to vector<2xbf16>
        %tp1_hi23r = vector.broadcast %tp1_a23_e : i32 to vector<1xi32>
        %tp1_hi23  = vector.bitcast %tp1_hi23r : vector<1xi32> to vector<2xbf16>
        %tp1_hi45r = vector.broadcast %tp1_a01_x : i32 to vector<1xi32>
        %tp1_hi45  = vector.bitcast %tp1_hi45r : vector<1xi32> to vector<2xbf16>
        %tp1_hi67r = vector.broadcast %tp1_a23_x : i32 to vector<1xi32>
        %tp1_hi67  = vector.bitcast %tp1_hi67r : vector<1xi32> to vector<2xbf16>
        %tp1_hi04 = vector.shuffle %tp1_hi01, %tp1_hi23 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp1_hi48 = vector.shuffle %tp1_hi45, %tp1_hi67 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp1_hivec = vector.shuffle %tp1_hi04, %tp1_hi48 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %tp1_b01_e = arith.select %tp_even, %tp1_b01i, %tp1_b01s : i32
        %tp1_b23_e = arith.select %tp_even, %tp1_b23i, %tp1_b23s : i32
        %tp1_b01_x = arith.select %tp_even, %tp1_b01s, %tp1_b01i : i32
        %tp1_b23_x = arith.select %tp_even, %tp1_b23s, %tp1_b23i : i32
        %tp1_lo01r = vector.broadcast %tp1_b01_e : i32 to vector<1xi32>
        %tp1_lo01  = vector.bitcast %tp1_lo01r : vector<1xi32> to vector<2xbf16>
        %tp1_lo23r = vector.broadcast %tp1_b23_e : i32 to vector<1xi32>
        %tp1_lo23  = vector.bitcast %tp1_lo23r : vector<1xi32> to vector<2xbf16>
        %tp1_lo45r = vector.broadcast %tp1_b01_x : i32 to vector<1xi32>
        %tp1_lo45  = vector.bitcast %tp1_lo45r : vector<1xi32> to vector<2xbf16>
        %tp1_lo67r = vector.broadcast %tp1_b23_x : i32 to vector<1xi32>
        %tp1_lo67  = vector.bitcast %tp1_lo67r : vector<1xi32> to vector<2xbf16>
        %tp1_lo04 = vector.shuffle %tp1_lo01, %tp1_lo23 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp1_lo48 = vector.shuffle %tp1_lo45, %tp1_lo67 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp1_lovec = vector.shuffle %tp1_lo04, %tp1_lo48 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %tp1_fvec = arith.select %tp_even, %tp1_hivec, %tp1_lovec : vector<8xbf16>
        %tp1_moff = arith.select %tp_even, %tp_mt2, %tp_mt3 : index
        %tp1_addr = arith.addi %tp_bn0, %tp1_moff overflow<nsw> : index
        vector.store %tp1_fvec, %tp_buf[%tp1_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // === Pair 2 (rb1,col0-col1): %137, %139 ===
        %v137_b16 = arith.truncf %137 : vector<4xf32> to vector<4xbf16>
        %v139_b16 = arith.truncf %139 : vector<4xf32> to vector<4xbf16>
        %tp2_a01v = vector.extract_strided_slice %v137_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp2_a23v = vector.extract_strided_slice %v137_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp2_a01r = vector.bitcast %tp2_a01v : vector<2xbf16> to vector<1xi32>
        %tp2_a23r = vector.bitcast %tp2_a23v : vector<2xbf16> to vector<1xi32>
        %tp2_a01i = vector.extract %tp2_a01r[0] : i32 from vector<1xi32>
        %tp2_a23i = vector.extract %tp2_a23r[0] : i32 from vector<1xi32>
        %tp2_a01s = amdgpu.permlane_swap %tp2_a01i 16 : i32
        %tp2_a23s = amdgpu.permlane_swap %tp2_a23i 16 : i32
        %tp2_b01v = vector.extract_strided_slice %v139_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp2_b23v = vector.extract_strided_slice %v139_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp2_b01r = vector.bitcast %tp2_b01v : vector<2xbf16> to vector<1xi32>
        %tp2_b23r = vector.bitcast %tp2_b23v : vector<2xbf16> to vector<1xi32>
        %tp2_b01i = vector.extract %tp2_b01r[0] : i32 from vector<1xi32>
        %tp2_b23i = vector.extract %tp2_b23r[0] : i32 from vector<1xi32>
        %tp2_b01s = amdgpu.permlane_swap %tp2_b01i 16 : i32
        %tp2_b23s = amdgpu.permlane_swap %tp2_b23i 16 : i32
        %tp2_a01_e = arith.select %tp_even, %tp2_a01i, %tp2_a01s : i32
        %tp2_a23_e = arith.select %tp_even, %tp2_a23i, %tp2_a23s : i32
        %tp2_a01_x = arith.select %tp_even, %tp2_a01s, %tp2_a01i : i32
        %tp2_a23_x = arith.select %tp_even, %tp2_a23s, %tp2_a23i : i32
        %tp2_hi01r = vector.broadcast %tp2_a01_e : i32 to vector<1xi32>
        %tp2_hi01  = vector.bitcast %tp2_hi01r : vector<1xi32> to vector<2xbf16>
        %tp2_hi23r = vector.broadcast %tp2_a23_e : i32 to vector<1xi32>
        %tp2_hi23  = vector.bitcast %tp2_hi23r : vector<1xi32> to vector<2xbf16>
        %tp2_hi45r = vector.broadcast %tp2_a01_x : i32 to vector<1xi32>
        %tp2_hi45  = vector.bitcast %tp2_hi45r : vector<1xi32> to vector<2xbf16>
        %tp2_hi67r = vector.broadcast %tp2_a23_x : i32 to vector<1xi32>
        %tp2_hi67  = vector.bitcast %tp2_hi67r : vector<1xi32> to vector<2xbf16>
        %tp2_hi04 = vector.shuffle %tp2_hi01, %tp2_hi23 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp2_hi48 = vector.shuffle %tp2_hi45, %tp2_hi67 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp2_hivec = vector.shuffle %tp2_hi04, %tp2_hi48 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %tp2_b01_e = arith.select %tp_even, %tp2_b01i, %tp2_b01s : i32
        %tp2_b23_e = arith.select %tp_even, %tp2_b23i, %tp2_b23s : i32
        %tp2_b01_x = arith.select %tp_even, %tp2_b01s, %tp2_b01i : i32
        %tp2_b23_x = arith.select %tp_even, %tp2_b23s, %tp2_b23i : i32
        %tp2_lo01r = vector.broadcast %tp2_b01_e : i32 to vector<1xi32>
        %tp2_lo01  = vector.bitcast %tp2_lo01r : vector<1xi32> to vector<2xbf16>
        %tp2_lo23r = vector.broadcast %tp2_b23_e : i32 to vector<1xi32>
        %tp2_lo23  = vector.bitcast %tp2_lo23r : vector<1xi32> to vector<2xbf16>
        %tp2_lo45r = vector.broadcast %tp2_b01_x : i32 to vector<1xi32>
        %tp2_lo45  = vector.bitcast %tp2_lo45r : vector<1xi32> to vector<2xbf16>
        %tp2_lo67r = vector.broadcast %tp2_b23_x : i32 to vector<1xi32>
        %tp2_lo67  = vector.bitcast %tp2_lo67r : vector<1xi32> to vector<2xbf16>
        %tp2_lo04 = vector.shuffle %tp2_lo01, %tp2_lo23 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp2_lo48 = vector.shuffle %tp2_lo45, %tp2_lo67 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp2_lovec = vector.shuffle %tp2_lo04, %tp2_lo48 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %tp2_fvec = arith.select %tp_even, %tp2_hivec, %tp2_lovec : vector<8xbf16>
        %tp2_moff = arith.select %tp_even, %tp_mt0, %tp_mt1 : index
        %tp2_addr = arith.addi %tp_bn1, %tp2_moff overflow<nsw> : index
        vector.store %tp2_fvec, %tp_buf[%tp2_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // === Pair 3 (rb1,col2-col3): %141, %143 ===
        %v141_b16 = arith.truncf %141 : vector<4xf32> to vector<4xbf16>
        %v143_b16 = arith.truncf %143 : vector<4xf32> to vector<4xbf16>
        %tp3_a01v = vector.extract_strided_slice %v141_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp3_a23v = vector.extract_strided_slice %v141_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp3_a01r = vector.bitcast %tp3_a01v : vector<2xbf16> to vector<1xi32>
        %tp3_a23r = vector.bitcast %tp3_a23v : vector<2xbf16> to vector<1xi32>
        %tp3_a01i = vector.extract %tp3_a01r[0] : i32 from vector<1xi32>
        %tp3_a23i = vector.extract %tp3_a23r[0] : i32 from vector<1xi32>
        %tp3_a01s = amdgpu.permlane_swap %tp3_a01i 16 : i32
        %tp3_a23s = amdgpu.permlane_swap %tp3_a23i 16 : i32
        %tp3_b01v = vector.extract_strided_slice %v143_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp3_b23v = vector.extract_strided_slice %v143_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp3_b01r = vector.bitcast %tp3_b01v : vector<2xbf16> to vector<1xi32>
        %tp3_b23r = vector.bitcast %tp3_b23v : vector<2xbf16> to vector<1xi32>
        %tp3_b01i = vector.extract %tp3_b01r[0] : i32 from vector<1xi32>
        %tp3_b23i = vector.extract %tp3_b23r[0] : i32 from vector<1xi32>
        %tp3_b01s = amdgpu.permlane_swap %tp3_b01i 16 : i32
        %tp3_b23s = amdgpu.permlane_swap %tp3_b23i 16 : i32
        %tp3_a01_e = arith.select %tp_even, %tp3_a01i, %tp3_a01s : i32
        %tp3_a23_e = arith.select %tp_even, %tp3_a23i, %tp3_a23s : i32
        %tp3_a01_x = arith.select %tp_even, %tp3_a01s, %tp3_a01i : i32
        %tp3_a23_x = arith.select %tp_even, %tp3_a23s, %tp3_a23i : i32
        %tp3_hi01r = vector.broadcast %tp3_a01_e : i32 to vector<1xi32>
        %tp3_hi01  = vector.bitcast %tp3_hi01r : vector<1xi32> to vector<2xbf16>
        %tp3_hi23r = vector.broadcast %tp3_a23_e : i32 to vector<1xi32>
        %tp3_hi23  = vector.bitcast %tp3_hi23r : vector<1xi32> to vector<2xbf16>
        %tp3_hi45r = vector.broadcast %tp3_a01_x : i32 to vector<1xi32>
        %tp3_hi45  = vector.bitcast %tp3_hi45r : vector<1xi32> to vector<2xbf16>
        %tp3_hi67r = vector.broadcast %tp3_a23_x : i32 to vector<1xi32>
        %tp3_hi67  = vector.bitcast %tp3_hi67r : vector<1xi32> to vector<2xbf16>
        %tp3_hi04 = vector.shuffle %tp3_hi01, %tp3_hi23 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp3_hi48 = vector.shuffle %tp3_hi45, %tp3_hi67 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp3_hivec = vector.shuffle %tp3_hi04, %tp3_hi48 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %tp3_b01_e = arith.select %tp_even, %tp3_b01i, %tp3_b01s : i32
        %tp3_b23_e = arith.select %tp_even, %tp3_b23i, %tp3_b23s : i32
        %tp3_b01_x = arith.select %tp_even, %tp3_b01s, %tp3_b01i : i32
        %tp3_b23_x = arith.select %tp_even, %tp3_b23s, %tp3_b23i : i32
        %tp3_lo01r = vector.broadcast %tp3_b01_e : i32 to vector<1xi32>
        %tp3_lo01  = vector.bitcast %tp3_lo01r : vector<1xi32> to vector<2xbf16>
        %tp3_lo23r = vector.broadcast %tp3_b23_e : i32 to vector<1xi32>
        %tp3_lo23  = vector.bitcast %tp3_lo23r : vector<1xi32> to vector<2xbf16>
        %tp3_lo45r = vector.broadcast %tp3_b01_x : i32 to vector<1xi32>
        %tp3_lo45  = vector.bitcast %tp3_lo45r : vector<1xi32> to vector<2xbf16>
        %tp3_lo67r = vector.broadcast %tp3_b23_x : i32 to vector<1xi32>
        %tp3_lo67  = vector.bitcast %tp3_lo67r : vector<1xi32> to vector<2xbf16>
        %tp3_lo04 = vector.shuffle %tp3_lo01, %tp3_lo23 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp3_lo48 = vector.shuffle %tp3_lo45, %tp3_lo67 [0, 1, 2, 3] : vector<2xbf16>, vector<2xbf16>
        %tp3_lovec = vector.shuffle %tp3_lo04, %tp3_lo48 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
        %tp3_fvec = arith.select %tp_even, %tp3_hivec, %tp3_lovec : vector<8xbf16>
        %tp3_moff = arith.select %tp_even, %tp_mt2, %tp_mt3 : index
        %tp3_addr = arith.addi %tp_bn1, %tp3_moff overflow<nsw> : index
        vector.store %tp3_fvec, %tp_buf[%tp3_addr] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

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

