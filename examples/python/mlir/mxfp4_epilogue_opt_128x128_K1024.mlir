#map = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 16) * 128)>
#map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
#map2 = affine_map<()[s0] -> (s0 mod 8)>
#map3 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8) floordiv 128) * 65536)>
#map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 16) * 128 + 64)>
#map5 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 128) * 65536 + 32768)>
#map6 = affine_map<()[s0, s1] -> (s0 * 4096 + s1 * 4 + (s1 floordiv 64) * 1024 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32)>
#map7 = affine_map<()[s0, s1, s2] -> (s0 * 4096 + s1 * 2048 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32)>
#map8 = affine_map<()[s0, s1, s2] -> (s0 * 4096 + s1 * 2048 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 1024)>
#map9 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
#map10 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8) floordiv 128) * 65536 + 128)>
#map11 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 128) * 65536 + 32896)>
#map12 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
#map13 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 4096 - (s0 floordiv 16) * 2048)>
#map14 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 4096 - (s0 floordiv 16) * 2048 + 2048)>
#map15 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048)>
#map16 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 2048)>
#map17 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 4096)>
#map18 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 6144)>
#map19 = affine_map<()[s0, s1] -> (s0 * 4096 + s1 * 4 + (s1 floordiv 64) * 1024 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32 + 256)>
#map20 = affine_map<()[s0, s1, s2] -> (s0 * 4096 + s1 * 2048 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 256)>
#map21 = affine_map<()[s0, s1, s2] -> (s0 * 4096 + s1 * 2048 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 1280)>
#map22 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
#map23 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8) floordiv 128) * 65536 + 256)>
#map24 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 128) * 65536 + 33024)>
#map25 = affine_map<()[s0, s1] -> (s0 * 4096 + s1 * 4 + (s1 floordiv 64) * 1024 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32 + 512)>
#map26 = affine_map<()[s0, s1, s2] -> (s0 * 4096 + s1 * 2048 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 512)>
#map27 = affine_map<()[s0, s1, s2] -> (s0 * 4096 + s1 * 2048 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 1536)>
#map28 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8) floordiv 128) * 65536 + 384)>
#map29 = affine_map<()[s0, s1, s2, s3] -> (s0 * 65536 + s1 * 16384 + s3 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 128) * 65536 + 33152)>
#map30 = affine_map<()[s0, s1] -> (s0 * 4096 + s1 * 4 + (s1 floordiv 64) * 1024 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32 + 768)>
#map31 = affine_map<()[s0, s1, s2] -> (s0 * 4096 + s1 * 2048 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 768)>
#map32 = affine_map<()[s0, s1, s2] -> (s0 * 4096 + s1 * 2048 + s2 * 4 + ((s2 mod 64) floordiv 16) * 64 + ((s2 mod 16) floordiv 8) * 32 - (s2 floordiv 8) * 32 + 1792)>
#map33 = affine_map<()[s0] -> (s0 * 128)>
#map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4)>
#map35 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16)>
#map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map39 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 16)>
#map40 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 32)>
#map41 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 48)>
#map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map45 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>
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
        %c32_i14 = arith.constant 32 : i14
        %c512_i14 = arith.constant 512 : i14
        %c2147483645_i64 = arith.constant 2147483645 : i64
        %c65536_i64 = arith.constant 65536 : i64
        %c1048576_i64 = arith.constant 1048576 : i64
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
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
        %12 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c1048576_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %12[%11], %alloc_2[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %13 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %14 = gpu.subgroup_broadcast %13, first_active_lane : index
        %15 = affine.apply #map5()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%15], %alloc_2[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_4 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_5 = memref.cast %reinterpret_cast_4 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %16 = affine.apply #map3()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        %17 = amdgpu.fat_raw_buffer_cast %cast_5 validBytes(%c1048576_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %17[%16], %alloc_0[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %18 = affine.apply #map5()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %17[%18], %alloc_0[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_6 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_7 = memref.cast %reinterpret_cast_6 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %19 = amdgpu.fat_raw_buffer_cast %cast_7 validBytes(%c65536_i64) cacheSwizzleStride(%c32_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %20 = affine.apply #map6()[%block_id_x, %thread_id_x]
        %21 = vector.load %19[%20] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %22 = vector.bitcast %21 : vector<4xi8> to vector<4xf8E8M0FNU>
        %reinterpret_cast_8 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_9 = memref.cast %reinterpret_cast_8 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %23 = amdgpu.fat_raw_buffer_cast %cast_9 validBytes(%c65536_i64) cacheSwizzleStride(%c32_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %24 = affine.apply #map7()[%block_id_y, %thread_id_y, %thread_id_x]
        %25 = vector.load %23[%24] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %26 = vector.bitcast %25 : vector<4xi8> to vector<4xf8E8M0FNU>
        %27 = affine.apply #map8()[%block_id_y, %thread_id_y, %thread_id_x]
        %28 = vector.load %23[%27] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %29 = vector.bitcast %28 : vector<4xi8> to vector<4xf8E8M0FNU>
        amdgpu.memory_counter_wait load(3)
        rocdl.s.barrier
        %30 = affine.apply #map9()[%thread_id_x, %thread_id_y]
        %31 = arith.index_cast %30 : index to i32
        %32 = arith.cmpi sge, %31, %c4_i32 : i32
        %33 = arith.cmpi slt, %31, %c4_i32 : i32
        scf.if %32 {
          rocdl.s.barrier
        }
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(0)
        rocdl.s.barrier
        %34 = affine.apply #map10()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%34], %alloc_1[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %35 = affine.apply #map11()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%35], %alloc_1[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %36 = affine.apply #map10()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %17[%36], %alloc[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %37 = affine.apply #map11()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %17[%37], %alloc[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        rocdl.sched.barrier 0
        %reinterpret_cast_10 = memref.reinterpret_cast %alloc_2 to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
        %38 = affine.apply #map12()[%thread_id_x]
        %39 = arith.xori %38, %9 : index
        %40 = affine.apply #map13()[%thread_id_x, %39]
        %41 = vector.load %reinterpret_cast_10[%40] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %42 = affine.apply #map14()[%thread_id_x, %39]
        %43 = vector.load %reinterpret_cast_10[%42] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %reinterpret_cast_11 = memref.reinterpret_cast %alloc_0 to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
        %44 = affine.apply #map15()[%thread_id_y, %thread_id_x, %39]
        %45 = vector.load %reinterpret_cast_11[%44] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %46 = affine.apply #map16()[%thread_id_y, %thread_id_x, %39]
        %47 = vector.load %reinterpret_cast_11[%46] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %48 = affine.apply #map17()[%thread_id_y, %thread_id_x, %39]
        %49 = vector.load %reinterpret_cast_11[%48] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %50 = affine.apply #map18()[%thread_id_y, %thread_id_x, %39]
        %51 = vector.load %reinterpret_cast_11[%50] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %52 = vector.bitcast %41 : vector<16xi8> to vector<32xf4E2M1FN>
        %53 = vector.bitcast %43 : vector<16xi8> to vector<32xf4E2M1FN>
        %54 = vector.bitcast %45 : vector<16xi8> to vector<32xf4E2M1FN>
        %55 = vector.bitcast %47 : vector<16xi8> to vector<32xf4E2M1FN>
        %56 = vector.bitcast %49 : vector<16xi8> to vector<32xf4E2M1FN>
        %57 = vector.bitcast %51 : vector<16xi8> to vector<32xf4E2M1FN>
        %58 = affine.apply #map19()[%block_id_x, %thread_id_x]
        %59 = vector.load %19[%58] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %60 = vector.bitcast %59 : vector<4xi8> to vector<4xf8E8M0FNU>
        %61 = affine.apply #map20()[%block_id_y, %thread_id_y, %thread_id_x]
        %62 = vector.load %23[%61] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %63 = vector.bitcast %62 : vector<4xi8> to vector<4xf8E8M0FNU>
        %64 = affine.apply #map21()[%block_id_y, %thread_id_y, %thread_id_x]
        %65 = vector.load %23[%64] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %66 = vector.bitcast %65 : vector<4xi8> to vector<4xf8E8M0FNU>
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %67 = amdgpu.scaled_mfma 16x16x128 (%22[0] * %52) * (%26[0] * %54) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %68 = amdgpu.scaled_mfma 16x16x128 (%22[0] * %52) * (%26[1] * %55) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %69 = amdgpu.scaled_mfma 16x16x128 (%22[0] * %52) * (%29[0] * %56) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %70 = amdgpu.scaled_mfma 16x16x128 (%22[0] * %52) * (%29[1] * %57) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %71 = amdgpu.scaled_mfma 16x16x128 (%22[1] * %53) * (%26[0] * %54) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %72 = amdgpu.scaled_mfma 16x16x128 (%22[1] * %53) * (%26[1] * %55) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %73 = amdgpu.scaled_mfma 16x16x128 (%22[1] * %53) * (%29[0] * %56) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %74 = amdgpu.scaled_mfma 16x16x128 (%22[1] * %53) * (%29[1] * %57) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        %75 = affine.apply #map22()[%thread_id_x]
        %76 = arith.xori %75, %9 : index
        %77 = affine.apply #map13()[%thread_id_x, %76]
        %78 = vector.load %reinterpret_cast_10[%77] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %79 = affine.apply #map14()[%thread_id_x, %76]
        %80 = vector.load %reinterpret_cast_10[%79] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %81 = affine.apply #map15()[%thread_id_y, %thread_id_x, %76]
        %82 = vector.load %reinterpret_cast_11[%81] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %83 = affine.apply #map16()[%thread_id_y, %thread_id_x, %76]
        %84 = vector.load %reinterpret_cast_11[%83] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %85 = affine.apply #map17()[%thread_id_y, %thread_id_x, %76]
        %86 = vector.load %reinterpret_cast_11[%85] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %87 = affine.apply #map18()[%thread_id_y, %thread_id_x, %76]
        %88 = vector.load %reinterpret_cast_11[%87] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %89 = vector.bitcast %78 : vector<16xi8> to vector<32xf4E2M1FN>
        %90 = vector.bitcast %80 : vector<16xi8> to vector<32xf4E2M1FN>
        %91 = vector.bitcast %82 : vector<16xi8> to vector<32xf4E2M1FN>
        %92 = vector.bitcast %84 : vector<16xi8> to vector<32xf4E2M1FN>
        %93 = vector.bitcast %86 : vector<16xi8> to vector<32xf4E2M1FN>
        %94 = vector.bitcast %88 : vector<16xi8> to vector<32xf4E2M1FN>
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(3)
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %95 = amdgpu.scaled_mfma 16x16x128 (%22[2] * %89) * (%26[2] * %91) + %67 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %96 = amdgpu.scaled_mfma 16x16x128 (%22[2] * %89) * (%26[3] * %92) + %68 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %97 = amdgpu.scaled_mfma 16x16x128 (%22[2] * %89) * (%29[2] * %93) + %69 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %98 = amdgpu.scaled_mfma 16x16x128 (%22[2] * %89) * (%29[3] * %94) + %70 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %99 = amdgpu.scaled_mfma 16x16x128 (%22[3] * %90) * (%26[2] * %91) + %71 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %100 = amdgpu.scaled_mfma 16x16x128 (%22[3] * %90) * (%26[3] * %92) + %72 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %101 = amdgpu.scaled_mfma 16x16x128 (%22[3] * %90) * (%29[2] * %93) + %73 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %102 = amdgpu.scaled_mfma 16x16x128 (%22[3] * %90) * (%29[3] * %94) + %74 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(0)
        rocdl.s.barrier
        %103 = affine.apply #map23()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%103], %alloc_2[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %104 = affine.apply #map24()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%104], %alloc_2[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %105 = affine.apply #map23()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %17[%105], %alloc_0[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %106 = affine.apply #map24()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %17[%106], %alloc_0[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        rocdl.sched.barrier 0
        %reinterpret_cast_12 = memref.reinterpret_cast %alloc_1 to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
        %107 = vector.load %reinterpret_cast_12[%40] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %108 = vector.load %reinterpret_cast_12[%42] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %reinterpret_cast_13 = memref.reinterpret_cast %alloc to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
        %109 = vector.load %reinterpret_cast_13[%44] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %110 = vector.load %reinterpret_cast_13[%46] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %111 = vector.load %reinterpret_cast_13[%48] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %112 = vector.load %reinterpret_cast_13[%50] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %113 = vector.bitcast %107 : vector<16xi8> to vector<32xf4E2M1FN>
        %114 = vector.bitcast %108 : vector<16xi8> to vector<32xf4E2M1FN>
        %115 = vector.bitcast %109 : vector<16xi8> to vector<32xf4E2M1FN>
        %116 = vector.bitcast %110 : vector<16xi8> to vector<32xf4E2M1FN>
        %117 = vector.bitcast %111 : vector<16xi8> to vector<32xf4E2M1FN>
        %118 = vector.bitcast %112 : vector<16xi8> to vector<32xf4E2M1FN>
        %119 = affine.apply #map25()[%block_id_x, %thread_id_x]
        %120 = vector.load %19[%119] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %121 = vector.bitcast %120 : vector<4xi8> to vector<4xf8E8M0FNU>
        %122 = affine.apply #map26()[%block_id_y, %thread_id_y, %thread_id_x]
        %123 = vector.load %23[%122] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %124 = vector.bitcast %123 : vector<4xi8> to vector<4xf8E8M0FNU>
        %125 = affine.apply #map27()[%block_id_y, %thread_id_y, %thread_id_x]
        %126 = vector.load %23[%125] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %127 = vector.bitcast %126 : vector<4xi8> to vector<4xf8E8M0FNU>
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %128 = amdgpu.scaled_mfma 16x16x128 (%60[0] * %113) * (%63[0] * %115) + %95 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %129 = amdgpu.scaled_mfma 16x16x128 (%60[0] * %113) * (%63[1] * %116) + %96 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %130 = amdgpu.scaled_mfma 16x16x128 (%60[0] * %113) * (%66[0] * %117) + %97 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %131 = amdgpu.scaled_mfma 16x16x128 (%60[0] * %113) * (%66[1] * %118) + %98 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %132 = amdgpu.scaled_mfma 16x16x128 (%60[1] * %114) * (%63[0] * %115) + %99 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %133 = amdgpu.scaled_mfma 16x16x128 (%60[1] * %114) * (%63[1] * %116) + %100 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %134 = amdgpu.scaled_mfma 16x16x128 (%60[1] * %114) * (%66[0] * %117) + %101 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %135 = amdgpu.scaled_mfma 16x16x128 (%60[1] * %114) * (%66[1] * %118) + %102 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        %136 = vector.load %reinterpret_cast_12[%77] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %137 = vector.load %reinterpret_cast_12[%79] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %138 = vector.load %reinterpret_cast_13[%81] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %139 = vector.load %reinterpret_cast_13[%83] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %140 = vector.load %reinterpret_cast_13[%85] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %141 = vector.load %reinterpret_cast_13[%87] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %142 = vector.bitcast %136 : vector<16xi8> to vector<32xf4E2M1FN>
        %143 = vector.bitcast %137 : vector<16xi8> to vector<32xf4E2M1FN>
        %144 = vector.bitcast %138 : vector<16xi8> to vector<32xf4E2M1FN>
        %145 = vector.bitcast %139 : vector<16xi8> to vector<32xf4E2M1FN>
        %146 = vector.bitcast %140 : vector<16xi8> to vector<32xf4E2M1FN>
        %147 = vector.bitcast %141 : vector<16xi8> to vector<32xf4E2M1FN>
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(3)
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %148 = amdgpu.scaled_mfma 16x16x128 (%60[2] * %142) * (%63[2] * %144) + %128 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %149 = amdgpu.scaled_mfma 16x16x128 (%60[2] * %142) * (%63[3] * %145) + %129 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %150 = amdgpu.scaled_mfma 16x16x128 (%60[2] * %142) * (%66[2] * %146) + %130 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %151 = amdgpu.scaled_mfma 16x16x128 (%60[2] * %142) * (%66[3] * %147) + %131 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %152 = amdgpu.scaled_mfma 16x16x128 (%60[3] * %143) * (%63[2] * %144) + %132 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %153 = amdgpu.scaled_mfma 16x16x128 (%60[3] * %143) * (%63[3] * %145) + %133 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %154 = amdgpu.scaled_mfma 16x16x128 (%60[3] * %143) * (%66[2] * %146) + %134 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %155 = amdgpu.scaled_mfma 16x16x128 (%60[3] * %143) * (%66[3] * %147) + %135 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(0)
        rocdl.s.barrier
        %156 = affine.apply #map28()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%156], %alloc_1[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %157 = affine.apply #map29()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%157], %alloc_1[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %158 = affine.apply #map28()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %17[%158], %alloc[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %159 = affine.apply #map29()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %17[%159], %alloc[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        rocdl.sched.barrier 0
        %160 = vector.load %reinterpret_cast_10[%40] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %161 = vector.load %reinterpret_cast_10[%42] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %162 = vector.load %reinterpret_cast_11[%44] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %163 = vector.load %reinterpret_cast_11[%46] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %164 = vector.load %reinterpret_cast_11[%48] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %165 = vector.load %reinterpret_cast_11[%50] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %166 = vector.bitcast %160 : vector<16xi8> to vector<32xf4E2M1FN>
        %167 = vector.bitcast %161 : vector<16xi8> to vector<32xf4E2M1FN>
        %168 = vector.bitcast %162 : vector<16xi8> to vector<32xf4E2M1FN>
        %169 = vector.bitcast %163 : vector<16xi8> to vector<32xf4E2M1FN>
        %170 = vector.bitcast %164 : vector<16xi8> to vector<32xf4E2M1FN>
        %171 = vector.bitcast %165 : vector<16xi8> to vector<32xf4E2M1FN>
        %172 = affine.apply #map30()[%block_id_x, %thread_id_x]
        %173 = vector.load %19[%172] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %174 = vector.bitcast %173 : vector<4xi8> to vector<4xf8E8M0FNU>
        %175 = affine.apply #map31()[%block_id_y, %thread_id_y, %thread_id_x]
        %176 = vector.load %23[%175] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %177 = vector.bitcast %176 : vector<4xi8> to vector<4xf8E8M0FNU>
        %178 = affine.apply #map32()[%block_id_y, %thread_id_y, %thread_id_x]
        %179 = vector.load %23[%178] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %180 = vector.bitcast %179 : vector<4xi8> to vector<4xf8E8M0FNU>
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %181 = amdgpu.scaled_mfma 16x16x128 (%121[0] * %166) * (%124[0] * %168) + %148 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %182 = amdgpu.scaled_mfma 16x16x128 (%121[0] * %166) * (%124[1] * %169) + %149 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %183 = amdgpu.scaled_mfma 16x16x128 (%121[0] * %166) * (%127[0] * %170) + %150 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %184 = amdgpu.scaled_mfma 16x16x128 (%121[0] * %166) * (%127[1] * %171) + %151 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %185 = amdgpu.scaled_mfma 16x16x128 (%121[1] * %167) * (%124[0] * %168) + %152 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %186 = amdgpu.scaled_mfma 16x16x128 (%121[1] * %167) * (%124[1] * %169) + %153 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %187 = amdgpu.scaled_mfma 16x16x128 (%121[1] * %167) * (%127[0] * %170) + %154 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %188 = amdgpu.scaled_mfma 16x16x128 (%121[1] * %167) * (%127[1] * %171) + %155 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.sched.barrier 0
        %189 = vector.load %reinterpret_cast_10[%77] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %190 = vector.load %reinterpret_cast_10[%79] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %191 = vector.load %reinterpret_cast_11[%81] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %192 = vector.load %reinterpret_cast_11[%83] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %193 = vector.load %reinterpret_cast_11[%85] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %194 = vector.load %reinterpret_cast_11[%87] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %195 = vector.bitcast %189 : vector<16xi8> to vector<32xf4E2M1FN>
        %196 = vector.bitcast %190 : vector<16xi8> to vector<32xf4E2M1FN>
        %197 = vector.bitcast %191 : vector<16xi8> to vector<32xf4E2M1FN>
        %198 = vector.bitcast %192 : vector<16xi8> to vector<32xf4E2M1FN>
        %199 = vector.bitcast %193 : vector<16xi8> to vector<32xf4E2M1FN>
        %200 = vector.bitcast %194 : vector<16xi8> to vector<32xf4E2M1FN>
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(3)
        rocdl.s.barrier
        rocdl.sched.barrier 0
        rocdl.s.setprio 1
        %201 = amdgpu.scaled_mfma 16x16x128 (%121[2] * %195) * (%124[2] * %197) + %181 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %202 = amdgpu.scaled_mfma 16x16x128 (%121[2] * %195) * (%124[3] * %198) + %182 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %203 = amdgpu.scaled_mfma 16x16x128 (%121[2] * %195) * (%127[2] * %199) + %183 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %204 = amdgpu.scaled_mfma 16x16x128 (%121[2] * %195) * (%127[3] * %200) + %184 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %205 = amdgpu.scaled_mfma 16x16x128 (%121[3] * %196) * (%124[2] * %197) + %185 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %206 = amdgpu.scaled_mfma 16x16x128 (%121[3] * %196) * (%124[3] * %198) + %186 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %207 = amdgpu.scaled_mfma 16x16x128 (%121[3] * %196) * (%127[2] * %199) + %187 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %208 = amdgpu.scaled_mfma 16x16x128 (%121[3] * %196) * (%127[3] * %200) + %188 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        rocdl.s.setprio 0
        rocdl.sched.barrier 0
        scf.if %33 {
          rocdl.s.barrier
        }
        amdgpu.lds_barrier
        %209 = vector.load %reinterpret_cast_13[%44] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %210 = vector.load %reinterpret_cast_13[%81] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %211 = vector.load %reinterpret_cast_13[%46] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %212 = vector.load %reinterpret_cast_13[%83] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %213 = vector.load %reinterpret_cast_13[%48] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %214 = vector.load %reinterpret_cast_13[%85] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %215 = vector.load %reinterpret_cast_13[%50] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %216 = vector.load %reinterpret_cast_13[%87] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %217 = vector.load %reinterpret_cast_12[%40] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %218 = vector.load %reinterpret_cast_12[%77] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %219 = vector.load %reinterpret_cast_12[%42] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %220 = vector.load %reinterpret_cast_12[%79] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %221 = vector.bitcast %217 : vector<16xi8> to vector<32xf4E2M1FN>
        %222 = vector.bitcast %218 : vector<16xi8> to vector<32xf4E2M1FN>
        %223 = vector.bitcast %219 : vector<16xi8> to vector<32xf4E2M1FN>
        %224 = vector.bitcast %220 : vector<16xi8> to vector<32xf4E2M1FN>
        %225 = vector.bitcast %209 : vector<16xi8> to vector<32xf4E2M1FN>
        %226 = vector.bitcast %210 : vector<16xi8> to vector<32xf4E2M1FN>
        %227 = vector.bitcast %211 : vector<16xi8> to vector<32xf4E2M1FN>
        %228 = vector.bitcast %212 : vector<16xi8> to vector<32xf4E2M1FN>
        %229 = vector.bitcast %213 : vector<16xi8> to vector<32xf4E2M1FN>
        %230 = vector.bitcast %214 : vector<16xi8> to vector<32xf4E2M1FN>
        %231 = vector.bitcast %215 : vector<16xi8> to vector<32xf4E2M1FN>
        %232 = vector.bitcast %216 : vector<16xi8> to vector<32xf4E2M1FN>
        %233 = amdgpu.scaled_mfma 16x16x128 (%174[0] * %221) * (%177[0] * %225) + %201 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %234 = amdgpu.scaled_mfma 16x16x128 (%174[2] * %222) * (%177[2] * %226) + %233 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %235 = amdgpu.scaled_mfma 16x16x128 (%174[0] * %221) * (%177[1] * %227) + %202 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %236 = amdgpu.scaled_mfma 16x16x128 (%174[2] * %222) * (%177[3] * %228) + %235 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %237 = amdgpu.scaled_mfma 16x16x128 (%174[0] * %221) * (%180[0] * %229) + %203 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %238 = amdgpu.scaled_mfma 16x16x128 (%174[2] * %222) * (%180[2] * %230) + %237 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %239 = amdgpu.scaled_mfma 16x16x128 (%174[0] * %221) * (%180[1] * %231) + %204 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %240 = amdgpu.scaled_mfma 16x16x128 (%174[2] * %222) * (%180[3] * %232) + %239 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %241 = amdgpu.scaled_mfma 16x16x128 (%174[1] * %223) * (%177[0] * %225) + %205 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %242 = amdgpu.scaled_mfma 16x16x128 (%174[3] * %224) * (%177[2] * %226) + %241 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %243 = amdgpu.scaled_mfma 16x16x128 (%174[1] * %223) * (%177[1] * %227) + %206 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %244 = amdgpu.scaled_mfma 16x16x128 (%174[3] * %224) * (%177[3] * %228) + %243 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %245 = amdgpu.scaled_mfma 16x16x128 (%174[1] * %223) * (%180[0] * %229) + %207 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %246 = amdgpu.scaled_mfma 16x16x128 (%174[3] * %224) * (%180[2] * %230) + %245 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %247 = amdgpu.scaled_mfma 16x16x128 (%174[1] * %223) * (%180[1] * %231) + %208 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %248 = amdgpu.scaled_mfma 16x16x128 (%174[3] * %224) * (%180[3] * %232) + %247 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        // ── Optimized epilogue: XOR-1 shuffle → vector<2xbf16> stores ──────
        %ep_blkx = affine.apply #map33()[%block_id_x]
        %ep_blky = affine.apply #map33()[%block_id_y]
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
        %ep_rb0_r0 = affine.apply #map34()[%thread_id_x]
        %ep_rb0_r1 = affine.apply #map36()[%thread_id_x]
        %ep_rb0_r2 = affine.apply #map37()[%thread_id_x]
        %ep_rb0_r3 = affine.apply #map38()[%thread_id_x]
        %ep_rb0_ra = arith.select %ep_is_even, %ep_rb0_r0, %ep_rb0_r2 : index
        %ep_rb0_rb = arith.select %ep_is_even, %ep_rb0_r1, %ep_rb0_r3 : index
        %ep_rb0_oa = arith.muli %ep_rb0_ra, %ep_stride#0 overflow<nsw> : index
        %ep_rb0_ob = arith.muli %ep_rb0_rb, %ep_stride#0 overflow<nsw> : index
        %ep_rb1_r0 = affine.apply #map42()[%thread_id_x]
        %ep_rb1_r1 = affine.apply #map43()[%thread_id_x]
        %ep_rb1_r2 = affine.apply #map44()[%thread_id_x]
        %ep_rb1_r3 = affine.apply #map45()[%thread_id_x]
        %ep_rb1_ra = arith.select %ep_is_even, %ep_rb1_r0, %ep_rb1_r2 : index
        %ep_rb1_rb = arith.select %ep_is_even, %ep_rb1_r1, %ep_rb1_r3 : index
        %ep_rb1_oa = arith.muli %ep_rb1_ra, %ep_stride#0 overflow<nsw> : index
        %ep_rb1_ob = arith.muli %ep_rb1_rb, %ep_stride#0 overflow<nsw> : index
        %ep_c0 = affine.apply #map35()[%thread_id_x, %thread_id_y]
        %ep_c0a = arith.subi %ep_c0, %ep_lparity : index
        %ep_c1 = affine.apply #map39()[%thread_id_x, %thread_id_y]
        %ep_c1a = arith.subi %ep_c1, %ep_lparity : index
        %ep_c2 = affine.apply #map40()[%thread_id_x, %thread_id_y]
        %ep_c2a = arith.subi %ep_c2, %ep_lparity : index
        %ep_c3 = affine.apply #map41()[%thread_id_x, %thread_id_y]
        %ep_c3a = arith.subi %ep_c3, %ep_lparity : index

        // MFMA %234 → rb0, col0
        %v234_e0 = vector.extract %234[0] : f32 from vector<4xf32>
        %v234_e1 = vector.extract %234[1] : f32 from vector<4xf32>
        %v234_e2 = vector.extract %234[2] : f32 from vector<4xf32>
        %v234_e3 = vector.extract %234[3] : f32 from vector<4xf32>
        %v234_e0n, %v234_e0v = gpu.shuffle xor %v234_e0, %ep_xor_off, %ep_xor_w : f32
        %v234_e1n, %v234_e1v = gpu.shuffle xor %v234_e1, %ep_xor_off, %ep_xor_w : f32
        %v234_e2n, %v234_e2v = gpu.shuffle xor %v234_e2, %ep_xor_off, %ep_xor_w : f32
        %v234_e3n, %v234_e3v = gpu.shuffle xor %v234_e3, %ep_xor_off, %ep_xor_w : f32
        %v234_e0lo = arith.select %ep_is_even, %v234_e0,  %v234_e0n : f32
        %v234_e0hi = arith.select %ep_is_even, %v234_e0n, %v234_e0  : f32
        %v234_e1lo = arith.select %ep_is_even, %v234_e1,  %v234_e1n : f32
        %v234_e1hi = arith.select %ep_is_even, %v234_e1n, %v234_e1  : f32
        %v234_e2lo = arith.select %ep_is_even, %v234_e2,  %v234_e2n : f32
        %v234_e2hi = arith.select %ep_is_even, %v234_e2n, %v234_e2  : f32
        %v234_e3lo = arith.select %ep_is_even, %v234_e3,  %v234_e3n : f32
        %v234_e3hi = arith.select %ep_is_even, %v234_e3n, %v234_e3  : f32
        %v234_alo = arith.select %ep_is_even, %v234_e0lo, %v234_e2lo : f32
        %v234_ahi = arith.select %ep_is_even, %v234_e0hi, %v234_e2hi : f32
        %v234_blo = arith.select %ep_is_even, %v234_e1lo, %v234_e3lo : f32
        %v234_bhi = arith.select %ep_is_even, %v234_e1hi, %v234_e3hi : f32
        %v234_va0 = vector.broadcast %v234_alo : f32 to vector<2xf32>
        %v234_va  = vector.insert %v234_ahi, %v234_va0 [1] : f32 into vector<2xf32>
        %v234_vab = arith.truncf %v234_va : vector<2xf32> to vector<2xbf16>
        %v234_vb0 = vector.broadcast %v234_blo : f32 to vector<2xf32>
        %v234_vb  = vector.insert %v234_bhi, %v234_vb0 [1] : f32 into vector<2xf32>
        %v234_vbb = arith.truncf %v234_vb : vector<2xf32> to vector<2xbf16>
        %v234_adra = arith.addi %ep_rb0_oa, %ep_c0a overflow<nsw> : index
        %v234_adrb = arith.addi %ep_rb0_ob, %ep_c0a overflow<nsw> : index
        vector.store %v234_vab, %ep_buf[%v234_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v234_vbb, %ep_buf[%v234_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %236 → rb0, col1
        %v236_e0 = vector.extract %236[0] : f32 from vector<4xf32>
        %v236_e1 = vector.extract %236[1] : f32 from vector<4xf32>
        %v236_e2 = vector.extract %236[2] : f32 from vector<4xf32>
        %v236_e3 = vector.extract %236[3] : f32 from vector<4xf32>
        %v236_e0n, %v236_e0v = gpu.shuffle xor %v236_e0, %ep_xor_off, %ep_xor_w : f32
        %v236_e1n, %v236_e1v = gpu.shuffle xor %v236_e1, %ep_xor_off, %ep_xor_w : f32
        %v236_e2n, %v236_e2v = gpu.shuffle xor %v236_e2, %ep_xor_off, %ep_xor_w : f32
        %v236_e3n, %v236_e3v = gpu.shuffle xor %v236_e3, %ep_xor_off, %ep_xor_w : f32
        %v236_e0lo = arith.select %ep_is_even, %v236_e0,  %v236_e0n : f32
        %v236_e0hi = arith.select %ep_is_even, %v236_e0n, %v236_e0  : f32
        %v236_e1lo = arith.select %ep_is_even, %v236_e1,  %v236_e1n : f32
        %v236_e1hi = arith.select %ep_is_even, %v236_e1n, %v236_e1  : f32
        %v236_e2lo = arith.select %ep_is_even, %v236_e2,  %v236_e2n : f32
        %v236_e2hi = arith.select %ep_is_even, %v236_e2n, %v236_e2  : f32
        %v236_e3lo = arith.select %ep_is_even, %v236_e3,  %v236_e3n : f32
        %v236_e3hi = arith.select %ep_is_even, %v236_e3n, %v236_e3  : f32
        %v236_alo = arith.select %ep_is_even, %v236_e0lo, %v236_e2lo : f32
        %v236_ahi = arith.select %ep_is_even, %v236_e0hi, %v236_e2hi : f32
        %v236_blo = arith.select %ep_is_even, %v236_e1lo, %v236_e3lo : f32
        %v236_bhi = arith.select %ep_is_even, %v236_e1hi, %v236_e3hi : f32
        %v236_va0 = vector.broadcast %v236_alo : f32 to vector<2xf32>
        %v236_va  = vector.insert %v236_ahi, %v236_va0 [1] : f32 into vector<2xf32>
        %v236_vab = arith.truncf %v236_va : vector<2xf32> to vector<2xbf16>
        %v236_vb0 = vector.broadcast %v236_blo : f32 to vector<2xf32>
        %v236_vb  = vector.insert %v236_bhi, %v236_vb0 [1] : f32 into vector<2xf32>
        %v236_vbb = arith.truncf %v236_vb : vector<2xf32> to vector<2xbf16>
        %v236_adra = arith.addi %ep_rb0_oa, %ep_c1a overflow<nsw> : index
        %v236_adrb = arith.addi %ep_rb0_ob, %ep_c1a overflow<nsw> : index
        vector.store %v236_vab, %ep_buf[%v236_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v236_vbb, %ep_buf[%v236_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %238 → rb0, col2
        %v238_e0 = vector.extract %238[0] : f32 from vector<4xf32>
        %v238_e1 = vector.extract %238[1] : f32 from vector<4xf32>
        %v238_e2 = vector.extract %238[2] : f32 from vector<4xf32>
        %v238_e3 = vector.extract %238[3] : f32 from vector<4xf32>
        %v238_e0n, %v238_e0v = gpu.shuffle xor %v238_e0, %ep_xor_off, %ep_xor_w : f32
        %v238_e1n, %v238_e1v = gpu.shuffle xor %v238_e1, %ep_xor_off, %ep_xor_w : f32
        %v238_e2n, %v238_e2v = gpu.shuffle xor %v238_e2, %ep_xor_off, %ep_xor_w : f32
        %v238_e3n, %v238_e3v = gpu.shuffle xor %v238_e3, %ep_xor_off, %ep_xor_w : f32
        %v238_e0lo = arith.select %ep_is_even, %v238_e0,  %v238_e0n : f32
        %v238_e0hi = arith.select %ep_is_even, %v238_e0n, %v238_e0  : f32
        %v238_e1lo = arith.select %ep_is_even, %v238_e1,  %v238_e1n : f32
        %v238_e1hi = arith.select %ep_is_even, %v238_e1n, %v238_e1  : f32
        %v238_e2lo = arith.select %ep_is_even, %v238_e2,  %v238_e2n : f32
        %v238_e2hi = arith.select %ep_is_even, %v238_e2n, %v238_e2  : f32
        %v238_e3lo = arith.select %ep_is_even, %v238_e3,  %v238_e3n : f32
        %v238_e3hi = arith.select %ep_is_even, %v238_e3n, %v238_e3  : f32
        %v238_alo = arith.select %ep_is_even, %v238_e0lo, %v238_e2lo : f32
        %v238_ahi = arith.select %ep_is_even, %v238_e0hi, %v238_e2hi : f32
        %v238_blo = arith.select %ep_is_even, %v238_e1lo, %v238_e3lo : f32
        %v238_bhi = arith.select %ep_is_even, %v238_e1hi, %v238_e3hi : f32
        %v238_va0 = vector.broadcast %v238_alo : f32 to vector<2xf32>
        %v238_va  = vector.insert %v238_ahi, %v238_va0 [1] : f32 into vector<2xf32>
        %v238_vab = arith.truncf %v238_va : vector<2xf32> to vector<2xbf16>
        %v238_vb0 = vector.broadcast %v238_blo : f32 to vector<2xf32>
        %v238_vb  = vector.insert %v238_bhi, %v238_vb0 [1] : f32 into vector<2xf32>
        %v238_vbb = arith.truncf %v238_vb : vector<2xf32> to vector<2xbf16>
        %v238_adra = arith.addi %ep_rb0_oa, %ep_c2a overflow<nsw> : index
        %v238_adrb = arith.addi %ep_rb0_ob, %ep_c2a overflow<nsw> : index
        vector.store %v238_vab, %ep_buf[%v238_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v238_vbb, %ep_buf[%v238_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %240 → rb0, col3
        %v240_e0 = vector.extract %240[0] : f32 from vector<4xf32>
        %v240_e1 = vector.extract %240[1] : f32 from vector<4xf32>
        %v240_e2 = vector.extract %240[2] : f32 from vector<4xf32>
        %v240_e3 = vector.extract %240[3] : f32 from vector<4xf32>
        %v240_e0n, %v240_e0v = gpu.shuffle xor %v240_e0, %ep_xor_off, %ep_xor_w : f32
        %v240_e1n, %v240_e1v = gpu.shuffle xor %v240_e1, %ep_xor_off, %ep_xor_w : f32
        %v240_e2n, %v240_e2v = gpu.shuffle xor %v240_e2, %ep_xor_off, %ep_xor_w : f32
        %v240_e3n, %v240_e3v = gpu.shuffle xor %v240_e3, %ep_xor_off, %ep_xor_w : f32
        %v240_e0lo = arith.select %ep_is_even, %v240_e0,  %v240_e0n : f32
        %v240_e0hi = arith.select %ep_is_even, %v240_e0n, %v240_e0  : f32
        %v240_e1lo = arith.select %ep_is_even, %v240_e1,  %v240_e1n : f32
        %v240_e1hi = arith.select %ep_is_even, %v240_e1n, %v240_e1  : f32
        %v240_e2lo = arith.select %ep_is_even, %v240_e2,  %v240_e2n : f32
        %v240_e2hi = arith.select %ep_is_even, %v240_e2n, %v240_e2  : f32
        %v240_e3lo = arith.select %ep_is_even, %v240_e3,  %v240_e3n : f32
        %v240_e3hi = arith.select %ep_is_even, %v240_e3n, %v240_e3  : f32
        %v240_alo = arith.select %ep_is_even, %v240_e0lo, %v240_e2lo : f32
        %v240_ahi = arith.select %ep_is_even, %v240_e0hi, %v240_e2hi : f32
        %v240_blo = arith.select %ep_is_even, %v240_e1lo, %v240_e3lo : f32
        %v240_bhi = arith.select %ep_is_even, %v240_e1hi, %v240_e3hi : f32
        %v240_va0 = vector.broadcast %v240_alo : f32 to vector<2xf32>
        %v240_va  = vector.insert %v240_ahi, %v240_va0 [1] : f32 into vector<2xf32>
        %v240_vab = arith.truncf %v240_va : vector<2xf32> to vector<2xbf16>
        %v240_vb0 = vector.broadcast %v240_blo : f32 to vector<2xf32>
        %v240_vb  = vector.insert %v240_bhi, %v240_vb0 [1] : f32 into vector<2xf32>
        %v240_vbb = arith.truncf %v240_vb : vector<2xf32> to vector<2xbf16>
        %v240_adra = arith.addi %ep_rb0_oa, %ep_c3a overflow<nsw> : index
        %v240_adrb = arith.addi %ep_rb0_ob, %ep_c3a overflow<nsw> : index
        vector.store %v240_vab, %ep_buf[%v240_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v240_vbb, %ep_buf[%v240_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %242 → rb1, col0
        %v242_e0 = vector.extract %242[0] : f32 from vector<4xf32>
        %v242_e1 = vector.extract %242[1] : f32 from vector<4xf32>
        %v242_e2 = vector.extract %242[2] : f32 from vector<4xf32>
        %v242_e3 = vector.extract %242[3] : f32 from vector<4xf32>
        %v242_e0n, %v242_e0v = gpu.shuffle xor %v242_e0, %ep_xor_off, %ep_xor_w : f32
        %v242_e1n, %v242_e1v = gpu.shuffle xor %v242_e1, %ep_xor_off, %ep_xor_w : f32
        %v242_e2n, %v242_e2v = gpu.shuffle xor %v242_e2, %ep_xor_off, %ep_xor_w : f32
        %v242_e3n, %v242_e3v = gpu.shuffle xor %v242_e3, %ep_xor_off, %ep_xor_w : f32
        %v242_e0lo = arith.select %ep_is_even, %v242_e0,  %v242_e0n : f32
        %v242_e0hi = arith.select %ep_is_even, %v242_e0n, %v242_e0  : f32
        %v242_e1lo = arith.select %ep_is_even, %v242_e1,  %v242_e1n : f32
        %v242_e1hi = arith.select %ep_is_even, %v242_e1n, %v242_e1  : f32
        %v242_e2lo = arith.select %ep_is_even, %v242_e2,  %v242_e2n : f32
        %v242_e2hi = arith.select %ep_is_even, %v242_e2n, %v242_e2  : f32
        %v242_e3lo = arith.select %ep_is_even, %v242_e3,  %v242_e3n : f32
        %v242_e3hi = arith.select %ep_is_even, %v242_e3n, %v242_e3  : f32
        %v242_alo = arith.select %ep_is_even, %v242_e0lo, %v242_e2lo : f32
        %v242_ahi = arith.select %ep_is_even, %v242_e0hi, %v242_e2hi : f32
        %v242_blo = arith.select %ep_is_even, %v242_e1lo, %v242_e3lo : f32
        %v242_bhi = arith.select %ep_is_even, %v242_e1hi, %v242_e3hi : f32
        %v242_va0 = vector.broadcast %v242_alo : f32 to vector<2xf32>
        %v242_va  = vector.insert %v242_ahi, %v242_va0 [1] : f32 into vector<2xf32>
        %v242_vab = arith.truncf %v242_va : vector<2xf32> to vector<2xbf16>
        %v242_vb0 = vector.broadcast %v242_blo : f32 to vector<2xf32>
        %v242_vb  = vector.insert %v242_bhi, %v242_vb0 [1] : f32 into vector<2xf32>
        %v242_vbb = arith.truncf %v242_vb : vector<2xf32> to vector<2xbf16>
        %v242_adra = arith.addi %ep_rb1_oa, %ep_c0a overflow<nsw> : index
        %v242_adrb = arith.addi %ep_rb1_ob, %ep_c0a overflow<nsw> : index
        vector.store %v242_vab, %ep_buf[%v242_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v242_vbb, %ep_buf[%v242_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %244 → rb1, col1
        %v244_e0 = vector.extract %244[0] : f32 from vector<4xf32>
        %v244_e1 = vector.extract %244[1] : f32 from vector<4xf32>
        %v244_e2 = vector.extract %244[2] : f32 from vector<4xf32>
        %v244_e3 = vector.extract %244[3] : f32 from vector<4xf32>
        %v244_e0n, %v244_e0v = gpu.shuffle xor %v244_e0, %ep_xor_off, %ep_xor_w : f32
        %v244_e1n, %v244_e1v = gpu.shuffle xor %v244_e1, %ep_xor_off, %ep_xor_w : f32
        %v244_e2n, %v244_e2v = gpu.shuffle xor %v244_e2, %ep_xor_off, %ep_xor_w : f32
        %v244_e3n, %v244_e3v = gpu.shuffle xor %v244_e3, %ep_xor_off, %ep_xor_w : f32
        %v244_e0lo = arith.select %ep_is_even, %v244_e0,  %v244_e0n : f32
        %v244_e0hi = arith.select %ep_is_even, %v244_e0n, %v244_e0  : f32
        %v244_e1lo = arith.select %ep_is_even, %v244_e1,  %v244_e1n : f32
        %v244_e1hi = arith.select %ep_is_even, %v244_e1n, %v244_e1  : f32
        %v244_e2lo = arith.select %ep_is_even, %v244_e2,  %v244_e2n : f32
        %v244_e2hi = arith.select %ep_is_even, %v244_e2n, %v244_e2  : f32
        %v244_e3lo = arith.select %ep_is_even, %v244_e3,  %v244_e3n : f32
        %v244_e3hi = arith.select %ep_is_even, %v244_e3n, %v244_e3  : f32
        %v244_alo = arith.select %ep_is_even, %v244_e0lo, %v244_e2lo : f32
        %v244_ahi = arith.select %ep_is_even, %v244_e0hi, %v244_e2hi : f32
        %v244_blo = arith.select %ep_is_even, %v244_e1lo, %v244_e3lo : f32
        %v244_bhi = arith.select %ep_is_even, %v244_e1hi, %v244_e3hi : f32
        %v244_va0 = vector.broadcast %v244_alo : f32 to vector<2xf32>
        %v244_va  = vector.insert %v244_ahi, %v244_va0 [1] : f32 into vector<2xf32>
        %v244_vab = arith.truncf %v244_va : vector<2xf32> to vector<2xbf16>
        %v244_vb0 = vector.broadcast %v244_blo : f32 to vector<2xf32>
        %v244_vb  = vector.insert %v244_bhi, %v244_vb0 [1] : f32 into vector<2xf32>
        %v244_vbb = arith.truncf %v244_vb : vector<2xf32> to vector<2xbf16>
        %v244_adra = arith.addi %ep_rb1_oa, %ep_c1a overflow<nsw> : index
        %v244_adrb = arith.addi %ep_rb1_ob, %ep_c1a overflow<nsw> : index
        vector.store %v244_vab, %ep_buf[%v244_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v244_vbb, %ep_buf[%v244_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %246 → rb1, col2
        %v246_e0 = vector.extract %246[0] : f32 from vector<4xf32>
        %v246_e1 = vector.extract %246[1] : f32 from vector<4xf32>
        %v246_e2 = vector.extract %246[2] : f32 from vector<4xf32>
        %v246_e3 = vector.extract %246[3] : f32 from vector<4xf32>
        %v246_e0n, %v246_e0v = gpu.shuffle xor %v246_e0, %ep_xor_off, %ep_xor_w : f32
        %v246_e1n, %v246_e1v = gpu.shuffle xor %v246_e1, %ep_xor_off, %ep_xor_w : f32
        %v246_e2n, %v246_e2v = gpu.shuffle xor %v246_e2, %ep_xor_off, %ep_xor_w : f32
        %v246_e3n, %v246_e3v = gpu.shuffle xor %v246_e3, %ep_xor_off, %ep_xor_w : f32
        %v246_e0lo = arith.select %ep_is_even, %v246_e0,  %v246_e0n : f32
        %v246_e0hi = arith.select %ep_is_even, %v246_e0n, %v246_e0  : f32
        %v246_e1lo = arith.select %ep_is_even, %v246_e1,  %v246_e1n : f32
        %v246_e1hi = arith.select %ep_is_even, %v246_e1n, %v246_e1  : f32
        %v246_e2lo = arith.select %ep_is_even, %v246_e2,  %v246_e2n : f32
        %v246_e2hi = arith.select %ep_is_even, %v246_e2n, %v246_e2  : f32
        %v246_e3lo = arith.select %ep_is_even, %v246_e3,  %v246_e3n : f32
        %v246_e3hi = arith.select %ep_is_even, %v246_e3n, %v246_e3  : f32
        %v246_alo = arith.select %ep_is_even, %v246_e0lo, %v246_e2lo : f32
        %v246_ahi = arith.select %ep_is_even, %v246_e0hi, %v246_e2hi : f32
        %v246_blo = arith.select %ep_is_even, %v246_e1lo, %v246_e3lo : f32
        %v246_bhi = arith.select %ep_is_even, %v246_e1hi, %v246_e3hi : f32
        %v246_va0 = vector.broadcast %v246_alo : f32 to vector<2xf32>
        %v246_va  = vector.insert %v246_ahi, %v246_va0 [1] : f32 into vector<2xf32>
        %v246_vab = arith.truncf %v246_va : vector<2xf32> to vector<2xbf16>
        %v246_vb0 = vector.broadcast %v246_blo : f32 to vector<2xf32>
        %v246_vb  = vector.insert %v246_bhi, %v246_vb0 [1] : f32 into vector<2xf32>
        %v246_vbb = arith.truncf %v246_vb : vector<2xf32> to vector<2xbf16>
        %v246_adra = arith.addi %ep_rb1_oa, %ep_c2a overflow<nsw> : index
        %v246_adrb = arith.addi %ep_rb1_ob, %ep_c2a overflow<nsw> : index
        vector.store %v246_vab, %ep_buf[%v246_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v246_vbb, %ep_buf[%v246_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        // MFMA %248 → rb1, col3
        %v248_e0 = vector.extract %248[0] : f32 from vector<4xf32>
        %v248_e1 = vector.extract %248[1] : f32 from vector<4xf32>
        %v248_e2 = vector.extract %248[2] : f32 from vector<4xf32>
        %v248_e3 = vector.extract %248[3] : f32 from vector<4xf32>
        %v248_e0n, %v248_e0v = gpu.shuffle xor %v248_e0, %ep_xor_off, %ep_xor_w : f32
        %v248_e1n, %v248_e1v = gpu.shuffle xor %v248_e1, %ep_xor_off, %ep_xor_w : f32
        %v248_e2n, %v248_e2v = gpu.shuffle xor %v248_e2, %ep_xor_off, %ep_xor_w : f32
        %v248_e3n, %v248_e3v = gpu.shuffle xor %v248_e3, %ep_xor_off, %ep_xor_w : f32
        %v248_e0lo = arith.select %ep_is_even, %v248_e0,  %v248_e0n : f32
        %v248_e0hi = arith.select %ep_is_even, %v248_e0n, %v248_e0  : f32
        %v248_e1lo = arith.select %ep_is_even, %v248_e1,  %v248_e1n : f32
        %v248_e1hi = arith.select %ep_is_even, %v248_e1n, %v248_e1  : f32
        %v248_e2lo = arith.select %ep_is_even, %v248_e2,  %v248_e2n : f32
        %v248_e2hi = arith.select %ep_is_even, %v248_e2n, %v248_e2  : f32
        %v248_e3lo = arith.select %ep_is_even, %v248_e3,  %v248_e3n : f32
        %v248_e3hi = arith.select %ep_is_even, %v248_e3n, %v248_e3  : f32
        %v248_alo = arith.select %ep_is_even, %v248_e0lo, %v248_e2lo : f32
        %v248_ahi = arith.select %ep_is_even, %v248_e0hi, %v248_e2hi : f32
        %v248_blo = arith.select %ep_is_even, %v248_e1lo, %v248_e3lo : f32
        %v248_bhi = arith.select %ep_is_even, %v248_e1hi, %v248_e3hi : f32
        %v248_va0 = vector.broadcast %v248_alo : f32 to vector<2xf32>
        %v248_va  = vector.insert %v248_ahi, %v248_va0 [1] : f32 into vector<2xf32>
        %v248_vab = arith.truncf %v248_va : vector<2xf32> to vector<2xbf16>
        %v248_vb0 = vector.broadcast %v248_blo : f32 to vector<2xf32>
        %v248_vb  = vector.insert %v248_bhi, %v248_vb0 [1] : f32 into vector<2xf32>
        %v248_vbb = arith.truncf %v248_vb : vector<2xf32> to vector<2xbf16>
        %v248_adra = arith.addi %ep_rb1_oa, %ep_c3a overflow<nsw> : index
        %v248_adrb = arith.addi %ep_rb1_ob, %ep_c3a overflow<nsw> : index
        vector.store %v248_vab, %ep_buf[%v248_adra] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
        vector.store %v248_vbb, %ep_buf[%v248_adrb] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: !hal.fence, %arg11: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg10) => %arg0 : !hal.buffer_view -> tensor<2048x512xi8>
    %1 = hal.tensor.import wait(%arg10) => %arg1 : !hal.buffer_view -> tensor<2048x32xi8>
    %2 = hal.tensor.import wait(%arg10) => %arg2 : !hal.buffer_view -> tensor<2048x512xi8>
    %3 = hal.tensor.import wait(%arg10) => %arg3 : !hal.buffer_view -> tensor<2048x32xi8>
    %4 = hal.tensor.import wait(%arg10) => %arg4 : !hal.buffer_view -> tensor<2048x2048xbf16>
    %5 = flow.dispatch @gemm::@gemm[%arg5, %arg6, %arg7, %arg8, %arg9](%0, %1, %2, %3, %4, %arg5, %arg6, %arg7, %arg8, %arg9) : (tensor<2048x512xi8>, tensor<2048x32xi8>, tensor<2048x512xi8>, tensor<2048x32xi8>, tensor<2048x2048xbf16>, index, index, index, index, index) -> %4
    %6 = hal.tensor.barrier join(%5 : tensor<2048x2048xbf16>) => %arg11 : !hal.fence
    %7 = hal.tensor.export %6 : tensor<2048x2048xbf16> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}

