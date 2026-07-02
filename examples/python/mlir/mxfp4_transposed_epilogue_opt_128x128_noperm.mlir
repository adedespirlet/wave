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
#map10 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
#map11 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 4096 - (s0 floordiv 16) * 2048)>
#map12 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 4096 - (s0 floordiv 16) * 2048 + 2048)>
#map13 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048)>
#map14 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 2048)>
#map15 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 4096)>
#map16 = affine_map<()[s0, s1, s2] -> (s0 * 8192 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 6144)>
#map17 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
#map18 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 65536 + s1 * 16384 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8) floordiv 128) * 65536 + 128)>
#map19 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 65536 + s1 * 16384 + s3 * 128 + s4 * 16 + (s2 floordiv 8) * 512 - ((s1 * 32 + s2 floordiv 8 + 64) floordiv 128) * 65536 + 32896)>
#map20 = affine_map<()[s0, s1, s2] -> (s0 * 4096 + s1 * 4 + s2 * 256 + (s1 floordiv 64) * 1024 + ((s1 mod 64) floordiv 16) * 64 + ((s1 mod 16) floordiv 8) * 32 - (s1 floordiv 8) * 32 + 256)>
#map21 = affine_map<()[s0, s1, s2, s3] -> (s0 * 4096 + s1 * 2048 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 + ((s3 mod 16) floordiv 8) * 32 - (s3 floordiv 8) * 32 + 256)>
#map22 = affine_map<()[s0, s1, s2, s3] -> (s0 * 4096 + s1 * 2048 + s2 * 256 + s3 * 4 + ((s3 mod 64) floordiv 16) * 64 + ((s3 mod 16) floordiv 8) * 32 - (s3 floordiv 8) * 32 + 1280)>
#map23 = affine_map<()[s0] -> (s0 * 128)>
#map24 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4)>
#map25 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16)>
#map26 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map27 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map28 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map29 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 16)>
#map30 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 32)>
#map31 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 48)>
#map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map35 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (index, index, index) {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      stream.return %c64, %c64, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index) attributes {translation_info = #translation} {
        %c4_i32 = arith.constant 4 : i32
        %c32_i14 = arith.constant 32 : i14
        %c512_i14 = arith.constant 512 : i14
        %c2147483645_i64 = arith.constant 2147483645 : i64
        %c3 = arith.constant 3 : index
        %c262144_i64 = arith.constant 262144 : i64
        %c4194304_i64 = arith.constant 4194304 : i64
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
        %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<bf16>
        %block_id_x = gpu.block_id x upper_bound 64
        %block_id_y = gpu.block_id y upper_bound 64
        %thread_id_x = gpu.thread_id x upper_bound 256
        %thread_id_y = gpu.thread_id y upper_bound 2
        %reinterpret_cast = memref.reinterpret_cast %4 to offset: [0], sizes: [8192, 8192], strides: [%arg9, 1] : memref<bf16> to memref<8192x8192xbf16, strided<[?, 1]>>
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
        %12 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c4194304_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %12[%11], %alloc_2[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %13 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %14 = gpu.subgroup_broadcast %13, first_active_lane : index
        %15 = affine.apply #map5()[%block_id_x, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %12[%15], %alloc_2[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_4 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_5 = memref.cast %reinterpret_cast_4 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %16 = affine.apply #map3()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        %17 = amdgpu.fat_raw_buffer_cast %cast_5 validBytes(%c4194304_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %17[%16], %alloc_0[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %18 = affine.apply #map5()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %17[%18], %alloc_0[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_6 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_7 = memref.cast %reinterpret_cast_6 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %19 = amdgpu.fat_raw_buffer_cast %cast_7 validBytes(%c262144_i64) cacheSwizzleStride(%c32_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %20 = affine.apply #map6()[%block_id_x, %thread_id_x]
        %21 = vector.load %19[%20] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %reinterpret_cast_8 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_9 = memref.cast %reinterpret_cast_8 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %22 = amdgpu.fat_raw_buffer_cast %cast_9 validBytes(%c262144_i64) cacheSwizzleStride(%c32_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
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
        %31 = affine.apply #map10()[%thread_id_x]
        %32 = arith.xori %31, %9 : index
        %33 = affine.apply #map11()[%thread_id_x, %32]
        %34 = affine.apply #map12()[%thread_id_x, %32]
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
        %47:15 = scf.for %arg10 = %c0 to %c3 step %c1 iter_args(%arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %21, %arg20 = %24, %arg21 = %26, %arg22 = %alloc_2, %arg23 = %alloc_1, %arg24 = %alloc_0, %arg25 = %alloc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>, memref<128x128xi8, #gpu.address_space<workgroup>>, memref<128x128xi8, #gpu.address_space<workgroup>>, memref<128x128xi8, #gpu.address_space<workgroup>>, memref<128x128xi8, #gpu.address_space<workgroup>>) {
          %189 = vector.bitcast %arg21 : vector<4xi8> to vector<4xf8E8M0FNU>
          %190 = vector.bitcast %arg20 : vector<4xi8> to vector<4xf8E8M0FNU>
          %191 = vector.bitcast %arg19 : vector<4xi8> to vector<4xf8E8M0FNU>
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(0)
          rocdl.s.barrier
          %192 = affine.apply #map18()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%192], %arg23[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          %193 = affine.apply #map19()[%block_id_x, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %12[%193], %arg23[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          %194 = affine.apply #map18()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %17[%194], %arg25[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          %195 = affine.apply #map19()[%block_id_y, %thread_id_y, %thread_id_x, %arg10, %10]
          amdgpu.gather_to_lds %17[%195], %arg25[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
          rocdl.sched.barrier 0
          %reinterpret_cast_14 = memref.reinterpret_cast %arg22 to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
          %196 = vector.load %reinterpret_cast_14[%33] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %197 = vector.load %reinterpret_cast_14[%34] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %reinterpret_cast_15 = memref.reinterpret_cast %arg24 to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
          %198 = vector.load %reinterpret_cast_15[%35] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %199 = vector.load %reinterpret_cast_15[%36] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %200 = vector.load %reinterpret_cast_15[%37] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %201 = vector.load %reinterpret_cast_15[%38] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %202 = vector.bitcast %196 : vector<16xi8> to vector<32xf4E2M1FN>
          %203 = vector.bitcast %197 : vector<16xi8> to vector<32xf4E2M1FN>
          %204 = vector.bitcast %198 : vector<16xi8> to vector<32xf4E2M1FN>
          %205 = vector.bitcast %199 : vector<16xi8> to vector<32xf4E2M1FN>
          %206 = vector.bitcast %200 : vector<16xi8> to vector<32xf4E2M1FN>
          %207 = vector.bitcast %201 : vector<16xi8> to vector<32xf4E2M1FN>
          %208 = affine.apply #map20()[%block_id_x, %thread_id_x, %arg10]
          %209 = vector.load %19[%208] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %210 = affine.apply #map21()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %211 = vector.load %22[%210] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          %212 = affine.apply #map22()[%block_id_y, %thread_id_y, %arg10, %thread_id_x]
          %213 = vector.load %22[%212] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %214 = amdgpu.scaled_mfma 16x16x128 (%191[0] * %202) * (%190[0] * %204) + %arg11 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %215 = amdgpu.scaled_mfma 16x16x128 (%191[0] * %202) * (%190[1] * %205) + %arg12 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %216 = amdgpu.scaled_mfma 16x16x128 (%191[0] * %202) * (%189[0] * %206) + %arg13 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %217 = amdgpu.scaled_mfma 16x16x128 (%191[0] * %202) * (%189[1] * %207) + %arg14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %218 = amdgpu.scaled_mfma 16x16x128 (%191[1] * %203) * (%190[0] * %204) + %arg15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %219 = amdgpu.scaled_mfma 16x16x128 (%191[1] * %203) * (%190[1] * %205) + %arg16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %220 = amdgpu.scaled_mfma 16x16x128 (%191[1] * %203) * (%189[0] * %206) + %arg17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %221 = amdgpu.scaled_mfma 16x16x128 (%191[1] * %203) * (%189[1] * %207) + %arg18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.sched.barrier 0
          %222 = vector.load %reinterpret_cast_14[%41] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %223 = vector.load %reinterpret_cast_14[%42] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %224 = vector.load %reinterpret_cast_15[%43] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %225 = vector.load %reinterpret_cast_15[%44] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %226 = vector.load %reinterpret_cast_15[%45] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %227 = vector.load %reinterpret_cast_15[%46] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %228 = vector.bitcast %222 : vector<16xi8> to vector<32xf4E2M1FN>
          %229 = vector.bitcast %223 : vector<16xi8> to vector<32xf4E2M1FN>
          %230 = vector.bitcast %224 : vector<16xi8> to vector<32xf4E2M1FN>
          %231 = vector.bitcast %225 : vector<16xi8> to vector<32xf4E2M1FN>
          %232 = vector.bitcast %226 : vector<16xi8> to vector<32xf4E2M1FN>
          %233 = vector.bitcast %227 : vector<16xi8> to vector<32xf4E2M1FN>
          rocdl.sched.barrier 0
          amdgpu.memory_counter_wait load(3)
          rocdl.s.barrier
          rocdl.sched.barrier 0
          rocdl.s.setprio 1
          %234 = amdgpu.scaled_mfma 16x16x128 (%191[2] * %228) * (%190[2] * %230) + %214 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %235 = amdgpu.scaled_mfma 16x16x128 (%191[2] * %228) * (%190[3] * %231) + %215 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %236 = amdgpu.scaled_mfma 16x16x128 (%191[2] * %228) * (%189[2] * %232) + %216 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %237 = amdgpu.scaled_mfma 16x16x128 (%191[2] * %228) * (%189[3] * %233) + %217 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %238 = amdgpu.scaled_mfma 16x16x128 (%191[3] * %229) * (%190[2] * %230) + %218 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %239 = amdgpu.scaled_mfma 16x16x128 (%191[3] * %229) * (%190[3] * %231) + %219 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %240 = amdgpu.scaled_mfma 16x16x128 (%191[3] * %229) * (%189[2] * %232) + %220 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          %241 = amdgpu.scaled_mfma 16x16x128 (%191[3] * %229) * (%189[3] * %233) + %221 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
          rocdl.s.setprio 0
          rocdl.sched.barrier 0
          scf.yield %234, %235, %236, %237, %238, %239, %240, %241, %209, %211, %213, %arg23, %arg22, %arg25, %arg24 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>, memref<128x128xi8, #gpu.address_space<workgroup>>, memref<128x128xi8, #gpu.address_space<workgroup>>, memref<128x128xi8, #gpu.address_space<workgroup>>, memref<128x128xi8, #gpu.address_space<workgroup>>
        }
        %48 = vector.bitcast %47#10 : vector<4xi8> to vector<4xf8E8M0FNU>
        %49 = vector.bitcast %47#9 : vector<4xi8> to vector<4xf8E8M0FNU>
        %50 = vector.bitcast %47#8 : vector<4xi8> to vector<4xf8E8M0FNU>
        scf.if %30 {
          rocdl.s.barrier
        }
        amdgpu.lds_barrier
        %reinterpret_cast_10 = memref.reinterpret_cast %47#13 to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
        %51 = vector.load %reinterpret_cast_10[%35] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %52 = vector.load %reinterpret_cast_10[%43] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %53 = vector.load %reinterpret_cast_10[%36] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %54 = vector.load %reinterpret_cast_10[%44] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %55 = vector.load %reinterpret_cast_10[%37] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %56 = vector.load %reinterpret_cast_10[%45] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %57 = vector.load %reinterpret_cast_10[%38] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %58 = vector.load %reinterpret_cast_10[%46] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %reinterpret_cast_11 = memref.reinterpret_cast %47#11 to offset: [0], sizes: [16384], strides: [1] : memref<128x128xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
        %59 = vector.load %reinterpret_cast_11[%33] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %60 = vector.load %reinterpret_cast_11[%41] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %61 = vector.load %reinterpret_cast_11[%34] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %62 = vector.load %reinterpret_cast_11[%42] : memref<16384xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %63 = vector.bitcast %59 : vector<16xi8> to vector<32xf4E2M1FN>
        %64 = vector.bitcast %60 : vector<16xi8> to vector<32xf4E2M1FN>
        %65 = vector.bitcast %61 : vector<16xi8> to vector<32xf4E2M1FN>
        %66 = vector.bitcast %62 : vector<16xi8> to vector<32xf4E2M1FN>
        %67 = vector.bitcast %51 : vector<16xi8> to vector<32xf4E2M1FN>
        %68 = vector.bitcast %52 : vector<16xi8> to vector<32xf4E2M1FN>
        %69 = vector.bitcast %53 : vector<16xi8> to vector<32xf4E2M1FN>
        %70 = vector.bitcast %54 : vector<16xi8> to vector<32xf4E2M1FN>
        %71 = vector.bitcast %55 : vector<16xi8> to vector<32xf4E2M1FN>
        %72 = vector.bitcast %56 : vector<16xi8> to vector<32xf4E2M1FN>
        %73 = vector.bitcast %57 : vector<16xi8> to vector<32xf4E2M1FN>
        %74 = vector.bitcast %58 : vector<16xi8> to vector<32xf4E2M1FN>
        %75 = amdgpu.scaled_mfma 16x16x128 (%50[0] * %63) * (%49[0] * %67) + %47#0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %76 = amdgpu.scaled_mfma 16x16x128 (%50[2] * %64) * (%49[2] * %68) + %75 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %77 = amdgpu.scaled_mfma 16x16x128 (%50[0] * %63) * (%49[1] * %69) + %47#1 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %78 = amdgpu.scaled_mfma 16x16x128 (%50[2] * %64) * (%49[3] * %70) + %77 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %79 = amdgpu.scaled_mfma 16x16x128 (%50[0] * %63) * (%48[0] * %71) + %47#2 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %80 = amdgpu.scaled_mfma 16x16x128 (%50[2] * %64) * (%48[2] * %72) + %79 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %81 = amdgpu.scaled_mfma 16x16x128 (%50[0] * %63) * (%48[1] * %73) + %47#3 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %82 = amdgpu.scaled_mfma 16x16x128 (%50[2] * %64) * (%48[3] * %74) + %81 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %83 = amdgpu.scaled_mfma 16x16x128 (%50[1] * %65) * (%49[0] * %67) + %47#4 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %84 = amdgpu.scaled_mfma 16x16x128 (%50[3] * %66) * (%49[2] * %68) + %83 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %85 = amdgpu.scaled_mfma 16x16x128 (%50[1] * %65) * (%49[1] * %69) + %47#5 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %86 = amdgpu.scaled_mfma 16x16x128 (%50[3] * %66) * (%49[3] * %70) + %85 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %87 = amdgpu.scaled_mfma 16x16x128 (%50[1] * %65) * (%48[0] * %71) + %47#6 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %88 = amdgpu.scaled_mfma 16x16x128 (%50[3] * %66) * (%48[2] * %72) + %87 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %89 = amdgpu.scaled_mfma 16x16x128 (%50[1] * %65) * (%48[1] * %73) + %47#7 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        %90 = amdgpu.scaled_mfma 16x16x128 (%50[3] * %66) * (%48[3] * %74) + %89 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
        // ── Transposed epilogue: permlane + bf16-shuffle opt ────────────────
        %tp_blkr = affine.apply #map23()[%block_id_y]
        %tp_blkc = affine.apply #map23()[%block_id_x]
        %tp_bbase, %tp_boff, %tp_bsz:2, %tp_strd:2 = memref.extract_strided_metadata %reinterpret_cast : memref<8192x8192xbf16, strided<[?, 1]>> -> memref<bf16>, index, index, index, index, index
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
        %tp_mbase = affine.apply #map25()[%thread_id_x, %thread_id_y]
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
        // === Pair 0 (rb0,col0-col1): %76, %78 ===
        %v76_b16 = arith.truncf %76 : vector<4xf32> to vector<4xbf16>
        %v78_b16 = arith.truncf %78 : vector<4xf32> to vector<4xbf16>
        %tp0_a01v = vector.extract_strided_slice %v76_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp0_a23v = vector.extract_strided_slice %v76_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp0_a01r = vector.bitcast %tp0_a01v : vector<2xbf16> to vector<1xi32>
        %tp0_a23r = vector.bitcast %tp0_a23v : vector<2xbf16> to vector<1xi32>
        %tp0_a01i = vector.extract %tp0_a01r[0] : i32 from vector<1xi32>
        %tp0_a23i = vector.extract %tp0_a23r[0] : i32 from vector<1xi32>
        %tp0_a01s = amdgpu.permlane_swap %tp0_a01i 16 : i32
        %tp0_a23s = amdgpu.permlane_swap %tp0_a23i 16 : i32
        %tp0_b01v = vector.extract_strided_slice %v78_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp0_b23v = vector.extract_strided_slice %v78_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
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

        // === Pair 1 (rb0,col2-col3): %80, %82 ===
        %v80_b16 = arith.truncf %80 : vector<4xf32> to vector<4xbf16>
        %v82_b16 = arith.truncf %82 : vector<4xf32> to vector<4xbf16>
        %tp1_a01v = vector.extract_strided_slice %v80_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp1_a23v = vector.extract_strided_slice %v80_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp1_a01r = vector.bitcast %tp1_a01v : vector<2xbf16> to vector<1xi32>
        %tp1_a23r = vector.bitcast %tp1_a23v : vector<2xbf16> to vector<1xi32>
        %tp1_a01i = vector.extract %tp1_a01r[0] : i32 from vector<1xi32>
        %tp1_a23i = vector.extract %tp1_a23r[0] : i32 from vector<1xi32>
        %tp1_a01s = amdgpu.permlane_swap %tp1_a01i 16 : i32
        %tp1_a23s = amdgpu.permlane_swap %tp1_a23i 16 : i32
        %tp1_b01v = vector.extract_strided_slice %v82_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp1_b23v = vector.extract_strided_slice %v82_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
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

        // === Pair 2 (rb1,col0-col1): %84, %86 ===
        %v84_b16 = arith.truncf %84 : vector<4xf32> to vector<4xbf16>
        %v86_b16 = arith.truncf %86 : vector<4xf32> to vector<4xbf16>
        %tp2_a01v = vector.extract_strided_slice %v84_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp2_a23v = vector.extract_strided_slice %v84_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp2_a01r = vector.bitcast %tp2_a01v : vector<2xbf16> to vector<1xi32>
        %tp2_a23r = vector.bitcast %tp2_a23v : vector<2xbf16> to vector<1xi32>
        %tp2_a01i = vector.extract %tp2_a01r[0] : i32 from vector<1xi32>
        %tp2_a23i = vector.extract %tp2_a23r[0] : i32 from vector<1xi32>
        %tp2_a01s = amdgpu.permlane_swap %tp2_a01i 16 : i32
        %tp2_a23s = amdgpu.permlane_swap %tp2_a23i 16 : i32
        %tp2_b01v = vector.extract_strided_slice %v86_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp2_b23v = vector.extract_strided_slice %v86_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
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

        // === Pair 3 (rb1,col2-col3): %88, %90 ===
        %v88_b16 = arith.truncf %88 : vector<4xf32> to vector<4xbf16>
        %v90_b16 = arith.truncf %90 : vector<4xf32> to vector<4xbf16>
        %tp3_a01v = vector.extract_strided_slice %v88_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp3_a23v = vector.extract_strided_slice %v88_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp3_a01r = vector.bitcast %tp3_a01v : vector<2xbf16> to vector<1xi32>
        %tp3_a23r = vector.bitcast %tp3_a23v : vector<2xbf16> to vector<1xi32>
        %tp3_a01i = vector.extract %tp3_a01r[0] : i32 from vector<1xi32>
        %tp3_a23i = vector.extract %tp3_a23r[0] : i32 from vector<1xi32>
        %tp3_a01s = amdgpu.permlane_swap %tp3_a01i 16 : i32
        %tp3_a23s = amdgpu.permlane_swap %tp3_a23i 16 : i32
        %tp3_b01v = vector.extract_strided_slice %v90_b16 {offsets = [0], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
        %tp3_b23v = vector.extract_strided_slice %v90_b16 {offsets = [2], sizes = [2], strides = [1]} : vector<4xbf16> to vector<2xbf16>
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
