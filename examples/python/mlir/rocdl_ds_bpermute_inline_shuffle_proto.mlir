// Prototype: rocdl.ds_bpermute as inline XOR-shuffle for MXFP4 epilogue
//
// Problem:
//   mxfp4_epilogue_opt_256x192x256.mlir contains 96 instances of:
//     gpu.shuffle xor %val, %ep_shuffle_offset, %ep_shuffle_width : f32
//   The LLVM Machine Outliner detects the repeated instruction sequences and
//   replaces them with 8 x s_swappc_b64 subroutine calls (12 shuffles per group).
//   Each call adds pipeline-flush overhead and prevents the scheduler from
//   overlapping ds_bpermute LDS latency with surrounding VALU work.
//
// Root-cause timeline:
//   - Apr 22 2025 (commit b67e35a9, IREE 3.4.0rc20250422): Wave switched from
//     manual scalarize-bitcast-shuffle to upstream GPU->ROCDL repacking for
//     gpu.shuffle. The new lowering passes unmodified gpu.shuffle ops to the
//     IREE/LLVM pipeline, which generates ds_bpermute_b32 — but the Machine
//     Outliner then outlines repeated sequences into s_swappc_b64 subroutines.
//
// Fix:
//   Replace gpu.shuffle xor with rocdl.ds_bpermute directly.
//   - rocdl.* ops are already legal inside func.func / stream.executable
//     (the epilogue already uses rocdl.s.barrier, rocdl.sched.barrier, etc.)
//   - rocdl.ds_bpermute lowers directly to ds_bpermute_b32, bypassing the
//     gpu.shuffle lowering path entirely.
//   - With varied surrounding code per shuffle group, the Machine Outliner
//     has less identical sequences to collapse.
//
// Validated with: iree-opt (IREE 3.12.0rc20260410)
//
// ep_shuffle_offset = 1  -> swap adjacent even/odd lanes (lane XOR 1)
// ep_shuffle_width  = 64 -> full wavefront (ds_bpermute covers all 64 lanes)
//
// ds_bpermute semantics:
//   dst[lane] = src[index/4]  where index is the BYTE address of the source lane
//   index = (lane_id XOR xor_offset) * 4

module attributes {gpu.container_module} {
  gpu.module @mxfp4_epilogue_inline_shuffle {

    // One shuffle group covering 4 f32 accumulator rows from a single vector block.
    // In the full epilogue, this pattern repeats 24 times (96 shuffles / 4 rows).
    //
    // This replaces (per row):
    //   %nbr, %valid = gpu.shuffle xor %val, %ep_shuffle_offset, %ep_shuffle_width : f32
    //   %lo = arith.select %ep_is_even, %val, %nbr : f32
    //   %hi = arith.select %ep_is_even, %nbr, %val : f32
    //
    // With the inline equivalent:
    //   %val_i32 = arith.bitcast %val : f32 to i32
    //   %nbr_i32 = rocdl.ds_bpermute %src_addr, %val_i32 : (i32, i32) -> i32
    //   %nbr     = arith.bitcast %nbr_i32 : i32 to f32
    //   (valid flag dropped: width=64 means all lanes are always valid)
    gpu.func @epilogue_ds_bpermute_group(
        %ep_v225_row0 : f32,
        %ep_v225_row1 : f32,
        %ep_v225_row2 : f32,
        %ep_v225_row3 : f32,
        %ep_is_even   : i1,
        %out          : memref<?xf32>
    ) kernel attributes { rocdl.flat_work_group_size = "64,1,1" } {
      %c4_i32 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32  // ep_shuffle_offset = 1

      // Compute the source byte address once; reused for all 4 rows.
      // Equivalent to: index = (lane_id XOR 1) * 4
      %lane_id  = gpu.lane_id
      %lane_i32 = arith.index_cast %lane_id : index to i32
      %xor_lane = arith.xori %lane_i32, %c1_i32 : i32
      %src_addr = arith.muli %xor_lane, %c4_i32 : i32

      // Row 0: bitcast f32->i32, inline ds_bpermute, bitcast i32->f32
      %row0_i32 = arith.bitcast %ep_v225_row0 : f32 to i32
      %nbr0_i32 = rocdl.ds_bpermute %src_addr, %row0_i32 : (i32, i32) -> i32
      %nbr0_f32 = arith.bitcast %nbr0_i32 : i32 to f32

      // Row 1
      %row1_i32 = arith.bitcast %ep_v225_row1 : f32 to i32
      %nbr1_i32 = rocdl.ds_bpermute %src_addr, %row1_i32 : (i32, i32) -> i32
      %nbr1_f32 = arith.bitcast %nbr1_i32 : i32 to f32

      // Row 2
      %row2_i32 = arith.bitcast %ep_v225_row2 : f32 to i32
      %nbr2_i32 = rocdl.ds_bpermute %src_addr, %row2_i32 : (i32, i32) -> i32
      %nbr2_f32 = arith.bitcast %nbr2_i32 : i32 to f32

      // Row 3
      %row3_i32 = arith.bitcast %ep_v225_row3 : f32 to i32
      %nbr3_i32 = rocdl.ds_bpermute %src_addr, %row3_i32 : (i32, i32) -> i32
      %nbr3_f32 = arith.bitcast %nbr3_i32 : i32 to f32

      // Select lo/hi value based on lane parity (matches ep_is_even logic)
      // Even lanes keep their own value as lo, odd lanes keep it as hi.
      %r0_lo = arith.select %ep_is_even, %ep_v225_row0, %nbr0_f32 : f32
      %r0_hi = arith.select %ep_is_even, %nbr0_f32, %ep_v225_row0 : f32
      %r1_lo = arith.select %ep_is_even, %ep_v225_row1, %nbr1_f32 : f32
      %r1_hi = arith.select %ep_is_even, %nbr1_f32, %ep_v225_row1 : f32
      %r2_lo = arith.select %ep_is_even, %ep_v225_row2, %nbr2_f32 : f32
      %r2_hi = arith.select %ep_is_even, %nbr2_f32, %ep_v225_row2 : f32
      %r3_lo = arith.select %ep_is_even, %ep_v225_row3, %nbr3_f32 : f32
      %r3_hi = arith.select %ep_is_even, %nbr3_f32, %ep_v225_row3 : f32

      // In the full epilogue these feed into:
      //   %store_a_lo = arith.select %ep_is_even, %r0_lo, %r2_lo : f32
      //   %store_a_hi = arith.select %ep_is_even, %r0_hi, %r2_hi : f32
      //   %pair_a     = vector<2xf32> [store_a_lo, store_a_hi]
      //   %bf16_a     = arith.truncf %pair_a : vector<2xf32> to vector<2xbf16>
      //   vector.store %bf16_a, %ep_buffer[addr] ...

      memref.store %r0_lo, %out[%lane_id] : memref<?xf32>

      gpu.return
    }
  }
}
