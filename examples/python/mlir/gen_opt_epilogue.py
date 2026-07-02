#!/usr/bin/env python3
"""
Generates mxfp4_epilogue_opt_256x256_K*.mlir files for every supported K value
by combining:
  - the pre-opt IR header (all ops up to and including the last MFMA)
  - a hand-written XOR-shuffle + vector<2xbf16> store epilogue
  - the @isolated_benchmark$async wrapper (K-dependent tensor shapes)

Usage:
    python3 gen_opt_epilogue.py
"""

import pathlib

HERE = pathlib.Path(__file__).parent
WAVE_ROOT = pathlib.Path("/home/adespirl/wave")

M, N = 2048, 2048  # output matrix fixed shape

# Per-K configuration ──────────────────────────────────────────────────────
# base_map  : map number for the block-offset map (s0 * 256)
# mfma_first: first MFMA f32 source variable number (step 2, 32 vars total)
# preopt    : path to the pre-opt IR dump file
K_CONFIGS = {
    512:  {"base_map": 49, "mfma_first": 259,
           "preopt": WAVE_ROOT / "debug_pre_opt_256x256_K512.txt"},
    1024: {"base_map": 61, "mfma_first": 542,
           "preopt": WAVE_ROOT / "debug_pre_opt_256x256_K1024_unroll.txt"},
    2048: {"base_map": 41, "mfma_first": 129,
           "preopt": WAVE_ROOT / "debug_pre_opt_256x256_K2048.txt"},
    4096: {"base_map": 41, "mfma_first": 129,
           "preopt": WAVE_ROOT / "debug_pre_opt_256x256_K4096.txt"},
    8192: {"base_map": 51, "mfma_first": 267,
           "preopt": WAVE_ROOT / "debug_pre_opt_256x256x256.txt"},
}

# Map layout relative to base_map offset:
#  +0  block-offset  (s0 * 256)
#  +1  rb0_row0
#  +2  col0
#  +3  rb0_row1   +4 rb0_row2   +5 rb0_row3
#  +6  col1 .. +12 col7
#  +13 rb1_row0 .. +16 rb1_row3
#  +17 rb2_row0 .. +20 rb2_row3
#  +21 rb3_row0 .. +24 rb3_row3
_RB_ROW_OFFSETS = {
    "rb0": (1, 3, 4, 5),
    "rb1": (13, 14, 15, 16),
    "rb2": (17, 18, 19, 20),
    "rb3": (21, 22, 23, 24),
}
_COL_OFFSETS = (2, 6, 7, 8, 9, 10, 11, 12)  # col0..col7


def gen_epilogue(base: int, mfma_first: int) -> str:
    m = lambda off: f"#map{base + off}"  # map reference helper
    mfma_vars = [mfma_first + 2 * i for i in range(32)]
    rb_names = list(_RB_ROW_OFFSETS.keys())

    lines = []
    a = lines.append

    a("        // ── Optimized epilogue: XOR-1 shuffle → vector<2xbf16> stores ─────")
    # Buffer setup
    a(f"        %ep_block_x = affine.apply {m(0)}()[%block_id_x]")
    a(f"        %ep_block_y = affine.apply {m(0)}()[%block_id_y]")
    a("        %ep_base_buf, %ep_offset, %ep_sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<2048x2048xbf16, strided<[?, 1]>> -> memref<bf16>, index, index, index, index, index")
    a("        %ep_block_row_off = arith.muli %ep_block_x, %strides#0 overflow<nsw> : index")
    a("        %ep_base_off = arith.addi %ep_block_row_off, %ep_block_y overflow<nsw> : index")
    a("        %ep_rcast = memref.reinterpret_cast %4 to offset: [%ep_base_off], sizes: [1073741822], strides: [1] : memref<bf16> to memref<1073741822xbf16, strided<[1], offset: ?>>")
    a("        %ep_cast  = memref.cast %ep_rcast : memref<1073741822xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>")
    a("        %ep_stride_i14 = arith.index_cast %strides#0 : index to i14")
    a("        %ep_buffer = amdgpu.fat_raw_buffer_cast %ep_cast validBytes(%c2147483645_i64) cacheSwizzleStride(%ep_stride_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>")
    a("")
    # Thread parity
    a("        %ep_shuffle_offset = arith.constant 1 : i32")
    a("        %ep_shuffle_width  = arith.constant 64 : i32")
    a("        %ep_one  = arith.constant 1 : index")
    a("        %ep_zero = arith.constant 0 : index")
    a("        %ep_lane_parity = arith.andi %thread_id_x, %ep_one : index")
    a("        %ep_is_even = arith.cmpi eq, %ep_lane_parity, %ep_zero : index")
    a("")
    # Row-bank store-offset precomputation
    for rb, (r0, r1, r2, r3) in _RB_ROW_OFFSETS.items():
        a(f"        %ep_{rb}_row0 = affine.apply {m(r0)}()[%thread_id_x]")
        a(f"        %ep_{rb}_row1 = affine.apply {m(r1)}()[%thread_id_x]")
        a(f"        %ep_{rb}_row2 = affine.apply {m(r2)}()[%thread_id_x]")
        a(f"        %ep_{rb}_row3 = affine.apply {m(r3)}()[%thread_id_x]")
        a(f"        %ep_{rb}_store_row_a = arith.select %ep_is_even, %ep_{rb}_row0, %ep_{rb}_row2 : index")
        a(f"        %ep_{rb}_store_row_b = arith.select %ep_is_even, %ep_{rb}_row1, %ep_{rb}_row3 : index")
        a(f"        %ep_{rb}_store_off_a = arith.muli %ep_{rb}_store_row_a, %strides#0 overflow<nsw> : index")
        a(f"        %ep_{rb}_store_off_b = arith.muli %ep_{rb}_store_row_b, %strides#0 overflow<nsw> : index")
        a("")
    # Column offsets
    for ci, co in enumerate(_COL_OFFSETS):
        a(f"        %ep_col{ci} = affine.apply {m(co)}()[%thread_id_x, %thread_id_y]")
        a(f"        %ep_col{ci}_adj = arith.subi %ep_col{ci}, %ep_lane_parity : index")
    a("")
    # Per-MFMA XOR shuffle + store
    for i, var in enumerate(mfma_vars):
        rb = rb_names[i // 8]
        ci = i % 8
        vn = f"v{var}"
        a(f"        // MFMA {i}: %{var} → {rb}, col{ci}")
        for row in range(4):
            a(f"        %ep_{vn}_row{row} = vector.extract %{var}[{row}] : f32 from vector<4xf32>")
        for row in range(4):
            a(f"        %ep_{vn}_row{row}_nbr, %ep_{vn}_row{row}_valid = gpu.shuffle xor %ep_{vn}_row{row}, %ep_shuffle_offset, %ep_shuffle_width : f32")
        for row in range(4):
            a(f"        %ep_{vn}_row{row}_lo = arith.select %ep_is_even, %ep_{vn}_row{row},     %ep_{vn}_row{row}_nbr : f32")
            a(f"        %ep_{vn}_row{row}_hi = arith.select %ep_is_even, %ep_{vn}_row{row}_nbr, %ep_{vn}_row{row}     : f32")
        a(f"        %ep_{vn}_store_a_lo = arith.select %ep_is_even, %ep_{vn}_row0_lo, %ep_{vn}_row2_lo : f32")
        a(f"        %ep_{vn}_store_a_hi = arith.select %ep_is_even, %ep_{vn}_row0_hi, %ep_{vn}_row2_hi : f32")
        a(f"        %ep_{vn}_store_b_lo = arith.select %ep_is_even, %ep_{vn}_row1_lo, %ep_{vn}_row3_lo : f32")
        a(f"        %ep_{vn}_store_b_hi = arith.select %ep_is_even, %ep_{vn}_row1_hi, %ep_{vn}_row3_hi : f32")
        a(f"        %ep_{vn}_pair_a_0 = vector.broadcast %ep_{vn}_store_a_lo : f32 to vector<2xf32>")
        a(f"        %ep_{vn}_pair_a   = vector.insert %ep_{vn}_store_a_hi, %ep_{vn}_pair_a_0 [1] : f32 into vector<2xf32>")
        a(f"        %ep_{vn}_store_a  = arith.truncf %ep_{vn}_pair_a : vector<2xf32> to vector<2xbf16>")
        a(f"        %ep_{vn}_pair_b_0 = vector.broadcast %ep_{vn}_store_b_lo : f32 to vector<2xf32>")
        a(f"        %ep_{vn}_pair_b   = vector.insert %ep_{vn}_store_b_hi, %ep_{vn}_pair_b_0 [1] : f32 into vector<2xf32>")
        a(f"        %ep_{vn}_store_b  = arith.truncf %ep_{vn}_pair_b : vector<2xf32> to vector<2xbf16>")
        a(f"        %ep_{vn}_addr_a = arith.addi %ep_{rb}_store_off_a, %ep_col{ci}_adj overflow<nsw> : index")
        a(f"        %ep_{vn}_addr_b = arith.addi %ep_{rb}_store_off_b, %ep_col{ci}_adj overflow<nsw> : index")
        a(f"        vector.store %ep_{vn}_store_a, %ep_buffer[%ep_{vn}_addr_a] {{alignment = 4 : i64}} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>")
        a(f"        vector.store %ep_{vn}_store_b, %ep_buffer[%ep_{vn}_addr_b] {{alignment = 4 : i64}} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>")
        a("")

    return "\n".join(lines)


def gen_benchmark_func(K: int) -> str:
    """@isolated_benchmark$async with K-specific tensor shapes."""
    # MXFP4: 4 bits per element → K/2 bytes; scales: 1 byte per 32 elems → K/32 bytes
    a_type   = f"tensor<{M}x{K // 2}xi8>"
    sc_type  = f"tensor<{M}x{K // 32}xi8>"
    out_type = f"tensor<{M}x{N}xbf16>"
    types = (f"({a_type}, {sc_type}, {a_type}, {sc_type}, {out_type}, "
             f"index, index, index, index, index) -> %4")
    return (
        f'  func.func @isolated_benchmark$async('
        f'%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, '
        f'%arg2: !hal.buffer_view, %arg3: !hal.buffer_view, '
        f'%arg4: !hal.buffer_view, %arg5: index, %arg6: index, '
        f'%arg7: index, %arg8: index, %arg9: index, '
        f'%arg10: !hal.fence, %arg11: !hal.fence) -> !hal.buffer_view {{\n'
        f'    %0 = hal.tensor.import wait(%arg10) => %arg0 : !hal.buffer_view -> {a_type}\n'
        f'    %1 = hal.tensor.import wait(%arg10) => %arg1 : !hal.buffer_view -> {sc_type}\n'
        f'    %2 = hal.tensor.import wait(%arg10) => %arg2 : !hal.buffer_view -> {a_type}\n'
        f'    %3 = hal.tensor.import wait(%arg10) => %arg3 : !hal.buffer_view -> {sc_type}\n'
        f'    %4 = hal.tensor.import wait(%arg10) => %arg4 : !hal.buffer_view -> {out_type}\n'
        f'    %5 = flow.dispatch @gemm::@gemm[%arg5, %arg6, %arg7, %arg8, %arg9]'
        f'(%0, %1, %2, %3, %4, %arg5, %arg6, %arg7, %arg8, %arg9) : {types}\n'
        f'    %6 = hal.tensor.barrier join(%5 : {out_type}) => %arg11 : !hal.fence\n'
        f'    %7 = hal.tensor.export %6 : {out_type} -> !hal.buffer_view\n'
        f'    return %7 : !hal.buffer_view\n'
        f'  }}'
    )


def gen_optimized_ir(K: int, out_path: pathlib.Path) -> None:
    cfg = K_CONFIGS[K]
    preopt_path: pathlib.Path = cfg["preopt"]
    base_map: int = cfg["base_map"]
    mfma_first: int = cfg["mfma_first"]

    all_lines = preopt_path.read_text().splitlines()

    # Strip any non-MLIR trailing lines (e.g. test-passed messages)
    last_brace = len(all_lines)
    for i in range(len(all_lines) - 1, -1, -1):
        if all_lines[i].strip() == "}":
            last_brace = i + 1
            break
    all_lines = all_lines[:last_brace]

    # Find first truncf = start of epilogue section to replace
    first_truncf = next(
        i for i, l in enumerate(all_lines)
        if "arith.truncf" in l and "vector<4xf32> to vector<4xbf16>" in l
    )

    header = "\n".join(all_lines[:first_truncf])

    tail = (
        "        return\n"
        "      }\n"
        "    }\n"
        "  }\n"
        + gen_benchmark_func(K) + "\n"
        + "}\n"
    )

    content = header + "\n" + gen_epilogue(base_map, mfma_first) + "\n" + tail
    out_path.write_text(content)
    print(f"  K={K:5d}: {out_path.name}  ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    OUT_NAMES = {
        512:  "mxfp4_epilogue_opt_256x256_K512.mlir",
        1024: "mxfp4_epilogue_opt_256x256_K1024.mlir",
        2048: "mxfp4_epilogue_opt_256x256_K2048.mlir",
        4096: "mxfp4_epilogue_opt_256x256_K4096.mlir",
        8192: "mxfp4_epilogue_opt_256x256x256.mlir",
    }
    print("Generating optimized epilogue IRs...")
    for K, fname in OUT_NAMES.items():
        gen_optimized_ir(K, HERE / fname)
    print("Done.")
