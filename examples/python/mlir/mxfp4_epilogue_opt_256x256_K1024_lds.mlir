// Handwritten MLIR for MXFP4 GEMM 256x256 K=1024 — LDS staging epilogue.
//
// VARIANT: lds — replaces the XOR-1 shuffle + vector<2xbf16> direct-to-global path
// with AITER-style LDS staging: MFMA → LDS (MFMA-layout-aware write) → barrier
// → linear read-back → buffer_store_dwordx4 to global.
//
// Shape: M=2048, N=2048, K=1024 (8×8 workgroup grid, each block 256×256×1024).
// Inputs: MXFP4 (f4E2M1FN) + f8E8M0FNU scales. Output: bf16, row-major.
//
// === LDS Layout (epilogue staging) ===
//
// A 256×256 bf16 tile (65536 elements = 128 KB) is staged in a flat
// memref<65536xbf16, workgroup> using row-major ordering:
//
//   element (local_row, local_col) → LDS index = local_row * 256 + local_col
//
// MFMA output layout for thread (tid_x, tid_y):
//   wave_id       = tid_x / 64
//   lane_group    = (tid_x % 64) / 16   (0..3 groups of 16 within the wave)
//   lane_in_group = tid_x % 16          (0..15, selects column within a group)
//
//   For MFMA group (rb=0..3, col=0..7), element e (0..3):
//     local_row = wave_id*64 + rb*16 + lane_group*4 + e
//     local_col = lane_in_group + tid_y*128 + col*16
//     lds_idx   = local_row*256 + local_col
//              = lds_rc_base + rb*4096 + col*16 + e*256
//
//   where lds_rc_base = (wave_id*64 + lane_group*4)*256 + lane_in_group + tid_y*128
//
// LDS write: 4 scalar bf16 writes per MFMA group (rows are 256 elements apart).
//            Lowers to 4 × ds_store_short.
//
// === Linear Read-back ===
//
//   global_tid g = tid_y*256 + tid_x  (0..511 across the workgroup)
//   row_base  = g / 2  = tid_y*128 + tid_x/2
//   col_base  = (g%2)*128 = (tid_x%2)*128   [g%2 = tid_x%2 since tid_y*256 is even]
//
//   Each thread reads 128 bf16 in 16 chunks of 8 bf16 (ds_load_b128):
//     chunk k (k=0..15): LDS[g*128 + k*8 .. +7]
//     → global at row_base, col_base + k*8   (always same row, consecutive cols)
//     → buffer_store_dwordx4 (128-bit) vs. the original 2× buffer_store_dword
//
// === LDS reuse strategy (GFX950 / MI300X limit: 160 KB) ===
//
// The A/B tile double-buffers (4 × 32 KB = 128 KB) are only written/read
// during the K-loop. The epilogue staging buffer (65536 × bf16 = 128 KB) is
// only written/read after the K-loop completes. Both use disjoint time ranges,
// so the epilogue staging buffer is mapped as a memref.view of the SAME 128 KB
// base allocation. Total LDS = 128 KB, well within the 160 KB hardware limit.
// This preserves 1–2 workgroup/CU occupancy on MI300X.
//
// Source: mxfp4_epilogue_opt_256x256_K1024.mlir

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
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      stream.return %c8, %c8, %c1 : index, index, index
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
        %block_id_x = gpu.block_id x upper_bound 8
        %block_id_y = gpu.block_id y upper_bound 8
        %thread_id_x = gpu.thread_id x upper_bound 256
        %thread_id_y = gpu.thread_id y upper_bound 2
        %reinterpret_cast = memref.reinterpret_cast %4 to offset: [0], sizes: [2048, 2048], strides: [%arg9, 1] : memref<bf16> to memref<2048x2048xbf16, strided<[?, 1]>>
        // Single 128 KB alloc; GEMM input and epilogue staging share the same
        // LDS region (valid because epilogue only runs after the K-loop finishes,
        // and an amdgpu.lds_barrier flushes all outstanding reads before writes).
        %alloc_base = memref.alloc() : memref<131072xi8, #gpu.address_space<workgroup>>
        %c32768_view = arith.constant 32768 : index
        %c65536_view = arith.constant 65536 : index
        %c98304_view = arith.constant 98304 : index
        %alloc = memref.view %alloc_base[%c0][] : memref<131072xi8, #gpu.address_space<workgroup>> to memref<256x128xi8, #gpu.address_space<workgroup>>
        %alloc_0 = memref.view %alloc_base[%c32768_view][] : memref<131072xi8, #gpu.address_space<workgroup>> to memref<256x128xi8, #gpu.address_space<workgroup>>
        %alloc_1 = memref.view %alloc_base[%c65536_view][] : memref<131072xi8, #gpu.address_space<workgroup>> to memref<256x128xi8, #gpu.address_space<workgroup>>
        %alloc_2 = memref.view %alloc_base[%c98304_view][] : memref<131072xi8, #gpu.address_space<workgroup>> to memref<256x128xi8, #gpu.address_space<workgroup>>
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
        %23 = amdgpu.fat_raw_buffer_cast %cast_5 validBytes(%c1048576_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.gather_to_lds %23[%22], %alloc_0[%6, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %24 = affine.apply #map5()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%24], %alloc_0[%14, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %25 = affine.apply #map7()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%25], %alloc_0[%17, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %26 = affine.apply #map9()[%block_id_y, %thread_id_y, %thread_id_x, %10]
        amdgpu.gather_to_lds %23[%26], %alloc_0[%20, %7] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
        %reinterpret_cast_6 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_7 = memref.cast %reinterpret_cast_6 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %27 = amdgpu.fat_raw_buffer_cast %cast_7 validBytes(%c65536_i64) cacheSwizzleStride(%c32_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
        %28 = affine.apply #map10()[%block_id_x, %thread_id_x]
        %29 = vector.load %27[%28] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %30 = vector.bitcast %29 : vector<4xi8> to vector<4xf8E8M0FNU>
        %31 = affine.apply #map11()[%block_id_x, %thread_id_x]
        %32 = vector.load %27[%31] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %33 = vector.bitcast %32 : vector<4xi8> to vector<4xf8E8M0FNU>
        %reinterpret_cast_8 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
        %cast_9 = memref.cast %reinterpret_cast_8 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
        %34 = amdgpu.fat_raw_buffer_cast %cast_9 validBytes(%c65536_i64) cacheSwizzleStride(%c32_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
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
        // ── LDS staging epilogue ──────────────────────────────────────────────
        // See file header for full documentation of the LDS layout and addressing.
        //
        // Step 1: each thread writes its 32 MFMA groups (128 bf16 values) to
        //         the epilogue staging LDS using MFMA-layout-aware addresses.
        // Step 2: gpu.barrier ensures all writes are visible across the workgroup.
        // Step 3: each thread reads 128 bf16 linearly from LDS and writes them
        //         to global memory as 16 × vector<8xbf16> (buffer_store_dwordx4).

        // ── Epilogue staging buffer: 256×256 bf16 = 128 KB ───────────────────
        %lds_ep = memref.view %alloc_base[%c0][] : memref<131072xi8, #gpu.address_space<workgroup>> to memref<65536xbf16, #gpu.address_space<workgroup>>

        // ── LDS write address precomputation ─────────────────────────────────
        // lds_rc_base = (wave_id*64 + lane_group*4) * 256 + lane_in_group + tid_y*128
        // For group (rb, col), element e:  lds_rc_base + rb*4096 + col*16 + e*256
        %lds_c4     = arith.constant    4 : index
        %lds_c8     = arith.constant    8 : index
        %lds_c16    = arith.constant   16 : index
        %lds_c64    = arith.constant   64 : index
        %lds_c128   = arith.constant  128 : index
        %lds_c256   = arith.constant  256 : index
        %lds_c512   = arith.constant  512 : index
        %lds_c768   = arith.constant  768 : index
        %lds_c4096  = arith.constant 4096 : index
        %lds_c8192  = arith.constant 8192 : index
        %lds_c12288 = arith.constant 12288 : index

        %lds_wave   = arith.divui %thread_id_x, %lds_c64  : index
        %lds_mod64  = arith.remui %thread_id_x, %lds_c64  : index
        %lds_lgrp   = arith.divui %lds_mod64,   %lds_c16  : index
        %lds_lin    = arith.remui %thread_id_x, %lds_c16  : index
        %lds_wb64   = arith.muli  %lds_wave, %lds_c64   overflow<nsw> : index
        %lds_lg4    = arith.muli  %lds_lgrp, %lds_c4    overflow<nsw> : index
        %lds_wrb    = arith.addi  %lds_wb64, %lds_lg4   overflow<nsw> : index
        %lds_tidy128 = arith.muli %thread_id_y, %lds_c128 overflow<nsw> : index
        %lds_wcb    = arith.addi  %lds_lin,  %lds_tidy128  overflow<nsw> : index
        %lds_wrb256 = arith.muli  %lds_wrb,  %lds_c256  overflow<nsw> : index
        %lds_rcbase = arith.addi  %lds_wrb256, %lds_wcb overflow<nsw> : index

        // Precompute column-offset bases for each row-band.
        // rb0: rcbase + col*16 (col = 0..7)
        %lds_rb0c1  = arith.addi %lds_rcbase, %lds_c16   overflow<nsw> : index
        %lds_rb0c2  = arith.addi %lds_rb0c1,  %lds_c16   overflow<nsw> : index
        %lds_rb0c3  = arith.addi %lds_rb0c2,  %lds_c16   overflow<nsw> : index
        %lds_rb0c4  = arith.addi %lds_rb0c3,  %lds_c16   overflow<nsw> : index
        %lds_rb0c5  = arith.addi %lds_rb0c4,  %lds_c16   overflow<nsw> : index
        %lds_rb0c6  = arith.addi %lds_rb0c5,  %lds_c16   overflow<nsw> : index
        %lds_rb0c7  = arith.addi %lds_rb0c6,  %lds_c16   overflow<nsw> : index
        // rb1: rcbase + 4096 + col*16
        %lds_rb1c0  = arith.addi %lds_rcbase, %lds_c4096  overflow<nsw> : index
        %lds_rb1c1  = arith.addi %lds_rb1c0,  %lds_c16   overflow<nsw> : index
        %lds_rb1c2  = arith.addi %lds_rb1c1,  %lds_c16   overflow<nsw> : index
        %lds_rb1c3  = arith.addi %lds_rb1c2,  %lds_c16   overflow<nsw> : index
        %lds_rb1c4  = arith.addi %lds_rb1c3,  %lds_c16   overflow<nsw> : index
        %lds_rb1c5  = arith.addi %lds_rb1c4,  %lds_c16   overflow<nsw> : index
        %lds_rb1c6  = arith.addi %lds_rb1c5,  %lds_c16   overflow<nsw> : index
        %lds_rb1c7  = arith.addi %lds_rb1c6,  %lds_c16   overflow<nsw> : index
        // rb2: rcbase + 8192 + col*16
        %lds_rb2c0  = arith.addi %lds_rcbase, %lds_c8192  overflow<nsw> : index
        %lds_rb2c1  = arith.addi %lds_rb2c0,  %lds_c16   overflow<nsw> : index
        %lds_rb2c2  = arith.addi %lds_rb2c1,  %lds_c16   overflow<nsw> : index
        %lds_rb2c3  = arith.addi %lds_rb2c2,  %lds_c16   overflow<nsw> : index
        %lds_rb2c4  = arith.addi %lds_rb2c3,  %lds_c16   overflow<nsw> : index
        %lds_rb2c5  = arith.addi %lds_rb2c4,  %lds_c16   overflow<nsw> : index
        %lds_rb2c6  = arith.addi %lds_rb2c5,  %lds_c16   overflow<nsw> : index
        %lds_rb2c7  = arith.addi %lds_rb2c6,  %lds_c16   overflow<nsw> : index
        // rb3: rcbase + 12288 + col*16
        %lds_rb3c0  = arith.addi %lds_rcbase, %lds_c12288 overflow<nsw> : index
        %lds_rb3c1  = arith.addi %lds_rb3c0,  %lds_c16   overflow<nsw> : index
        %lds_rb3c2  = arith.addi %lds_rb3c1,  %lds_c16   overflow<nsw> : index
        %lds_rb3c3  = arith.addi %lds_rb3c2,  %lds_c16   overflow<nsw> : index
        %lds_rb3c4  = arith.addi %lds_rb3c3,  %lds_c16   overflow<nsw> : index
        %lds_rb3c5  = arith.addi %lds_rb3c4,  %lds_c16   overflow<nsw> : index
        %lds_rb3c6  = arith.addi %lds_rb3c5,  %lds_c16   overflow<nsw> : index
        %lds_rb3c7  = arith.addi %lds_rb3c6,  %lds_c16   overflow<nsw> : index

        // ── LDS fence: wait for all GEMM LDS reads before writing epilogue ──
        // The epilogue staging buffer aliases the GEMM input LDS region.
        // s_waitcnt lgkmcnt(0) + s_barrier ensures all outstanding DS_READs
        // (from reinterpret_cast_12 / _13) complete before the DS_WRITEs below.
        amdgpu.lds_barrier

        // ── Step 1: truncf f32→bf16, write 4 scalar bf16 per MFMA group ──────
        // Each group writes to rows {base, base+1, base+2, base+3} at same col.
        // Row stride in LDS = 256 bf16; element e → LDS[group_base + e*256].

        // Group (rb=0, col=0) → acc %542
        %lds_542    = arith.truncf %542 : vector<4xf32> to vector<4xbf16>
        %lds_542_e0 = vector.extract %lds_542[0] : bf16 from vector<4xbf16>
        %lds_542_e1 = vector.extract %lds_542[1] : bf16 from vector<4xbf16>
        %lds_542_e2 = vector.extract %lds_542[2] : bf16 from vector<4xbf16>
        %lds_542_e3 = vector.extract %lds_542[3] : bf16 from vector<4xbf16>
        memref.store %lds_542_e0, %lds_ep[%lds_rcbase] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_542_i1 = arith.addi %lds_rcbase, %lds_c256 overflow<nsw> : index
        memref.store %lds_542_e1, %lds_ep[%lds_542_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_542_i2 = arith.addi %lds_rcbase, %lds_c512 overflow<nsw> : index
        memref.store %lds_542_e2, %lds_ep[%lds_542_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_542_i3 = arith.addi %lds_rcbase, %lds_c768 overflow<nsw> : index
        memref.store %lds_542_e3, %lds_ep[%lds_542_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=0, col=1) → acc %544
        %lds_544    = arith.truncf %544 : vector<4xf32> to vector<4xbf16>
        %lds_544_e0 = vector.extract %lds_544[0] : bf16 from vector<4xbf16>
        %lds_544_e1 = vector.extract %lds_544[1] : bf16 from vector<4xbf16>
        %lds_544_e2 = vector.extract %lds_544[2] : bf16 from vector<4xbf16>
        %lds_544_e3 = vector.extract %lds_544[3] : bf16 from vector<4xbf16>
        memref.store %lds_544_e0, %lds_ep[%lds_rb0c1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_544_i1 = arith.addi %lds_rb0c1, %lds_c256 overflow<nsw> : index
        memref.store %lds_544_e1, %lds_ep[%lds_544_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_544_i2 = arith.addi %lds_rb0c1, %lds_c512 overflow<nsw> : index
        memref.store %lds_544_e2, %lds_ep[%lds_544_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_544_i3 = arith.addi %lds_rb0c1, %lds_c768 overflow<nsw> : index
        memref.store %lds_544_e3, %lds_ep[%lds_544_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=0, col=2) → acc %546
        %lds_546    = arith.truncf %546 : vector<4xf32> to vector<4xbf16>
        %lds_546_e0 = vector.extract %lds_546[0] : bf16 from vector<4xbf16>
        %lds_546_e1 = vector.extract %lds_546[1] : bf16 from vector<4xbf16>
        %lds_546_e2 = vector.extract %lds_546[2] : bf16 from vector<4xbf16>
        %lds_546_e3 = vector.extract %lds_546[3] : bf16 from vector<4xbf16>
        memref.store %lds_546_e0, %lds_ep[%lds_rb0c2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_546_i1 = arith.addi %lds_rb0c2, %lds_c256 overflow<nsw> : index
        memref.store %lds_546_e1, %lds_ep[%lds_546_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_546_i2 = arith.addi %lds_rb0c2, %lds_c512 overflow<nsw> : index
        memref.store %lds_546_e2, %lds_ep[%lds_546_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_546_i3 = arith.addi %lds_rb0c2, %lds_c768 overflow<nsw> : index
        memref.store %lds_546_e3, %lds_ep[%lds_546_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=0, col=3) → acc %548
        %lds_548    = arith.truncf %548 : vector<4xf32> to vector<4xbf16>
        %lds_548_e0 = vector.extract %lds_548[0] : bf16 from vector<4xbf16>
        %lds_548_e1 = vector.extract %lds_548[1] : bf16 from vector<4xbf16>
        %lds_548_e2 = vector.extract %lds_548[2] : bf16 from vector<4xbf16>
        %lds_548_e3 = vector.extract %lds_548[3] : bf16 from vector<4xbf16>
        memref.store %lds_548_e0, %lds_ep[%lds_rb0c3] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_548_i1 = arith.addi %lds_rb0c3, %lds_c256 overflow<nsw> : index
        memref.store %lds_548_e1, %lds_ep[%lds_548_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_548_i2 = arith.addi %lds_rb0c3, %lds_c512 overflow<nsw> : index
        memref.store %lds_548_e2, %lds_ep[%lds_548_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_548_i3 = arith.addi %lds_rb0c3, %lds_c768 overflow<nsw> : index
        memref.store %lds_548_e3, %lds_ep[%lds_548_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=0, col=4) → acc %550
        %lds_550    = arith.truncf %550 : vector<4xf32> to vector<4xbf16>
        %lds_550_e0 = vector.extract %lds_550[0] : bf16 from vector<4xbf16>
        %lds_550_e1 = vector.extract %lds_550[1] : bf16 from vector<4xbf16>
        %lds_550_e2 = vector.extract %lds_550[2] : bf16 from vector<4xbf16>
        %lds_550_e3 = vector.extract %lds_550[3] : bf16 from vector<4xbf16>
        memref.store %lds_550_e0, %lds_ep[%lds_rb0c4] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_550_i1 = arith.addi %lds_rb0c4, %lds_c256 overflow<nsw> : index
        memref.store %lds_550_e1, %lds_ep[%lds_550_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_550_i2 = arith.addi %lds_rb0c4, %lds_c512 overflow<nsw> : index
        memref.store %lds_550_e2, %lds_ep[%lds_550_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_550_i3 = arith.addi %lds_rb0c4, %lds_c768 overflow<nsw> : index
        memref.store %lds_550_e3, %lds_ep[%lds_550_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=0, col=5) → acc %552
        %lds_552    = arith.truncf %552 : vector<4xf32> to vector<4xbf16>
        %lds_552_e0 = vector.extract %lds_552[0] : bf16 from vector<4xbf16>
        %lds_552_e1 = vector.extract %lds_552[1] : bf16 from vector<4xbf16>
        %lds_552_e2 = vector.extract %lds_552[2] : bf16 from vector<4xbf16>
        %lds_552_e3 = vector.extract %lds_552[3] : bf16 from vector<4xbf16>
        memref.store %lds_552_e0, %lds_ep[%lds_rb0c5] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_552_i1 = arith.addi %lds_rb0c5, %lds_c256 overflow<nsw> : index
        memref.store %lds_552_e1, %lds_ep[%lds_552_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_552_i2 = arith.addi %lds_rb0c5, %lds_c512 overflow<nsw> : index
        memref.store %lds_552_e2, %lds_ep[%lds_552_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_552_i3 = arith.addi %lds_rb0c5, %lds_c768 overflow<nsw> : index
        memref.store %lds_552_e3, %lds_ep[%lds_552_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=0, col=6) → acc %554
        %lds_554    = arith.truncf %554 : vector<4xf32> to vector<4xbf16>
        %lds_554_e0 = vector.extract %lds_554[0] : bf16 from vector<4xbf16>
        %lds_554_e1 = vector.extract %lds_554[1] : bf16 from vector<4xbf16>
        %lds_554_e2 = vector.extract %lds_554[2] : bf16 from vector<4xbf16>
        %lds_554_e3 = vector.extract %lds_554[3] : bf16 from vector<4xbf16>
        memref.store %lds_554_e0, %lds_ep[%lds_rb0c6] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_554_i1 = arith.addi %lds_rb0c6, %lds_c256 overflow<nsw> : index
        memref.store %lds_554_e1, %lds_ep[%lds_554_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_554_i2 = arith.addi %lds_rb0c6, %lds_c512 overflow<nsw> : index
        memref.store %lds_554_e2, %lds_ep[%lds_554_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_554_i3 = arith.addi %lds_rb0c6, %lds_c768 overflow<nsw> : index
        memref.store %lds_554_e3, %lds_ep[%lds_554_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=0, col=7) → acc %556
        %lds_556    = arith.truncf %556 : vector<4xf32> to vector<4xbf16>
        %lds_556_e0 = vector.extract %lds_556[0] : bf16 from vector<4xbf16>
        %lds_556_e1 = vector.extract %lds_556[1] : bf16 from vector<4xbf16>
        %lds_556_e2 = vector.extract %lds_556[2] : bf16 from vector<4xbf16>
        %lds_556_e3 = vector.extract %lds_556[3] : bf16 from vector<4xbf16>
        memref.store %lds_556_e0, %lds_ep[%lds_rb0c7] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_556_i1 = arith.addi %lds_rb0c7, %lds_c256 overflow<nsw> : index
        memref.store %lds_556_e1, %lds_ep[%lds_556_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_556_i2 = arith.addi %lds_rb0c7, %lds_c512 overflow<nsw> : index
        memref.store %lds_556_e2, %lds_ep[%lds_556_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_556_i3 = arith.addi %lds_rb0c7, %lds_c768 overflow<nsw> : index
        memref.store %lds_556_e3, %lds_ep[%lds_556_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=1, col=0) → acc %558
        %lds_558    = arith.truncf %558 : vector<4xf32> to vector<4xbf16>
        %lds_558_e0 = vector.extract %lds_558[0] : bf16 from vector<4xbf16>
        %lds_558_e1 = vector.extract %lds_558[1] : bf16 from vector<4xbf16>
        %lds_558_e2 = vector.extract %lds_558[2] : bf16 from vector<4xbf16>
        %lds_558_e3 = vector.extract %lds_558[3] : bf16 from vector<4xbf16>
        memref.store %lds_558_e0, %lds_ep[%lds_rb1c0] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_558_i1 = arith.addi %lds_rb1c0, %lds_c256 overflow<nsw> : index
        memref.store %lds_558_e1, %lds_ep[%lds_558_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_558_i2 = arith.addi %lds_rb1c0, %lds_c512 overflow<nsw> : index
        memref.store %lds_558_e2, %lds_ep[%lds_558_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_558_i3 = arith.addi %lds_rb1c0, %lds_c768 overflow<nsw> : index
        memref.store %lds_558_e3, %lds_ep[%lds_558_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=1, col=1) → acc %560
        %lds_560    = arith.truncf %560 : vector<4xf32> to vector<4xbf16>
        %lds_560_e0 = vector.extract %lds_560[0] : bf16 from vector<4xbf16>
        %lds_560_e1 = vector.extract %lds_560[1] : bf16 from vector<4xbf16>
        %lds_560_e2 = vector.extract %lds_560[2] : bf16 from vector<4xbf16>
        %lds_560_e3 = vector.extract %lds_560[3] : bf16 from vector<4xbf16>
        memref.store %lds_560_e0, %lds_ep[%lds_rb1c1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_560_i1 = arith.addi %lds_rb1c1, %lds_c256 overflow<nsw> : index
        memref.store %lds_560_e1, %lds_ep[%lds_560_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_560_i2 = arith.addi %lds_rb1c1, %lds_c512 overflow<nsw> : index
        memref.store %lds_560_e2, %lds_ep[%lds_560_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_560_i3 = arith.addi %lds_rb1c1, %lds_c768 overflow<nsw> : index
        memref.store %lds_560_e3, %lds_ep[%lds_560_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=1, col=2) → acc %562
        %lds_562    = arith.truncf %562 : vector<4xf32> to vector<4xbf16>
        %lds_562_e0 = vector.extract %lds_562[0] : bf16 from vector<4xbf16>
        %lds_562_e1 = vector.extract %lds_562[1] : bf16 from vector<4xbf16>
        %lds_562_e2 = vector.extract %lds_562[2] : bf16 from vector<4xbf16>
        %lds_562_e3 = vector.extract %lds_562[3] : bf16 from vector<4xbf16>
        memref.store %lds_562_e0, %lds_ep[%lds_rb1c2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_562_i1 = arith.addi %lds_rb1c2, %lds_c256 overflow<nsw> : index
        memref.store %lds_562_e1, %lds_ep[%lds_562_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_562_i2 = arith.addi %lds_rb1c2, %lds_c512 overflow<nsw> : index
        memref.store %lds_562_e2, %lds_ep[%lds_562_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_562_i3 = arith.addi %lds_rb1c2, %lds_c768 overflow<nsw> : index
        memref.store %lds_562_e3, %lds_ep[%lds_562_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=1, col=3) → acc %564
        %lds_564    = arith.truncf %564 : vector<4xf32> to vector<4xbf16>
        %lds_564_e0 = vector.extract %lds_564[0] : bf16 from vector<4xbf16>
        %lds_564_e1 = vector.extract %lds_564[1] : bf16 from vector<4xbf16>
        %lds_564_e2 = vector.extract %lds_564[2] : bf16 from vector<4xbf16>
        %lds_564_e3 = vector.extract %lds_564[3] : bf16 from vector<4xbf16>
        memref.store %lds_564_e0, %lds_ep[%lds_rb1c3] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_564_i1 = arith.addi %lds_rb1c3, %lds_c256 overflow<nsw> : index
        memref.store %lds_564_e1, %lds_ep[%lds_564_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_564_i2 = arith.addi %lds_rb1c3, %lds_c512 overflow<nsw> : index
        memref.store %lds_564_e2, %lds_ep[%lds_564_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_564_i3 = arith.addi %lds_rb1c3, %lds_c768 overflow<nsw> : index
        memref.store %lds_564_e3, %lds_ep[%lds_564_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=1, col=4) → acc %566
        %lds_566    = arith.truncf %566 : vector<4xf32> to vector<4xbf16>
        %lds_566_e0 = vector.extract %lds_566[0] : bf16 from vector<4xbf16>
        %lds_566_e1 = vector.extract %lds_566[1] : bf16 from vector<4xbf16>
        %lds_566_e2 = vector.extract %lds_566[2] : bf16 from vector<4xbf16>
        %lds_566_e3 = vector.extract %lds_566[3] : bf16 from vector<4xbf16>
        memref.store %lds_566_e0, %lds_ep[%lds_rb1c4] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_566_i1 = arith.addi %lds_rb1c4, %lds_c256 overflow<nsw> : index
        memref.store %lds_566_e1, %lds_ep[%lds_566_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_566_i2 = arith.addi %lds_rb1c4, %lds_c512 overflow<nsw> : index
        memref.store %lds_566_e2, %lds_ep[%lds_566_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_566_i3 = arith.addi %lds_rb1c4, %lds_c768 overflow<nsw> : index
        memref.store %lds_566_e3, %lds_ep[%lds_566_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=1, col=5) → acc %568
        %lds_568    = arith.truncf %568 : vector<4xf32> to vector<4xbf16>
        %lds_568_e0 = vector.extract %lds_568[0] : bf16 from vector<4xbf16>
        %lds_568_e1 = vector.extract %lds_568[1] : bf16 from vector<4xbf16>
        %lds_568_e2 = vector.extract %lds_568[2] : bf16 from vector<4xbf16>
        %lds_568_e3 = vector.extract %lds_568[3] : bf16 from vector<4xbf16>
        memref.store %lds_568_e0, %lds_ep[%lds_rb1c5] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_568_i1 = arith.addi %lds_rb1c5, %lds_c256 overflow<nsw> : index
        memref.store %lds_568_e1, %lds_ep[%lds_568_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_568_i2 = arith.addi %lds_rb1c5, %lds_c512 overflow<nsw> : index
        memref.store %lds_568_e2, %lds_ep[%lds_568_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_568_i3 = arith.addi %lds_rb1c5, %lds_c768 overflow<nsw> : index
        memref.store %lds_568_e3, %lds_ep[%lds_568_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=1, col=6) → acc %570
        %lds_570    = arith.truncf %570 : vector<4xf32> to vector<4xbf16>
        %lds_570_e0 = vector.extract %lds_570[0] : bf16 from vector<4xbf16>
        %lds_570_e1 = vector.extract %lds_570[1] : bf16 from vector<4xbf16>
        %lds_570_e2 = vector.extract %lds_570[2] : bf16 from vector<4xbf16>
        %lds_570_e3 = vector.extract %lds_570[3] : bf16 from vector<4xbf16>
        memref.store %lds_570_e0, %lds_ep[%lds_rb1c6] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_570_i1 = arith.addi %lds_rb1c6, %lds_c256 overflow<nsw> : index
        memref.store %lds_570_e1, %lds_ep[%lds_570_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_570_i2 = arith.addi %lds_rb1c6, %lds_c512 overflow<nsw> : index
        memref.store %lds_570_e2, %lds_ep[%lds_570_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_570_i3 = arith.addi %lds_rb1c6, %lds_c768 overflow<nsw> : index
        memref.store %lds_570_e3, %lds_ep[%lds_570_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=1, col=7) → acc %572
        %lds_572    = arith.truncf %572 : vector<4xf32> to vector<4xbf16>
        %lds_572_e0 = vector.extract %lds_572[0] : bf16 from vector<4xbf16>
        %lds_572_e1 = vector.extract %lds_572[1] : bf16 from vector<4xbf16>
        %lds_572_e2 = vector.extract %lds_572[2] : bf16 from vector<4xbf16>
        %lds_572_e3 = vector.extract %lds_572[3] : bf16 from vector<4xbf16>
        memref.store %lds_572_e0, %lds_ep[%lds_rb1c7] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_572_i1 = arith.addi %lds_rb1c7, %lds_c256 overflow<nsw> : index
        memref.store %lds_572_e1, %lds_ep[%lds_572_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_572_i2 = arith.addi %lds_rb1c7, %lds_c512 overflow<nsw> : index
        memref.store %lds_572_e2, %lds_ep[%lds_572_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_572_i3 = arith.addi %lds_rb1c7, %lds_c768 overflow<nsw> : index
        memref.store %lds_572_e3, %lds_ep[%lds_572_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=2, col=0) → acc %574
        %lds_574    = arith.truncf %574 : vector<4xf32> to vector<4xbf16>
        %lds_574_e0 = vector.extract %lds_574[0] : bf16 from vector<4xbf16>
        %lds_574_e1 = vector.extract %lds_574[1] : bf16 from vector<4xbf16>
        %lds_574_e2 = vector.extract %lds_574[2] : bf16 from vector<4xbf16>
        %lds_574_e3 = vector.extract %lds_574[3] : bf16 from vector<4xbf16>
        memref.store %lds_574_e0, %lds_ep[%lds_rb2c0] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_574_i1 = arith.addi %lds_rb2c0, %lds_c256 overflow<nsw> : index
        memref.store %lds_574_e1, %lds_ep[%lds_574_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_574_i2 = arith.addi %lds_rb2c0, %lds_c512 overflow<nsw> : index
        memref.store %lds_574_e2, %lds_ep[%lds_574_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_574_i3 = arith.addi %lds_rb2c0, %lds_c768 overflow<nsw> : index
        memref.store %lds_574_e3, %lds_ep[%lds_574_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=2, col=1) → acc %576
        %lds_576    = arith.truncf %576 : vector<4xf32> to vector<4xbf16>
        %lds_576_e0 = vector.extract %lds_576[0] : bf16 from vector<4xbf16>
        %lds_576_e1 = vector.extract %lds_576[1] : bf16 from vector<4xbf16>
        %lds_576_e2 = vector.extract %lds_576[2] : bf16 from vector<4xbf16>
        %lds_576_e3 = vector.extract %lds_576[3] : bf16 from vector<4xbf16>
        memref.store %lds_576_e0, %lds_ep[%lds_rb2c1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_576_i1 = arith.addi %lds_rb2c1, %lds_c256 overflow<nsw> : index
        memref.store %lds_576_e1, %lds_ep[%lds_576_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_576_i2 = arith.addi %lds_rb2c1, %lds_c512 overflow<nsw> : index
        memref.store %lds_576_e2, %lds_ep[%lds_576_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_576_i3 = arith.addi %lds_rb2c1, %lds_c768 overflow<nsw> : index
        memref.store %lds_576_e3, %lds_ep[%lds_576_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=2, col=2) → acc %578
        %lds_578    = arith.truncf %578 : vector<4xf32> to vector<4xbf16>
        %lds_578_e0 = vector.extract %lds_578[0] : bf16 from vector<4xbf16>
        %lds_578_e1 = vector.extract %lds_578[1] : bf16 from vector<4xbf16>
        %lds_578_e2 = vector.extract %lds_578[2] : bf16 from vector<4xbf16>
        %lds_578_e3 = vector.extract %lds_578[3] : bf16 from vector<4xbf16>
        memref.store %lds_578_e0, %lds_ep[%lds_rb2c2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_578_i1 = arith.addi %lds_rb2c2, %lds_c256 overflow<nsw> : index
        memref.store %lds_578_e1, %lds_ep[%lds_578_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_578_i2 = arith.addi %lds_rb2c2, %lds_c512 overflow<nsw> : index
        memref.store %lds_578_e2, %lds_ep[%lds_578_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_578_i3 = arith.addi %lds_rb2c2, %lds_c768 overflow<nsw> : index
        memref.store %lds_578_e3, %lds_ep[%lds_578_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=2, col=3) → acc %580
        %lds_580    = arith.truncf %580 : vector<4xf32> to vector<4xbf16>
        %lds_580_e0 = vector.extract %lds_580[0] : bf16 from vector<4xbf16>
        %lds_580_e1 = vector.extract %lds_580[1] : bf16 from vector<4xbf16>
        %lds_580_e2 = vector.extract %lds_580[2] : bf16 from vector<4xbf16>
        %lds_580_e3 = vector.extract %lds_580[3] : bf16 from vector<4xbf16>
        memref.store %lds_580_e0, %lds_ep[%lds_rb2c3] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_580_i1 = arith.addi %lds_rb2c3, %lds_c256 overflow<nsw> : index
        memref.store %lds_580_e1, %lds_ep[%lds_580_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_580_i2 = arith.addi %lds_rb2c3, %lds_c512 overflow<nsw> : index
        memref.store %lds_580_e2, %lds_ep[%lds_580_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_580_i3 = arith.addi %lds_rb2c3, %lds_c768 overflow<nsw> : index
        memref.store %lds_580_e3, %lds_ep[%lds_580_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=2, col=4) → acc %582
        %lds_582    = arith.truncf %582 : vector<4xf32> to vector<4xbf16>
        %lds_582_e0 = vector.extract %lds_582[0] : bf16 from vector<4xbf16>
        %lds_582_e1 = vector.extract %lds_582[1] : bf16 from vector<4xbf16>
        %lds_582_e2 = vector.extract %lds_582[2] : bf16 from vector<4xbf16>
        %lds_582_e3 = vector.extract %lds_582[3] : bf16 from vector<4xbf16>
        memref.store %lds_582_e0, %lds_ep[%lds_rb2c4] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_582_i1 = arith.addi %lds_rb2c4, %lds_c256 overflow<nsw> : index
        memref.store %lds_582_e1, %lds_ep[%lds_582_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_582_i2 = arith.addi %lds_rb2c4, %lds_c512 overflow<nsw> : index
        memref.store %lds_582_e2, %lds_ep[%lds_582_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_582_i3 = arith.addi %lds_rb2c4, %lds_c768 overflow<nsw> : index
        memref.store %lds_582_e3, %lds_ep[%lds_582_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=2, col=5) → acc %584
        %lds_584    = arith.truncf %584 : vector<4xf32> to vector<4xbf16>
        %lds_584_e0 = vector.extract %lds_584[0] : bf16 from vector<4xbf16>
        %lds_584_e1 = vector.extract %lds_584[1] : bf16 from vector<4xbf16>
        %lds_584_e2 = vector.extract %lds_584[2] : bf16 from vector<4xbf16>
        %lds_584_e3 = vector.extract %lds_584[3] : bf16 from vector<4xbf16>
        memref.store %lds_584_e0, %lds_ep[%lds_rb2c5] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_584_i1 = arith.addi %lds_rb2c5, %lds_c256 overflow<nsw> : index
        memref.store %lds_584_e1, %lds_ep[%lds_584_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_584_i2 = arith.addi %lds_rb2c5, %lds_c512 overflow<nsw> : index
        memref.store %lds_584_e2, %lds_ep[%lds_584_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_584_i3 = arith.addi %lds_rb2c5, %lds_c768 overflow<nsw> : index
        memref.store %lds_584_e3, %lds_ep[%lds_584_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=2, col=6) → acc %586
        %lds_586    = arith.truncf %586 : vector<4xf32> to vector<4xbf16>
        %lds_586_e0 = vector.extract %lds_586[0] : bf16 from vector<4xbf16>
        %lds_586_e1 = vector.extract %lds_586[1] : bf16 from vector<4xbf16>
        %lds_586_e2 = vector.extract %lds_586[2] : bf16 from vector<4xbf16>
        %lds_586_e3 = vector.extract %lds_586[3] : bf16 from vector<4xbf16>
        memref.store %lds_586_e0, %lds_ep[%lds_rb2c6] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_586_i1 = arith.addi %lds_rb2c6, %lds_c256 overflow<nsw> : index
        memref.store %lds_586_e1, %lds_ep[%lds_586_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_586_i2 = arith.addi %lds_rb2c6, %lds_c512 overflow<nsw> : index
        memref.store %lds_586_e2, %lds_ep[%lds_586_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_586_i3 = arith.addi %lds_rb2c6, %lds_c768 overflow<nsw> : index
        memref.store %lds_586_e3, %lds_ep[%lds_586_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=2, col=7) → acc %588
        %lds_588    = arith.truncf %588 : vector<4xf32> to vector<4xbf16>
        %lds_588_e0 = vector.extract %lds_588[0] : bf16 from vector<4xbf16>
        %lds_588_e1 = vector.extract %lds_588[1] : bf16 from vector<4xbf16>
        %lds_588_e2 = vector.extract %lds_588[2] : bf16 from vector<4xbf16>
        %lds_588_e3 = vector.extract %lds_588[3] : bf16 from vector<4xbf16>
        memref.store %lds_588_e0, %lds_ep[%lds_rb2c7] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_588_i1 = arith.addi %lds_rb2c7, %lds_c256 overflow<nsw> : index
        memref.store %lds_588_e1, %lds_ep[%lds_588_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_588_i2 = arith.addi %lds_rb2c7, %lds_c512 overflow<nsw> : index
        memref.store %lds_588_e2, %lds_ep[%lds_588_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_588_i3 = arith.addi %lds_rb2c7, %lds_c768 overflow<nsw> : index
        memref.store %lds_588_e3, %lds_ep[%lds_588_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=3, col=0) → acc %590
        %lds_590    = arith.truncf %590 : vector<4xf32> to vector<4xbf16>
        %lds_590_e0 = vector.extract %lds_590[0] : bf16 from vector<4xbf16>
        %lds_590_e1 = vector.extract %lds_590[1] : bf16 from vector<4xbf16>
        %lds_590_e2 = vector.extract %lds_590[2] : bf16 from vector<4xbf16>
        %lds_590_e3 = vector.extract %lds_590[3] : bf16 from vector<4xbf16>
        memref.store %lds_590_e0, %lds_ep[%lds_rb3c0] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_590_i1 = arith.addi %lds_rb3c0, %lds_c256 overflow<nsw> : index
        memref.store %lds_590_e1, %lds_ep[%lds_590_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_590_i2 = arith.addi %lds_rb3c0, %lds_c512 overflow<nsw> : index
        memref.store %lds_590_e2, %lds_ep[%lds_590_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_590_i3 = arith.addi %lds_rb3c0, %lds_c768 overflow<nsw> : index
        memref.store %lds_590_e3, %lds_ep[%lds_590_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=3, col=1) → acc %592
        %lds_592    = arith.truncf %592 : vector<4xf32> to vector<4xbf16>
        %lds_592_e0 = vector.extract %lds_592[0] : bf16 from vector<4xbf16>
        %lds_592_e1 = vector.extract %lds_592[1] : bf16 from vector<4xbf16>
        %lds_592_e2 = vector.extract %lds_592[2] : bf16 from vector<4xbf16>
        %lds_592_e3 = vector.extract %lds_592[3] : bf16 from vector<4xbf16>
        memref.store %lds_592_e0, %lds_ep[%lds_rb3c1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_592_i1 = arith.addi %lds_rb3c1, %lds_c256 overflow<nsw> : index
        memref.store %lds_592_e1, %lds_ep[%lds_592_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_592_i2 = arith.addi %lds_rb3c1, %lds_c512 overflow<nsw> : index
        memref.store %lds_592_e2, %lds_ep[%lds_592_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_592_i3 = arith.addi %lds_rb3c1, %lds_c768 overflow<nsw> : index
        memref.store %lds_592_e3, %lds_ep[%lds_592_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=3, col=2) → acc %594
        %lds_594    = arith.truncf %594 : vector<4xf32> to vector<4xbf16>
        %lds_594_e0 = vector.extract %lds_594[0] : bf16 from vector<4xbf16>
        %lds_594_e1 = vector.extract %lds_594[1] : bf16 from vector<4xbf16>
        %lds_594_e2 = vector.extract %lds_594[2] : bf16 from vector<4xbf16>
        %lds_594_e3 = vector.extract %lds_594[3] : bf16 from vector<4xbf16>
        memref.store %lds_594_e0, %lds_ep[%lds_rb3c2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_594_i1 = arith.addi %lds_rb3c2, %lds_c256 overflow<nsw> : index
        memref.store %lds_594_e1, %lds_ep[%lds_594_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_594_i2 = arith.addi %lds_rb3c2, %lds_c512 overflow<nsw> : index
        memref.store %lds_594_e2, %lds_ep[%lds_594_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_594_i3 = arith.addi %lds_rb3c2, %lds_c768 overflow<nsw> : index
        memref.store %lds_594_e3, %lds_ep[%lds_594_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=3, col=3) → acc %596
        %lds_596    = arith.truncf %596 : vector<4xf32> to vector<4xbf16>
        %lds_596_e0 = vector.extract %lds_596[0] : bf16 from vector<4xbf16>
        %lds_596_e1 = vector.extract %lds_596[1] : bf16 from vector<4xbf16>
        %lds_596_e2 = vector.extract %lds_596[2] : bf16 from vector<4xbf16>
        %lds_596_e3 = vector.extract %lds_596[3] : bf16 from vector<4xbf16>
        memref.store %lds_596_e0, %lds_ep[%lds_rb3c3] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_596_i1 = arith.addi %lds_rb3c3, %lds_c256 overflow<nsw> : index
        memref.store %lds_596_e1, %lds_ep[%lds_596_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_596_i2 = arith.addi %lds_rb3c3, %lds_c512 overflow<nsw> : index
        memref.store %lds_596_e2, %lds_ep[%lds_596_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_596_i3 = arith.addi %lds_rb3c3, %lds_c768 overflow<nsw> : index
        memref.store %lds_596_e3, %lds_ep[%lds_596_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=3, col=4) → acc %598
        %lds_598    = arith.truncf %598 : vector<4xf32> to vector<4xbf16>
        %lds_598_e0 = vector.extract %lds_598[0] : bf16 from vector<4xbf16>
        %lds_598_e1 = vector.extract %lds_598[1] : bf16 from vector<4xbf16>
        %lds_598_e2 = vector.extract %lds_598[2] : bf16 from vector<4xbf16>
        %lds_598_e3 = vector.extract %lds_598[3] : bf16 from vector<4xbf16>
        memref.store %lds_598_e0, %lds_ep[%lds_rb3c4] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_598_i1 = arith.addi %lds_rb3c4, %lds_c256 overflow<nsw> : index
        memref.store %lds_598_e1, %lds_ep[%lds_598_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_598_i2 = arith.addi %lds_rb3c4, %lds_c512 overflow<nsw> : index
        memref.store %lds_598_e2, %lds_ep[%lds_598_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_598_i3 = arith.addi %lds_rb3c4, %lds_c768 overflow<nsw> : index
        memref.store %lds_598_e3, %lds_ep[%lds_598_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=3, col=5) → acc %600
        %lds_600    = arith.truncf %600 : vector<4xf32> to vector<4xbf16>
        %lds_600_e0 = vector.extract %lds_600[0] : bf16 from vector<4xbf16>
        %lds_600_e1 = vector.extract %lds_600[1] : bf16 from vector<4xbf16>
        %lds_600_e2 = vector.extract %lds_600[2] : bf16 from vector<4xbf16>
        %lds_600_e3 = vector.extract %lds_600[3] : bf16 from vector<4xbf16>
        memref.store %lds_600_e0, %lds_ep[%lds_rb3c5] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_600_i1 = arith.addi %lds_rb3c5, %lds_c256 overflow<nsw> : index
        memref.store %lds_600_e1, %lds_ep[%lds_600_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_600_i2 = arith.addi %lds_rb3c5, %lds_c512 overflow<nsw> : index
        memref.store %lds_600_e2, %lds_ep[%lds_600_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_600_i3 = arith.addi %lds_rb3c5, %lds_c768 overflow<nsw> : index
        memref.store %lds_600_e3, %lds_ep[%lds_600_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=3, col=6) → acc %602
        %lds_602    = arith.truncf %602 : vector<4xf32> to vector<4xbf16>
        %lds_602_e0 = vector.extract %lds_602[0] : bf16 from vector<4xbf16>
        %lds_602_e1 = vector.extract %lds_602[1] : bf16 from vector<4xbf16>
        %lds_602_e2 = vector.extract %lds_602[2] : bf16 from vector<4xbf16>
        %lds_602_e3 = vector.extract %lds_602[3] : bf16 from vector<4xbf16>
        memref.store %lds_602_e0, %lds_ep[%lds_rb3c6] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_602_i1 = arith.addi %lds_rb3c6, %lds_c256 overflow<nsw> : index
        memref.store %lds_602_e1, %lds_ep[%lds_602_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_602_i2 = arith.addi %lds_rb3c6, %lds_c512 overflow<nsw> : index
        memref.store %lds_602_e2, %lds_ep[%lds_602_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_602_i3 = arith.addi %lds_rb3c6, %lds_c768 overflow<nsw> : index
        memref.store %lds_602_e3, %lds_ep[%lds_602_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // Group (rb=3, col=7) → acc %604
        %lds_604    = arith.truncf %604 : vector<4xf32> to vector<4xbf16>
        %lds_604_e0 = vector.extract %lds_604[0] : bf16 from vector<4xbf16>
        %lds_604_e1 = vector.extract %lds_604[1] : bf16 from vector<4xbf16>
        %lds_604_e2 = vector.extract %lds_604[2] : bf16 from vector<4xbf16>
        %lds_604_e3 = vector.extract %lds_604[3] : bf16 from vector<4xbf16>
        memref.store %lds_604_e0, %lds_ep[%lds_rb3c7] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_604_i1 = arith.addi %lds_rb3c7, %lds_c256 overflow<nsw> : index
        memref.store %lds_604_e1, %lds_ep[%lds_604_i1] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_604_i2 = arith.addi %lds_rb3c7, %lds_c512 overflow<nsw> : index
        memref.store %lds_604_e2, %lds_ep[%lds_604_i2] : memref<65536xbf16, #gpu.address_space<workgroup>>
        %lds_604_i3 = arith.addi %lds_rb3c7, %lds_c768 overflow<nsw> : index
        memref.store %lds_604_e3, %lds_ep[%lds_604_i3] : memref<65536xbf16, #gpu.address_space<workgroup>>

        // ── Step 2: Workgroup barrier (ensure all LDS writes are visible) ─────
        gpu.barrier

        // ── Global output buffer setup ────────────────────────────────────────
        %ep_block_x = affine.apply #map61()[%block_id_x]
        %ep_block_y = affine.apply #map61()[%block_id_y]
        %ep_base_buf, %ep_offset, %ep_sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<2048x2048xbf16, strided<[?, 1]>> -> memref<bf16>, index, index, index, index, index
        %ep_block_row_off = arith.muli %ep_block_x, %strides#0 overflow<nsw> : index
        %ep_base_off = arith.addi %ep_block_row_off, %ep_block_y overflow<nsw> : index
        %ep_rcast = memref.reinterpret_cast %4 to offset: [%ep_base_off], sizes: [1073741822], strides: [1] : memref<bf16> to memref<1073741822xbf16, strided<[1], offset: ?>>
        %ep_cast  = memref.cast %ep_rcast : memref<1073741822xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
        %ep_stride_i14 = arith.index_cast %strides#0 : index to i14
        %ep_buffer = amdgpu.fat_raw_buffer_cast %ep_cast validBytes(%c2147483645_i64) cacheSwizzleStride(%ep_stride_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>

        // ── Step 3: Linear LDS read → buffer_store_dwordx4 to global ─────────
        // global_tid g = tid_y*256 + tid_x  (0..511)
        // row_base  = g / 2             (output local row, 0..255)
        // col_base  = (g % 2) * 128     (0 or 128; g%2 == tid_x%2)
        // lds_g_base = g * 128          (LDS read start index for this thread)
        // Each chunk k (0..15): read LDS[lds_g_base+k*8], write global at col_base+k*8.
        %lds_c2      = arith.constant    2 : index

        %lds_tidy256 = arith.muli %thread_id_y, %lds_c256  overflow<nsw> : index
        %lds_g       = arith.addi %lds_tidy256, %thread_id_x overflow<nsw> : index
        %lds_g128    = arith.muli %lds_g, %lds_c128 overflow<nsw> : index
        %lds_row_base = arith.divui %lds_g, %lds_c2 : index
        %lds_g_mod2  = arith.remui %thread_id_x, %lds_c2 : index
        %lds_col_base = arith.muli %lds_g_mod2, %lds_c128 overflow<nsw> : index
        %lds_row_off  = arith.muli %lds_row_base, %strides#0 overflow<nsw> : index
        %lds_gw_base  = arith.addi %lds_row_off, %lds_col_base overflow<nsw> : index

        // chunk 0: LDS[g*128+0 .. +7] → global col_base+0
        %lds_rd00 = vector.load %lds_ep[%lds_g128] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        vector.store %lds_rd00, %ep_buffer[%lds_gw_base] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 1: LDS[g*128+8 .. +7] → global col_base+8
        %lds_g128_k1  = arith.addi %lds_g128,   %lds_c8   overflow<nsw> : index
        %lds_rd01     = vector.load %lds_ep[%lds_g128_k1] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k1    = arith.addi %lds_gw_base, %lds_c8  overflow<nsw> : index
        vector.store %lds_rd01, %ep_buffer[%lds_gw_k1] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 2: LDS[g*128+16 .. +7] → global col_base+16
        %lds_c16_x2   = arith.constant  16 : index
        %lds_g128_k2  = arith.addi %lds_g128,    %lds_c16_x2  overflow<nsw> : index
        %lds_rd02     = vector.load %lds_ep[%lds_g128_k2] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k2    = arith.addi %lds_gw_base, %lds_c16_x2  overflow<nsw> : index
        vector.store %lds_rd02, %ep_buffer[%lds_gw_k2] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 3: LDS[g*128+24 .. +7] → global col_base+24
        %lds_c24      = arith.constant  24 : index
        %lds_g128_k3  = arith.addi %lds_g128,    %lds_c24  overflow<nsw> : index
        %lds_rd03     = vector.load %lds_ep[%lds_g128_k3] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k3    = arith.addi %lds_gw_base, %lds_c24  overflow<nsw> : index
        vector.store %lds_rd03, %ep_buffer[%lds_gw_k3] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 4: LDS[g*128+32 .. +7] → global col_base+32
        %lds_c32      = arith.constant  32 : index
        %lds_g128_k4  = arith.addi %lds_g128,    %lds_c32  overflow<nsw> : index
        %lds_rd04     = vector.load %lds_ep[%lds_g128_k4] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k4    = arith.addi %lds_gw_base, %lds_c32  overflow<nsw> : index
        vector.store %lds_rd04, %ep_buffer[%lds_gw_k4] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 5: LDS[g*128+40 .. +7] → global col_base+40
        %lds_c40      = arith.constant  40 : index
        %lds_g128_k5  = arith.addi %lds_g128,    %lds_c40  overflow<nsw> : index
        %lds_rd05     = vector.load %lds_ep[%lds_g128_k5] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k5    = arith.addi %lds_gw_base, %lds_c40  overflow<nsw> : index
        vector.store %lds_rd05, %ep_buffer[%lds_gw_k5] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 6: LDS[g*128+48 .. +7] → global col_base+48
        %lds_c48      = arith.constant  48 : index
        %lds_g128_k6  = arith.addi %lds_g128,    %lds_c48  overflow<nsw> : index
        %lds_rd06     = vector.load %lds_ep[%lds_g128_k6] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k6    = arith.addi %lds_gw_base, %lds_c48  overflow<nsw> : index
        vector.store %lds_rd06, %ep_buffer[%lds_gw_k6] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 7: LDS[g*128+56 .. +7] → global col_base+56
        %lds_c56      = arith.constant  56 : index
        %lds_g128_k7  = arith.addi %lds_g128,    %lds_c56  overflow<nsw> : index
        %lds_rd07     = vector.load %lds_ep[%lds_g128_k7] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k7    = arith.addi %lds_gw_base, %lds_c56  overflow<nsw> : index
        vector.store %lds_rd07, %ep_buffer[%lds_gw_k7] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 8: LDS[g*128+64 .. +7] → global col_base+64
        %lds_c64b     = arith.constant  64 : index
        %lds_g128_k8  = arith.addi %lds_g128,    %lds_c64b overflow<nsw> : index
        %lds_rd08     = vector.load %lds_ep[%lds_g128_k8] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k8    = arith.addi %lds_gw_base, %lds_c64b overflow<nsw> : index
        vector.store %lds_rd08, %ep_buffer[%lds_gw_k8] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 9: LDS[g*128+72 .. +7] → global col_base+72
        %lds_c72      = arith.constant  72 : index
        %lds_g128_k9  = arith.addi %lds_g128,    %lds_c72  overflow<nsw> : index
        %lds_rd09     = vector.load %lds_ep[%lds_g128_k9] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k9    = arith.addi %lds_gw_base, %lds_c72  overflow<nsw> : index
        vector.store %lds_rd09, %ep_buffer[%lds_gw_k9] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 10: LDS[g*128+80 .. +7] → global col_base+80
        %lds_c80      = arith.constant  80 : index
        %lds_g128_k10 = arith.addi %lds_g128,    %lds_c80  overflow<nsw> : index
        %lds_rd10     = vector.load %lds_ep[%lds_g128_k10] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k10   = arith.addi %lds_gw_base, %lds_c80  overflow<nsw> : index
        vector.store %lds_rd10, %ep_buffer[%lds_gw_k10] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 11: LDS[g*128+88 .. +7] → global col_base+88
        %lds_c88      = arith.constant  88 : index
        %lds_g128_k11 = arith.addi %lds_g128,    %lds_c88  overflow<nsw> : index
        %lds_rd11     = vector.load %lds_ep[%lds_g128_k11] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k11   = arith.addi %lds_gw_base, %lds_c88  overflow<nsw> : index
        vector.store %lds_rd11, %ep_buffer[%lds_gw_k11] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 12: LDS[g*128+96 .. +7] → global col_base+96
        %lds_c96      = arith.constant  96 : index
        %lds_g128_k12 = arith.addi %lds_g128,    %lds_c96  overflow<nsw> : index
        %lds_rd12     = vector.load %lds_ep[%lds_g128_k12] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k12   = arith.addi %lds_gw_base, %lds_c96  overflow<nsw> : index
        vector.store %lds_rd12, %ep_buffer[%lds_gw_k12] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 13: LDS[g*128+104 .. +7] → global col_base+104
        %lds_c104     = arith.constant 104 : index
        %lds_g128_k13 = arith.addi %lds_g128,    %lds_c104 overflow<nsw> : index
        %lds_rd13     = vector.load %lds_ep[%lds_g128_k13] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k13   = arith.addi %lds_gw_base, %lds_c104 overflow<nsw> : index
        vector.store %lds_rd13, %ep_buffer[%lds_gw_k13] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 14: LDS[g*128+112 .. +7] → global col_base+112
        %lds_c112     = arith.constant 112 : index
        %lds_g128_k14 = arith.addi %lds_g128,    %lds_c112 overflow<nsw> : index
        %lds_rd14     = vector.load %lds_ep[%lds_g128_k14] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k14   = arith.addi %lds_gw_base, %lds_c112 overflow<nsw> : index
        vector.store %lds_rd14, %ep_buffer[%lds_gw_k14] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

        // chunk 15: LDS[g*128+120 .. +7] → global col_base+120
        %lds_c120     = arith.constant 120 : index
        %lds_g128_k15 = arith.addi %lds_g128,    %lds_c120 overflow<nsw> : index
        %lds_rd15     = vector.load %lds_ep[%lds_g128_k15] : memref<65536xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
        %lds_gw_k15   = arith.addi %lds_gw_base, %lds_c120 overflow<nsw> : index
        vector.store %lds_rd15, %ep_buffer[%lds_gw_k15] {alignment = 16 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>

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
