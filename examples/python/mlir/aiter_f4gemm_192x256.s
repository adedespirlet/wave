
/sgl-workspace/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256.co:	file format elf64-amdgpu

Disassembly of section .text:

0000000000002c00 <_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256E>:
	s_and_b32 s1, s1, 0xffff                                   // 000000002C00: 8601FF01 0000FFFF
	s_load_dwordx2 s[4:5], s[0:1], 0x0                         // 000000002C08: C0060100 00000000
	s_load_dwordx2 s[8:9], s[0:1], 0x10                        // 000000002C10: C0060200 00000010
	s_load_dwordx2 s[12:13], s[0:1], 0x20                      // 000000002C18: C0060300 00000020
	s_load_dwordx2 s[16:17], s[0:1], 0x30                      // 000000002C20: C0060400 00000030
	s_load_dword s41, s[0:1], 0x40                             // 000000002C28: C0020A40 00000040
	s_load_dword s42, s[0:1], 0x50                             // 000000002C30: C0020A80 00000050
	s_load_dword s36, s[0:1], 0x80                             // 000000002C38: C0020900 00000080
	s_load_dword s37, s[0:1], 0xa0                             // 000000002C40: C0020940 000000A0
	s_load_dword s38, s[0:1], 0xc0                             // 000000002C48: C0020980 000000C0
	s_load_dword s43, s[0:1], 0xe0                             // 000000002C50: C0020AC0 000000E0
	s_load_dword s44, s[0:1], 0xf0                             // 000000002C58: C0020B00 000000F0
	s_load_dword s45, s[0:1], 0x100                            // 000000002C60: C0020B40 00000100
	s_load_dwordx2 s[20:21], s[0:1], 0x110                     // 000000002C68: C0060500 00000110
	s_load_dwordx2 s[24:25], s[0:1], 0x120                     // 000000002C70: C0060600 00000120
	s_load_dword s39, s[0:1], 0x130                            // 000000002C78: C00209C0 00000130
	s_load_dword s40, s[0:1], 0x150                            // 000000002C80: C0020A00 00000150
	v_lshrrev_b32_e32 v1, 10, v0                               // 000000002C88: 2002008A
	v_lshrrev_b32_e32 v2, 10, v1                               // 000000002C8C: 2004028A
	v_and_b32_e32 v2, 0x3ff, v2                                // 000000002C90: 260404FF 000003FF
	v_and_b32_e32 v1, 0x3ff, v1                                // 000000002C98: 260202FF 000003FF
	v_and_b32_e32 v0, 0x3ff, v0                                // 000000002CA0: 260000FF 000003FF
	v_lshrrev_b32_e32 v3, 6, v0                                // 000000002CA8: 20060086
	v_and_b32_e32 v0, 63, v0                                   // 000000002CAC: 260000BF
	v_readfirstlane_b32 s46, v3                                // 000000002CB0: 7E5C0503
	s_waitcnt lgkmcnt(0)                                       // 000000002CB4: BF8CC07F
	s_mul_i32 s63, 0xc0, 8                                     // 000000002CB8: 923F88FF 000000C0
	v_cvt_f32_u32_e32 v4, s63                                  // 000000002CC0: 7E080C3F
	s_sub_i32 s62, 0, s63                                      // 000000002CC4: 81BE3F80
	v_rcp_iflag_f32_e32 v4, v4                                 // 000000002CC8: 7E084704
	s_nop 0                                                    // 000000002CCC: BF800000
	v_mul_f32_e32 v4, 0x4f7ffffe, v4                           // 000000002CD0: 0A0808FF 4F7FFFFE
	v_cvt_u32_f32_e32 v4, v4                                   // 000000002CD8: 7E080F04
	v_mul_lo_u32 v5, s62, v4                                   // 000000002CDC: D2850005 0002083E
	v_mul_hi_u32 v5, v4, v5                                    // 000000002CE4: D2860005 00020B04
	v_add_u32_e32 v4, v4, v5                                   // 000000002CEC: 68080B04
	v_mul_hi_u32 v4, s43, v4                                   // 000000002CF0: D2860004 0002082B
	v_mul_lo_u32 v5, v4, s63                                   // 000000002CF8: D2850005 00007F04
	v_sub_u32_e32 v7, s43, v5                                  // 000000002D00: 6A0E0A2B
	v_add_u32_e32 v6, 1, v4                                    // 000000002D04: 680C0881
	v_cmp_le_u32_e32 vcc, s63, v7                              // 000000002D08: 7D960E3F
	v_subrev_u32_e32 v5, s63, v7                               // 000000002D0C: 6C0A0E3F
	s_nop 0                                                    // 000000002D10: BF800000
	v_cndmask_b32_e32 v4, v4, v6, vcc                          // 000000002D14: 00080D04
	v_cndmask_b32_e32 v7, v7, v5, vcc                          // 000000002D18: 000E0B07
	v_add_u32_e32 v5, 1, v4                                    // 000000002D1C: 680A0881
	v_cmp_le_u32_e32 vcc, s63, v7                              // 000000002D20: 7D960E3F
	s_nop 1                                                    // 000000002D24: BF800001
	v_cndmask_b32_e32 v7, v4, v5, vcc                          // 000000002D28: 000E0B04
	s_nop 3                                                    // 000000002D2C: BF800003
	v_readfirstlane_b32 s62, v7                                // 000000002D30: 7E7C0507
	s_nop 3                                                    // 000000002D34: BF800003
	s_lshl_b32 s62, s62, 3                                     // 000000002D38: 8E3E833E
	s_cmp_lt_i32 s3, s62                                       // 000000002D3C: BF043E03
	s_cbranch_scc0 label_0072                                  // 000000002D40: BF840021
	s_add_u32 s49, s44, 0xff                                   // 000000002D44: 8031FF2C 000000FF
	s_lshr_b32 s48, s49, 8                                     // 000000002D4C: 8F308831
	s_mul_i32 s49, s48, s3                                     // 000000002D50: 92310330
	s_add_i32 s49, s49, s2                                     // 000000002D54: 81310231
	s_lshr_b32 s63, s44, 13                                    // 000000002D58: 8F3F8D2C
	s_lshl_b32 s47, s63, 5                                     // 000000002D5C: 8E2F853F
	s_mul_i32 s62, s62, s47                                    // 000000002D60: 923E2F3E
	s_cmp_lt_i32 s49, s62                                      // 000000002D64: BF043E31
	s_cbranch_scc0 label_0068                                  // 000000002D68: BF84000D
	s_and_b32 s62, s49, 0xff                                   // 000000002D6C: 863EFF31 000000FF
	s_and_b32 s47, s62, 31                                     // 000000002D74: 862F9F3E
	s_lshr_b32 s48, s62, 5                                     // 000000002D78: 8F30853E
	s_lshr_b32 s49, s49, 8                                     // 000000002D7C: 8F318831

0000000000002d80 <label_0060>:
	s_cmp_lt_i32 s49, s63                                      // 000000002D80: BF043F31
	s_cbranch_scc1 label_0065                                  // 000000002D84: BF850003
	s_sub_i32 s49, s49, s63                                    // 000000002D88: 81B13F31
	s_add_i32 s48, s48, 8                                      // 000000002D8C: 81308830
	s_branch label_0060                                        // 000000002D90: BF82FFFB

0000000000002d94 <label_0065>:
	s_mul_i32 s49, s49, 32                                     // 000000002D94: 9231A031
	s_add_i32 s47, s47, s49                                    // 000000002D98: 812F312F
	s_branch label_0074                                        // 000000002D9C: BF82000C

0000000000002da0 <label_0068>:
	s_sub_i32 s49, s49, s62                                    // 000000002DA0: 81B13E31
	s_sub_i32 s63, s48, s47                                    // 000000002DA4: 81BF2F30
	s_mov_b32 s48, 0                                           // 000000002DA8: BEB00080

0000000000002dac <label_006B>:
	s_cmp_lt_i32 s49, s63                                      // 000000002DAC: BF043F31
	s_cbranch_scc1 label_0070                                  // 000000002DB0: BF850003
	s_sub_i32 s49, s49, s63                                    // 000000002DB4: 81B13F31
	s_add_i32 s48, s48, 1                                      // 000000002DB8: 81308130
	s_branch label_006B                                        // 000000002DBC: BF82FFFB

0000000000002dc0 <label_0070>:
	s_add_i32 s47, s47, s49                                    // 000000002DC0: 812F312F
	s_branch label_0074                                        // 000000002DC4: BF820002

0000000000002dc8 <label_0072>:
	s_mov_b32 s47, s2                                          // 000000002DC8: BEAF0002
	s_mov_b32 s48, s3                                          // 000000002DCC: BEB00003

0000000000002dd0 <label_0074>:
	s_lshr_b32 s37, s37, 1                                     // 000000002DD0: 8F258125
	s_mul_i32 s62, s48, 0xc0                                   // 000000002DD4: 923EFF30 000000C0
	s_mul_hi_u32 s63, s37, s62                                 // 000000002DDC: 963F3E25
	s_add_u32 s13, s13, s63                                    // 000000002DE0: 800D3F0D
	s_mul_i32 s63, s37, s62                                    // 000000002DE4: 923F3E25
	s_add_u32 s12, s12, s63                                    // 000000002DE8: 800C3F0C
	s_addc_u32 s13, s13, 0                                     // 000000002DEC: 820D800D
	s_sub_i32 s63, s43, s62                                    // 000000002DF0: 81BF3E2B
	s_cmp_lt_u32 s63, 0xc0                                     // 000000002DF4: BF0AFF3F 000000C0
	s_cselect_b32 s62, s63, 0xc0                               // 000000002DFC: 853EFF3F 000000C0
	s_mul_i32 s14, s37, s62                                    // 000000002E04: 920E3E25
	s_mov_b32 s15, 0x20000                                     // 000000002E08: BE8F00FF 00020000
	v_lshrrev_b32_e32 v4, 3, v0                                // 000000002E10: 20080083
	v_lshrrev_b32_e32 v5, 2, v4                                // 000000002E14: 200A0882
	v_lshlrev_b32_e32 v5, 4, v5                                // 000000002E18: 240A0A84
	v_and_b32_e32 v4, 3, v4                                    // 000000002E1C: 26080883
	v_lshrrev_b32_e32 v6, 1, v4                                // 000000002E20: 200C0881
	v_lshlrev_b32_e32 v6, 2, v6                                // 000000002E24: 240C0C82
	v_add_u32_e32 v5, v5, v6                                   // 000000002E28: 680A0D05
	v_and_b32_e32 v4, 1, v4                                    // 000000002E2C: 26080881
	v_add_u32_e32 v5, v5, v4                                   // 000000002E30: 680A0905
	v_mul_lo_u32 v178, s37, v5                                 // 000000002E34: D28500B2 00020A25
	v_and_b32_e32 v4, 7, v0                                    // 000000002E3C: 26080087
	v_lshlrev_b32_e32 v4, 4, v4                                // 000000002E40: 24080884
	v_add_u32_e32 v178, v4, v178                               // 000000002E44: 69656504
	s_lshr_b32 s62, s46, 1                                     // 000000002E48: 8F3E812E
	s_mul_i32 s62, s62, 8                                      // 000000002E4C: 923E883E
	s_and_b32 s63, s46, 1                                      // 000000002E50: 863F812E
	s_mul_i32 s63, s63, 2                                      // 000000002E54: 923F823F
	s_add_u32 s62, s62, s63                                    // 000000002E58: 803E3F3E
	s_mul_i32 s62, s37, s62                                    // 000000002E5C: 923E3E25
	v_add_u32_e32 v178, s62, v178                              // 000000002E60: 6965643E
	s_mul_i32 s62, s37, 32                                     // 000000002E64: 923EA025
	v_add_u32_e32 v179, s62, v178                              // 000000002E68: 6967643E
	v_add_u32_e32 v180, s62, v179                              // 000000002E6C: 6969663E
	v_add_u32_e32 v181, s62, v180                              // 000000002E70: 696B683E
	v_add_u32_e32 v182, s62, v181                              // 000000002E74: 696D6A3E
	v_add_u32_e32 v183, s62, v182                              // 000000002E78: 696F6C3E
	s_mul_i32 s64, 0x420, s46                                  // 000000002E7C: 92402EFF 00000420
	s_add_u32 s64, 0x1000, s64                                 // 000000002E84: 804040FF 00001000
	v_and_b32_e32 v4, 15, v0                                   // 000000002E8C: 2608008F
	v_lshrrev_b32_e32 v5, 3, v4                                // 000000002E90: 200A0883
	v_mul_i32_i24_e32 v5, 2, v5                                // 000000002E94: 0C0A0A82
	v_and_b32_e32 v4, 3, v0                                    // 000000002E98: 26080083
	v_lshrrev_b32_e32 v6, 1, v4                                // 000000002E9C: 200C0881
	v_add_u32_e32 v4, v5, v6                                   // 000000002EA0: 68080D05
	v_mul_i32_i24_e32 v184, 0x420, v4                          // 000000002EA4: 0D7008FF 00000420
	v_and_b32_e32 v4, 7, v0                                    // 000000002EAC: 26080087
	v_lshrrev_b32_e32 v5, 2, v4                                // 000000002EB0: 200A0882
	v_mul_i32_i24_e32 v5, 0x100, v5                            // 000000002EB4: 0C0A0AFF 00000100
	v_add_u32_e32 v184, v5, v184                               // 000000002EBC: 69717105
	v_and_b32_e32 v4, 1, v0                                    // 000000002EC0: 26080081
	v_mul_i32_i24_e32 v6, 0x80, v4                             // 000000002EC4: 0C0C08FF 00000080
	v_add_u32_e32 v184, v6, v184                               // 000000002ECC: 69717106
	v_lshrrev_b32_e32 v4, 4, v0                                // 000000002ED0: 20080084
	v_mul_i32_i24_e32 v4, 16, v4                               // 000000002ED4: 0C080890
	v_add_u32_e32 v184, v4, v184                               // 000000002ED8: 69717104
	v_add_u32_e32 v184, 0x1000, v184                           // 000000002EDC: 697170FF 00001000
	v_add_u32_e32 v185, 0x6300, v184                           // 000000002EE4: 697370FF 00006300
	s_mul_i32 s62, s48, 0xc0                                   // 000000002EEC: 923EFF30 000000C0
	s_mul_hi_u32 s63, s39, s62                                 // 000000002EF4: 963F3E27
	s_add_u32 s21, s21, s63                                    // 000000002EF8: 80153F15
	s_mul_i32 s63, s39, s62                                    // 000000002EFC: 923F3E27
	s_add_u32 s20, s20, s63                                    // 000000002F00: 80143F14
	s_addc_u32 s21, s21, 0                                     // 000000002F04: 82158015
	s_add_u32 s63, s43, 31                                     // 000000002F08: 803F9F2B
	s_lshr_b32 s63, s63, 5                                     // 000000002F0C: 8F3F853F
	s_lshl_b32 s63, s63, 5                                     // 000000002F10: 8E3F853F
	s_sub_i32 s63, s63, s62                                    // 000000002F14: 81BF3E3F
	s_cmp_lt_u32 s63, 0xc0                                     // 000000002F18: BF0AFF3F 000000C0
	s_cselect_b32 s62, s63, 0xc0                               // 000000002F20: 853EFF3F 000000C0
	s_mul_i32 s22, s39, s62                                    // 000000002F28: 92163E27
	s_mov_b32 s23, 0x20000                                     // 000000002F2C: BE9700FF 00020000
	v_lshlrev_b32_e32 v186, 2, v0                              // 000000002F34: 25740082
	s_mul_i32 s63, s46, 32                                     // 000000002F38: 923FA02E
	s_mul_i32 s63, s63, s39                                    // 000000002F3C: 923F273F
	v_add_u32_e32 v186, s63, v186                              // 000000002F40: 6975743F
	s_mul_i32 s63, 0x80, s39                                   // 000000002F44: 923F27FF 00000080
	v_add_u32_e32 v187, s63, v186                              // 000000002F4C: 6977743F
	s_mul_i32 s65, s46, 0x100                                  // 000000002F50: 9241FF2E 00000100
	s_add_i32 s65, s65, 0                                      // 000000002F58: 81418041
	v_lshlrev_b32_e32 v188, 2, v0                              // 000000002F5C: 25780082
	v_add_u32_e32 v188, 0, v188                                // 000000002F60: 69797880
	s_lshr_b32 s38, s38, 1                                     // 000000002F64: 8F268126
	s_mul_i32 s62, s47, 0x100                                  // 000000002F68: 923EFF2F 00000100
	s_mul_hi_u32 s63, s38, s62                                 // 000000002F70: 963F3E26
	s_add_u32 s17, s17, s63                                    // 000000002F74: 80113F11
	s_mul_i32 s63, s38, s62                                    // 000000002F78: 923F3E26
	s_add_u32 s16, s16, s63                                    // 000000002F7C: 80103F10
	s_addc_u32 s17, s17, 0                                     // 000000002F80: 82118011
	s_sub_i32 s63, s44, s62                                    // 000000002F84: 81BF3E2C
	s_cmp_lt_u32 s63, 0x100                                    // 000000002F88: BF0AFF3F 00000100
	s_cselect_b32 s62, s63, 0x100                              // 000000002F90: 853EFF3F 00000100
	s_mul_i32 s18, s38, s62                                    // 000000002F98: 92123E26
	s_mov_b32 s19, 0x20000                                     // 000000002F9C: BE9300FF 00020000
	v_lshlrev_b32_e32 v189, 4, v0                              // 000000002FA4: 257A0084
	s_mul_i32 s63, s46, 64                                     // 000000002FA8: 923FC02E
	s_mul_i32 s62, s63, s38                                    // 000000002FAC: 923E263F
	v_add_u32_e32 v189, s62, v189                              // 000000002FB0: 697B7A3E
	s_mul_i32 s62, 16, s38                                     // 000000002FB4: 923E2690
	v_add_u32_e32 v190, s62, v189                              // 000000002FB8: 697D7A3E
	v_add_u32_e32 v191, s62, v190                              // 000000002FBC: 697F7C3E
	v_add_u32_e32 v192, s62, v191                              // 000000002FC0: 69817E3E
	s_mul_i32 s62, s47, 0x100                                  // 000000002FC4: 923EFF2F 00000100
	s_mul_hi_u32 s63, s40, s62                                 // 000000002FCC: 963F3E28
	s_add_u32 s25, s25, s63                                    // 000000002FD0: 80193F19
	s_mul_i32 s63, s40, s62                                    // 000000002FD4: 923F3E28
	s_add_u32 s24, s24, s63                                    // 000000002FD8: 80183F18
	s_addc_u32 s25, s25, 0                                     // 000000002FDC: 82198019
	s_sub_i32 s63, s44, s62                                    // 000000002FE0: 81BF3E2C
	s_cmp_lt_u32 s63, 0x100                                    // 000000002FE4: BF0AFF3F 00000100
	s_cselect_b32 s62, s63, 0x100                              // 000000002FEC: 853EFF3F 00000100
	s_mul_i32 s26, s40, s62                                    // 000000002FF4: 921A3E28
	s_mov_b32 s27, 0x20000                                     // 000000002FF8: BE9B00FF 00020000
	v_lshlrev_b32_e32 v193, 2, v0                              // 000000003000: 25820082
	s_mul_i32 s63, s46, 64                                     // 000000003004: 923FC02E
	s_mul_i32 s63, s63, s40                                    // 000000003008: 923F283F
	v_add_u32_e32 v193, s63, v193                              // 00000000300C: 6983823F
	s_mul_i32 s62, 32, s40                                     // 000000003010: 923E28A0
	v_add_u32_e32 v194, s62, v193                              // 000000003014: 6985823E
	s_mov_b32 s66, 0x80                                        // 000000003018: BEC200FF 00000080
	s_mov_b32 s67, 0x800                                       // 000000003020: BEC300FF 00000800
	s_mov_b32 s68, 0x100                                       // 000000003028: BEC400FF 00000100
	s_mov_b32 s69, 0x100                                       // 000000003030: BEC500FF 00000100
	s_mov_b32 s60, 0                                           // 000000003038: BEBC0080
	s_mov_b32 s61, s45                                         // 00000000303C: BEBD002D
	s_add_u32 m0, 0, s65                                       // 000000003040: 807C4180
	buffer_load_dword v186, s[20:23], 0 offen lds              // 000000003044: E0511000 800500BA
	v_accvgpr_write_b32 a0, 0                                  // 00000000304C: D3D94000 18000080
	v_accvgpr_write_b32 a1, 0                                  // 000000003054: D3D94001 18000080
	v_accvgpr_write_b32 a2, 0                                  // 00000000305C: D3D94002 18000080
	v_accvgpr_write_b32 a3, 0                                  // 000000003064: D3D94003 18000080
	v_accvgpr_write_b32 a4, 0                                  // 00000000306C: D3D94004 18000080
	v_accvgpr_write_b32 a5, 0                                  // 000000003074: D3D94005 18000080
	v_accvgpr_write_b32 a6, 0                                  // 00000000307C: D3D94006 18000080
	v_accvgpr_write_b32 a7, 0                                  // 000000003084: D3D94007 18000080
	s_add_u32 m0, 0x400, s65                                   // 00000000308C: 807C41FF 00000400
	buffer_load_dword v187, s[20:23], 0 offen lds              // 000000003094: E0511000 800500BB
	v_accvgpr_write_b32 a8, 0                                  // 00000000309C: D3D94008 18000080
	v_accvgpr_write_b32 a9, 0                                  // 0000000030A4: D3D94009 18000080
	v_accvgpr_write_b32 a10, 0                                 // 0000000030AC: D3D9400A 18000080
	v_accvgpr_write_b32 a11, 0                                 // 0000000030B4: D3D9400B 18000080
	v_accvgpr_write_b32 a12, 0                                 // 0000000030BC: D3D9400C 18000080
	v_accvgpr_write_b32 a13, 0                                 // 0000000030C4: D3D9400D 18000080
	v_accvgpr_write_b32 a14, 0                                 // 0000000030CC: D3D9400E 18000080
	v_accvgpr_write_b32 a15, 0                                 // 0000000030D4: D3D9400F 18000080
	s_add_u32 m0, 0, s64                                       // 0000000030DC: 807C4080
	buffer_load_dwordx4 v178, s[12:15], 0 offen lds            // 0000000030E0: E05D1000 800300B2
	v_accvgpr_write_b32 a16, 0                                 // 0000000030E8: D3D94010 18000080
	v_accvgpr_write_b32 a17, 0                                 // 0000000030F0: D3D94011 18000080
	v_accvgpr_write_b32 a18, 0                                 // 0000000030F8: D3D94012 18000080
	v_accvgpr_write_b32 a19, 0                                 // 000000003100: D3D94013 18000080
	v_accvgpr_write_b32 a20, 0                                 // 000000003108: D3D94014 18000080
	v_accvgpr_write_b32 a21, 0                                 // 000000003110: D3D94015 18000080
	v_accvgpr_write_b32 a22, 0                                 // 000000003118: D3D94016 18000080
	v_accvgpr_write_b32 a23, 0                                 // 000000003120: D3D94017 18000080
	s_add_u32 m0, 0x1080, s64                                  // 000000003128: 807C40FF 00001080
	buffer_load_dwordx4 v179, s[12:15], 0 offen lds            // 000000003130: E05D1000 800300B3
	v_accvgpr_write_b32 a24, 0                                 // 000000003138: D3D94018 18000080
	v_accvgpr_write_b32 a25, 0                                 // 000000003140: D3D94019 18000080
	v_accvgpr_write_b32 a26, 0                                 // 000000003148: D3D9401A 18000080
	v_accvgpr_write_b32 a27, 0                                 // 000000003150: D3D9401B 18000080
	v_accvgpr_write_b32 a28, 0                                 // 000000003158: D3D9401C 18000080
	v_accvgpr_write_b32 a29, 0                                 // 000000003160: D3D9401D 18000080
	v_accvgpr_write_b32 a30, 0                                 // 000000003168: D3D9401E 18000080
	v_accvgpr_write_b32 a31, 0                                 // 000000003170: D3D9401F 18000080
	s_add_u32 m0, 0x2100, s64                                  // 000000003178: 807C40FF 00002100
	buffer_load_dwordx4 v180, s[12:15], 0 offen lds            // 000000003180: E05D1000 800300B4
	v_accvgpr_write_b32 a32, 0                                 // 000000003188: D3D94020 18000080
	v_accvgpr_write_b32 a33, 0                                 // 000000003190: D3D94021 18000080
	v_accvgpr_write_b32 a34, 0                                 // 000000003198: D3D94022 18000080
	v_accvgpr_write_b32 a35, 0                                 // 0000000031A0: D3D94023 18000080
	v_accvgpr_write_b32 a36, 0                                 // 0000000031A8: D3D94024 18000080
	v_accvgpr_write_b32 a37, 0                                 // 0000000031B0: D3D94025 18000080
	v_accvgpr_write_b32 a38, 0                                 // 0000000031B8: D3D94026 18000080
	v_accvgpr_write_b32 a39, 0                                 // 0000000031C0: D3D94027 18000080
	s_add_u32 m0, 0x3180, s64                                  // 0000000031C8: 807C40FF 00003180
	buffer_load_dwordx4 v181, s[12:15], 0 offen lds            // 0000000031D0: E05D1000 800300B5
	v_accvgpr_write_b32 a40, 0                                 // 0000000031D8: D3D94028 18000080
	v_accvgpr_write_b32 a41, 0                                 // 0000000031E0: D3D94029 18000080
	v_accvgpr_write_b32 a42, 0                                 // 0000000031E8: D3D9402A 18000080
	v_accvgpr_write_b32 a43, 0                                 // 0000000031F0: D3D9402B 18000080
	v_accvgpr_write_b32 a44, 0                                 // 0000000031F8: D3D9402C 18000080
	v_accvgpr_write_b32 a45, 0                                 // 000000003200: D3D9402D 18000080
	v_accvgpr_write_b32 a46, 0                                 // 000000003208: D3D9402E 18000080
	v_accvgpr_write_b32 a47, 0                                 // 000000003210: D3D9402F 18000080
	s_add_u32 m0, 0x4200, s64                                  // 000000003218: 807C40FF 00004200
	buffer_load_dwordx4 v182, s[12:15], 0 offen lds            // 000000003220: E05D1000 800300B6
	v_accvgpr_write_b32 a48, 0                                 // 000000003228: D3D94030 18000080
	v_accvgpr_write_b32 a49, 0                                 // 000000003230: D3D94031 18000080
	v_accvgpr_write_b32 a50, 0                                 // 000000003238: D3D94032 18000080
	v_accvgpr_write_b32 a51, 0                                 // 000000003240: D3D94033 18000080
	v_accvgpr_write_b32 a52, 0                                 // 000000003248: D3D94034 18000080
	v_accvgpr_write_b32 a53, 0                                 // 000000003250: D3D94035 18000080
	v_accvgpr_write_b32 a54, 0                                 // 000000003258: D3D94036 18000080
	v_accvgpr_write_b32 a55, 0                                 // 000000003260: D3D94037 18000080
	s_add_u32 m0, 0x5280, s64                                  // 000000003268: 807C40FF 00005280
	buffer_load_dwordx4 v183, s[12:15], 0 offen lds            // 000000003270: E05D1000 800300B7
	v_accvgpr_write_b32 a56, 0                                 // 000000003278: D3D94038 18000080
	v_accvgpr_write_b32 a57, 0                                 // 000000003280: D3D94039 18000080
	v_accvgpr_write_b32 a58, 0                                 // 000000003288: D3D9403A 18000080
	v_accvgpr_write_b32 a59, 0                                 // 000000003290: D3D9403B 18000080
	v_accvgpr_write_b32 a60, 0                                 // 000000003298: D3D9403C 18000080
	v_accvgpr_write_b32 a61, 0                                 // 0000000032A0: D3D9403D 18000080
	v_accvgpr_write_b32 a62, 0                                 // 0000000032A8: D3D9403E 18000080
	v_accvgpr_write_b32 a63, 0                                 // 0000000032B0: D3D9403F 18000080
	s_add_u32 s62, 0x100, s60                                  // 0000000032B8: 803E3CFF 00000100
	s_cmp_lt_u32 s62, s61                                      // 0000000032C0: BF0A3D3E
	s_cselect_b32 s66, s66, 0                                  // 0000000032C4: 85428042
	s_cselect_b32 s68, s68, 0                                  // 0000000032C8: 85448044
	s_add_u32 s12, s12, s66                                    // 0000000032CC: 800C420C
	s_addc_u32 s13, 0, s13                                     // 0000000032D0: 820D0D80
	s_sub_u32 s14, s14, s66                                    // 0000000032D4: 808E420E
	s_add_u32 s20, s20, s68                                    // 0000000032D8: 80144414
	s_addc_u32 s21, 0, s21                                     // 0000000032DC: 82151580
	s_sub_u32 s22, s22, s68                                    // 0000000032E0: 80964416
	buffer_load_dwordx4 v[104:107], v189, s[16:19], 0 offen    // 0000000032E4: E05C1000 800468BD
	v_accvgpr_write_b32 a64, 0                                 // 0000000032EC: D3D94040 18000080
	v_accvgpr_write_b32 a65, 0                                 // 0000000032F4: D3D94041 18000080
	v_accvgpr_write_b32 a66, 0                                 // 0000000032FC: D3D94042 18000080
	v_accvgpr_write_b32 a67, 0                                 // 000000003304: D3D94043 18000080
	v_accvgpr_write_b32 a68, 0                                 // 00000000330C: D3D94044 18000080
	v_accvgpr_write_b32 a69, 0                                 // 000000003314: D3D94045 18000080
	v_accvgpr_write_b32 a70, 0                                 // 00000000331C: D3D94046 18000080
	v_accvgpr_write_b32 a71, 0                                 // 000000003324: D3D94047 18000080
	buffer_load_dwordx4 v[108:111], v190, s[16:19], 0 offen    // 00000000332C: E05C1000 80046CBE
	v_accvgpr_write_b32 a72, 0                                 // 000000003334: D3D94048 18000080
	v_accvgpr_write_b32 a73, 0                                 // 00000000333C: D3D94049 18000080
	v_accvgpr_write_b32 a74, 0                                 // 000000003344: D3D9404A 18000080
	v_accvgpr_write_b32 a75, 0                                 // 00000000334C: D3D9404B 18000080
	v_accvgpr_write_b32 a76, 0                                 // 000000003354: D3D9404C 18000080
	v_accvgpr_write_b32 a77, 0                                 // 00000000335C: D3D9404D 18000080
	v_accvgpr_write_b32 a78, 0                                 // 000000003364: D3D9404E 18000080
	v_accvgpr_write_b32 a79, 0                                 // 00000000336C: D3D9404F 18000080
	buffer_load_dwordx4 v[112:115], v189, s[16:19], 0 offen offset:1024// 000000003374: E05C1400 800470BD
	v_accvgpr_write_b32 a80, 0                                 // 00000000337C: D3D94050 18000080
	v_accvgpr_write_b32 a81, 0                                 // 000000003384: D3D94051 18000080
	v_accvgpr_write_b32 a82, 0                                 // 00000000338C: D3D94052 18000080
	v_accvgpr_write_b32 a83, 0                                 // 000000003394: D3D94053 18000080
	v_accvgpr_write_b32 a84, 0                                 // 00000000339C: D3D94054 18000080
	v_accvgpr_write_b32 a85, 0                                 // 0000000033A4: D3D94055 18000080
	v_accvgpr_write_b32 a86, 0                                 // 0000000033AC: D3D94056 18000080
	v_accvgpr_write_b32 a87, 0                                 // 0000000033B4: D3D94057 18000080
	buffer_load_dwordx4 v[116:119], v190, s[16:19], 0 offen offset:1024// 0000000033BC: E05C1400 800474BE
	v_accvgpr_write_b32 a88, 0                                 // 0000000033C4: D3D94058 18000080
	v_accvgpr_write_b32 a89, 0                                 // 0000000033CC: D3D94059 18000080
	v_accvgpr_write_b32 a90, 0                                 // 0000000033D4: D3D9405A 18000080
	v_accvgpr_write_b32 a91, 0                                 // 0000000033DC: D3D9405B 18000080
	v_accvgpr_write_b32 a92, 0                                 // 0000000033E4: D3D9405C 18000080
	v_accvgpr_write_b32 a93, 0                                 // 0000000033EC: D3D9405D 18000080
	v_accvgpr_write_b32 a94, 0                                 // 0000000033F4: D3D9405E 18000080
	v_accvgpr_write_b32 a95, 0                                 // 0000000033FC: D3D9405F 18000080
	buffer_load_dword v174, v193, s[24:27], 0 offen            // 000000003404: E0501000 8006AEC1
	v_accvgpr_write_b32 a96, 0                                 // 00000000340C: D3D94060 18000080
	v_accvgpr_write_b32 a97, 0                                 // 000000003414: D3D94061 18000080
	v_accvgpr_write_b32 a98, 0                                 // 00000000341C: D3D94062 18000080
	v_accvgpr_write_b32 a99, 0                                 // 000000003424: D3D94063 18000080
	v_accvgpr_write_b32 a100, 0                                // 00000000342C: D3D94064 18000080
	v_accvgpr_write_b32 a101, 0                                // 000000003434: D3D94065 18000080
	v_accvgpr_write_b32 a102, 0                                // 00000000343C: D3D94066 18000080
	v_accvgpr_write_b32 a103, 0                                // 000000003444: D3D94067 18000080
	buffer_load_dwordx4 v[120:123], v191, s[16:19], 0 offen    // 00000000344C: E05C1000 800478BF
	v_accvgpr_write_b32 a104, 0                                // 000000003454: D3D94068 18000080
	v_accvgpr_write_b32 a105, 0                                // 00000000345C: D3D94069 18000080
	v_accvgpr_write_b32 a106, 0                                // 000000003464: D3D9406A 18000080
	v_accvgpr_write_b32 a107, 0                                // 00000000346C: D3D9406B 18000080
	v_accvgpr_write_b32 a108, 0                                // 000000003474: D3D9406C 18000080
	v_accvgpr_write_b32 a109, 0                                // 00000000347C: D3D9406D 18000080
	v_accvgpr_write_b32 a110, 0                                // 000000003484: D3D9406E 18000080
	v_accvgpr_write_b32 a111, 0                                // 00000000348C: D3D9406F 18000080
	buffer_load_dwordx4 v[124:127], v192, s[16:19], 0 offen    // 000000003494: E05C1000 80047CC0
	v_accvgpr_write_b32 a112, 0                                // 00000000349C: D3D94070 18000080
	v_accvgpr_write_b32 a113, 0                                // 0000000034A4: D3D94071 18000080
	v_accvgpr_write_b32 a114, 0                                // 0000000034AC: D3D94072 18000080
	v_accvgpr_write_b32 a115, 0                                // 0000000034B4: D3D94073 18000080
	v_accvgpr_write_b32 a116, 0                                // 0000000034BC: D3D94074 18000080
	v_accvgpr_write_b32 a117, 0                                // 0000000034C4: D3D94075 18000080
	v_accvgpr_write_b32 a118, 0                                // 0000000034CC: D3D94076 18000080
	v_accvgpr_write_b32 a119, 0                                // 0000000034D4: D3D94077 18000080
	buffer_load_dwordx4 v[128:131], v191, s[16:19], 0 offen offset:1024// 0000000034DC: E05C1400 800480BF
	v_accvgpr_write_b32 a120, 0                                // 0000000034E4: D3D94078 18000080
	v_accvgpr_write_b32 a121, 0                                // 0000000034EC: D3D94079 18000080
	v_accvgpr_write_b32 a122, 0                                // 0000000034F4: D3D9407A 18000080
	v_accvgpr_write_b32 a123, 0                                // 0000000034FC: D3D9407B 18000080
	v_accvgpr_write_b32 a124, 0                                // 000000003504: D3D9407C 18000080
	v_accvgpr_write_b32 a125, 0                                // 00000000350C: D3D9407D 18000080
	v_accvgpr_write_b32 a126, 0                                // 000000003514: D3D9407E 18000080
	v_accvgpr_write_b32 a127, 0                                // 00000000351C: D3D9407F 18000080
	buffer_load_dwordx4 v[132:135], v192, s[16:19], 0 offen offset:1024// 000000003524: E05C1400 800484C0
	v_accvgpr_write_b32 a128, 0                                // 00000000352C: D3D94080 18000080
	v_accvgpr_write_b32 a129, 0                                // 000000003534: D3D94081 18000080
	v_accvgpr_write_b32 a130, 0                                // 00000000353C: D3D94082 18000080
	v_accvgpr_write_b32 a131, 0                                // 000000003544: D3D94083 18000080
	v_accvgpr_write_b32 a132, 0                                // 00000000354C: D3D94084 18000080
	v_accvgpr_write_b32 a133, 0                                // 000000003554: D3D94085 18000080
	v_accvgpr_write_b32 a134, 0                                // 00000000355C: D3D94086 18000080
	v_accvgpr_write_b32 a135, 0                                // 000000003564: D3D94087 18000080
	buffer_load_dword v175, v194, s[24:27], 0 offen            // 00000000356C: E0501000 8006AFC2
	v_accvgpr_write_b32 a136, 0                                // 000000003574: D3D94088 18000080
	v_accvgpr_write_b32 a137, 0                                // 00000000357C: D3D94089 18000080
	v_accvgpr_write_b32 a138, 0                                // 000000003584: D3D9408A 18000080
	v_accvgpr_write_b32 a139, 0                                // 00000000358C: D3D9408B 18000080
	v_accvgpr_write_b32 a140, 0                                // 000000003594: D3D9408C 18000080
	v_accvgpr_write_b32 a141, 0                                // 00000000359C: D3D9408D 18000080
	v_accvgpr_write_b32 a142, 0                                // 0000000035A4: D3D9408E 18000080
	v_accvgpr_write_b32 a143, 0                                // 0000000035AC: D3D9408F 18000080
	s_add_u32 s63, 0x100, s60                                  // 0000000035B4: 803F3CFF 00000100
	s_cmp_lt_u32 s63, s61                                      // 0000000035BC: BF0A3D3F
	s_cselect_b32 s67, s67, 0                                  // 0000000035C0: 85438043
	s_cselect_b32 s69, s69, 0                                  // 0000000035C4: 85458045
	s_add_u32 s16, s16, s67                                    // 0000000035C8: 80104310
	s_addc_u32 s17, 0, s17                                     // 0000000035CC: 82111180
	s_sub_u32 s18, s18, s67                                    // 0000000035D0: 80924312
	s_add_u32 s24, s24, s69                                    // 0000000035D4: 80184518
	s_addc_u32 s25, 0, s25                                     // 0000000035D8: 82191980
	s_sub_u32 s26, s26, s69                                    // 0000000035DC: 809A451A
	s_add_u32 m0, 0x800, s65                                   // 0000000035E0: 807C41FF 00000800
	buffer_load_dword v186, s[20:23], 0 offen lds              // 0000000035E8: E0511000 800500BA
	v_accvgpr_write_b32 a144, 0                                // 0000000035F0: D3D94090 18000080
	v_accvgpr_write_b32 a145, 0                                // 0000000035F8: D3D94091 18000080
	v_accvgpr_write_b32 a146, 0                                // 000000003600: D3D94092 18000080
	v_accvgpr_write_b32 a147, 0                                // 000000003608: D3D94093 18000080
	v_accvgpr_write_b32 a148, 0                                // 000000003610: D3D94094 18000080
	v_accvgpr_write_b32 a149, 0                                // 000000003618: D3D94095 18000080
	v_accvgpr_write_b32 a150, 0                                // 000000003620: D3D94096 18000080
	v_accvgpr_write_b32 a151, 0                                // 000000003628: D3D94097 18000080
	s_add_u32 m0, 0xc00, s65                                   // 000000003630: 807C41FF 00000C00
	buffer_load_dword v187, s[20:23], 0 offen lds              // 000000003638: E0511000 800500BB
	v_accvgpr_write_b32 a152, 0                                // 000000003640: D3D94098 18000080
	v_accvgpr_write_b32 a153, 0                                // 000000003648: D3D94099 18000080
	v_accvgpr_write_b32 a154, 0                                // 000000003650: D3D9409A 18000080
	v_accvgpr_write_b32 a155, 0                                // 000000003658: D3D9409B 18000080
	v_accvgpr_write_b32 a156, 0                                // 000000003660: D3D9409C 18000080
	v_accvgpr_write_b32 a157, 0                                // 000000003668: D3D9409D 18000080
	v_accvgpr_write_b32 a158, 0                                // 000000003670: D3D9409E 18000080
	v_accvgpr_write_b32 a159, 0                                // 000000003678: D3D9409F 18000080
	s_add_u32 m0, 0x6300, s64                                  // 000000003680: 807C40FF 00006300
	buffer_load_dwordx4 v178, s[12:15], 0 offen lds            // 000000003688: E05D1000 800300B2
	v_accvgpr_write_b32 a160, 0                                // 000000003690: D3D940A0 18000080
	v_accvgpr_write_b32 a161, 0                                // 000000003698: D3D940A1 18000080
	v_accvgpr_write_b32 a162, 0                                // 0000000036A0: D3D940A2 18000080
	v_accvgpr_write_b32 a163, 0                                // 0000000036A8: D3D940A3 18000080
	v_accvgpr_write_b32 a164, 0                                // 0000000036B0: D3D940A4 18000080
	v_accvgpr_write_b32 a165, 0                                // 0000000036B8: D3D940A5 18000080
	v_accvgpr_write_b32 a166, 0                                // 0000000036C0: D3D940A6 18000080
	v_accvgpr_write_b32 a167, 0                                // 0000000036C8: D3D940A7 18000080
	s_add_u32 m0, 0x7380, s64                                  // 0000000036D0: 807C40FF 00007380
	buffer_load_dwordx4 v179, s[12:15], 0 offen lds            // 0000000036D8: E05D1000 800300B3
	v_accvgpr_write_b32 a168, 0                                // 0000000036E0: D3D940A8 18000080
	v_accvgpr_write_b32 a169, 0                                // 0000000036E8: D3D940A9 18000080
	v_accvgpr_write_b32 a170, 0                                // 0000000036F0: D3D940AA 18000080
	v_accvgpr_write_b32 a171, 0                                // 0000000036F8: D3D940AB 18000080
	v_accvgpr_write_b32 a172, 0                                // 000000003700: D3D940AC 18000080
	v_accvgpr_write_b32 a173, 0                                // 000000003708: D3D940AD 18000080
	v_accvgpr_write_b32 a174, 0                                // 000000003710: D3D940AE 18000080
	v_accvgpr_write_b32 a175, 0                                // 000000003718: D3D940AF 18000080
	s_add_u32 m0, 0x8400, s64                                  // 000000003720: 807C40FF 00008400
	buffer_load_dwordx4 v180, s[12:15], 0 offen lds            // 000000003728: E05D1000 800300B4
	v_accvgpr_write_b32 a176, 0                                // 000000003730: D3D940B0 18000080
	v_accvgpr_write_b32 a177, 0                                // 000000003738: D3D940B1 18000080
	v_accvgpr_write_b32 a178, 0                                // 000000003740: D3D940B2 18000080
	v_accvgpr_write_b32 a179, 0                                // 000000003748: D3D940B3 18000080
	v_accvgpr_write_b32 a180, 0                                // 000000003750: D3D940B4 18000080
	v_accvgpr_write_b32 a181, 0                                // 000000003758: D3D940B5 18000080
	v_accvgpr_write_b32 a182, 0                                // 000000003760: D3D940B6 18000080
	v_accvgpr_write_b32 a183, 0                                // 000000003768: D3D940B7 18000080
	s_add_u32 m0, 0x9480, s64                                  // 000000003770: 807C40FF 00009480
	buffer_load_dwordx4 v181, s[12:15], 0 offen lds            // 000000003778: E05D1000 800300B5
	v_accvgpr_write_b32 a184, 0                                // 000000003780: D3D940B8 18000080
	v_accvgpr_write_b32 a185, 0                                // 000000003788: D3D940B9 18000080
	v_accvgpr_write_b32 a186, 0                                // 000000003790: D3D940BA 18000080
	v_accvgpr_write_b32 a187, 0                                // 000000003798: D3D940BB 18000080
	v_accvgpr_write_b32 a188, 0                                // 0000000037A0: D3D940BC 18000080
	v_accvgpr_write_b32 a189, 0                                // 0000000037A8: D3D940BD 18000080
	v_accvgpr_write_b32 a190, 0                                // 0000000037B0: D3D940BE 18000080
	v_accvgpr_write_b32 a191, 0                                // 0000000037B8: D3D940BF 18000080
	s_add_u32 m0, 0xa500, s64                                  // 0000000037C0: 807C40FF 0000A500
	buffer_load_dwordx4 v182, s[12:15], 0 offen lds            // 0000000037C8: E05D1000 800300B6
	s_add_u32 m0, 0xb580, s64                                  // 0000000037D0: 807C40FF 0000B580
	buffer_load_dwordx4 v183, s[12:15], 0 offen lds            // 0000000037D8: E05D1000 800300B7
	s_add_u32 s62, 0x200, s60                                  // 0000000037E0: 803E3CFF 00000200
	s_cmp_lt_u32 s62, s61                                      // 0000000037E8: BF0A3D3E
	s_cselect_b32 s66, s66, 0                                  // 0000000037EC: 85428042
	s_cselect_b32 s68, s68, 0                                  // 0000000037F0: 85448044
	s_add_u32 s12, s12, s66                                    // 0000000037F4: 800C420C
	s_addc_u32 s13, 0, s13                                     // 0000000037F8: 820D0D80
	s_sub_u32 s14, s14, s66                                    // 0000000037FC: 808E420E
	s_add_u32 s20, s20, s68                                    // 000000003800: 80144414
	s_addc_u32 s21, 0, s21                                     // 000000003804: 82151580
	s_sub_u32 s22, s22, s68                                    // 000000003808: 80964416
	buffer_load_dwordx4 v[136:139], v189, s[16:19], 0 offen    // 00000000380C: E05C1000 800488BD
	buffer_load_dwordx4 v[140:143], v190, s[16:19], 0 offen    // 000000003814: E05C1000 80048CBE
	buffer_load_dwordx4 v[144:147], v189, s[16:19], 0 offen offset:1024// 00000000381C: E05C1400 800490BD
	buffer_load_dwordx4 v[148:151], v190, s[16:19], 0 offen offset:1024// 000000003824: E05C1400 800494BE
	buffer_load_dword v176, v193, s[24:27], 0 offen            // 00000000382C: E0501000 8006B0C1
	s_waitcnt vmcnt(27)                                        // 000000003834: BF8C4F7B
	s_barrier                                                  // 000000003838: BF8A0000
	ds_read_b128 v[8:11], v184                                 // 00000000383C: D9FE0000 080000B8
	ds_read_b128 v[16:19], v184 offset:64                      // 000000003844: D9FE0040 100000B8
	ds_read_b128 v[12:15], v184 offset:512                     // 00000000384C: D9FE0200 0C0000B8
	ds_read_b128 v[20:23], v184 offset:576                     // 000000003854: D9FE0240 140000B8
	ds_read_b32 v168, v188                                     // 00000000385C: D86C0000 A80000BC
	ds_read_b128 v[24:27], v184 offset:4224                    // 000000003864: D9FE1080 180000B8
	ds_read_b128 v[32:35], v184 offset:4288                    // 00000000386C: D9FE10C0 200000B8
	ds_read_b128 v[28:31], v184 offset:4736                    // 000000003874: D9FE1280 1C0000B8
	ds_read_b128 v[36:39], v184 offset:4800                    // 00000000387C: D9FE12C0 240000B8
	ds_read_b32 v169, v188 offset:256                          // 000000003884: D86C0100 A90000BC
	s_nop 0                                                    // 00000000388C: BF800000
	s_nop 0                                                    // 000000003890: BF800000
	s_nop 0                                                    // 000000003894: BF800000
	s_nop 0                                                    // 000000003898: BF800000
	s_nop 0                                                    // 00000000389C: BF800000
	s_lshl_b32 s36, s36, 1                                     // 0000000038A0: 8E248124
	s_and_b32 s5, s5, 0xffff                                   // 0000000038A4: 8605FF05 0000FFFF
	s_or_b32 s5, s5, 0x40000                                   // 0000000038AC: 8705FF05 00040000
	s_mul_i32 s62, s48, 0xc0                                   // 0000000038B4: 923EFF30 000000C0
	s_mul_hi_u32 s63, s36, s62                                 // 0000000038BC: 963F3E24
	s_add_u32 s5, s5, s63                                      // 0000000038C0: 80053F05
	s_mul_i32 s63, s36, s62                                    // 0000000038C4: 923F3E24
	s_add_u32 s4, s4, s63                                      // 0000000038C8: 80043F04
	s_addc_u32 s5, s5, 0                                       // 0000000038CC: 82058005
	s_mul_i32 s63, s47, 0x100                                  // 0000000038D0: 923FFF2F 00000100
	s_lshl_b32 s63, s63, 1                                     // 0000000038D8: 8E3F813F
	s_add_u32 s4, s4, s63                                      // 0000000038DC: 80043F04
	s_addc_u32 s5, s5, 0                                       // 0000000038E0: 82058005
	s_sub_i32 s62, s43, s62                                    // 0000000038E4: 81BE3E2B
	s_cmp_lt_u32 s62, 0xc0                                     // 0000000038E8: BF0AFF3E 000000C0
	s_cselect_b32 s62, s62, 0xc0                               // 0000000038F0: 853EFF3E 000000C0
	s_mul_i32 s62, s36, s62                                    // 0000000038F8: 923E3E24
	s_sub_i32 s62, s62, s63                                    // 0000000038FC: 81BE3F3E
	s_mov_b32 s6, s62                                          // 000000003900: BE86003E
	s_mov_b32 s7, 0x20000                                      // 000000003904: BE8700FF 00020000
	v_lshrrev_b32_e32 v4, 3, v0                                // 00000000390C: 20080083
	v_mul_lo_u32 v195, v4, s36                                 // 000000003910: D28500C3 00004904
	s_mul_i32 s62, s46, 64                                     // 000000003918: 923EC02E
	s_lshl_b32 s62, s62, 1                                     // 00000000391C: 8E3E813E
	v_and_b32_e32 v4, 7, v0                                    // 000000003920: 26080087
	v_mul_i32_i24_e32 v4, 16, v4                               // 000000003924: 0C080890
	v_add_u32_e32 v4, s62, v4                                  // 000000003928: 6808083E
	v_add_u32_e32 v195, v195, v4                               // 00000000392C: 698609C3
	s_mul_i32 s62, s36, 8                                      // 000000003930: 923E8824
	v_add_u32_e32 v196, s62, v195                              // 000000003934: 6989863E
	v_add_u32_e32 v197, s62, v196                              // 000000003938: 698B883E
	v_add_u32_e32 v198, s62, v197                              // 00000000393C: 698D8A3E
	v_add_u32_e32 v199, s62, v198                              // 000000003940: 698F8C3E
	v_add_u32_e32 v200, s62, v199                              // 000000003944: 69918E3E
	v_add_u32_e32 v201, s62, v200                              // 000000003948: 6993903E
	v_add_u32_e32 v202, s62, v201                              // 00000000394C: 6995923E
	v_add_u32_e32 v203, s62, v202                              // 000000003950: 6997943E
	v_add_u32_e32 v204, s62, v203                              // 000000003954: 6999963E
	v_add_u32_e32 v205, s62, v204                              // 000000003958: 699B983E
	v_add_u32_e32 v206, s62, v205                              // 00000000395C: 699D9A3E
	v_add_u32_e32 v207, s62, v206                              // 000000003960: 699F9C3E
	v_add_u32_e32 v208, s62, v207                              // 000000003964: 69A19E3E
	v_add_u32_e32 v209, s62, v208                              // 000000003968: 69A3A03E
	v_add_u32_e32 v210, s62, v209                              // 00000000396C: 69A5A23E
	v_add_u32_e32 v211, s62, v210                              // 000000003970: 69A7A43E
	v_add_u32_e32 v212, s62, v211                              // 000000003974: 69A9A63E
	v_add_u32_e32 v213, s62, v212                              // 000000003978: 69ABA83E
	v_add_u32_e32 v214, s62, v213                              // 00000000397C: 69ADAA3E
	v_add_u32_e32 v215, s62, v214                              // 000000003980: 69AFAC3E
	v_add_u32_e32 v216, s62, v215                              // 000000003984: 69B1AE3E
	v_add_u32_e32 v217, s62, v216                              // 000000003988: 69B3B03E
	v_add_u32_e32 v218, s62, v217                              // 00000000398C: 69B5B23E
	s_cmp_lt_i32 s46, 2                                        // 000000003990: BF04822E
	s_cbranch_scc0 label_078D                                  // 000000003994: BF840427

0000000000003998 <label_0366>:
	s_waitcnt vmcnt(18)                                        // 000000003998: BF8C4F72
	s_barrier                                                  // 00000000399C: BF8A0000
	s_waitcnt lgkmcnt(5)                                       // 0000000039A0: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[8:11], a[0:3], v174, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000039A4: D3AC6000 000351AE D3AD8C00 84021168
	ds_read_b128 v[40:43], v184 offset:8448                    // 0000000039B4: D9FE2100 280000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[12:15], a[4:7], v174, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000039BC: D3AC7000 000351AE D3AD8C04 84121968
	buffer_load_dwordx4 v[152:155], v191, s[16:19], 0 offen    // 0000000039CC: E05C1000 800498BF
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[8:11], a[8:11], v174, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000039D4: D3AC6800 000351AE D3AD8C08 8422116C
	ds_read_b128 v[48:51], v184 offset:8512                    // 0000000039E4: D9FE2140 300000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[12:15], a[12:15], v174, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000039EC: D3AC7800 000351AE D3AD8C0C 8432196C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:115], v[16:19], a[0:3], v174, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000039FC: D3AC6000 180351AE D3AD8C00 84022170
	ds_read_b128 v[44:47], v184 offset:8960                    // 000000003A0C: D9FE2300 2C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:115], v[20:23], a[4:7], v174, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003A14: D3AC7000 180351AE D3AD8C04 84122970
	buffer_load_dwordx4 v[156:159], v192, s[16:19], 0 offen    // 000000003A24: E05C1000 80049CC0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[116:119], v[16:19], a[8:11], v174, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003A2C: D3AC6800 180351AE D3AD8C08 84222174
	ds_read_b128 v[52:55], v184 offset:9024                    // 000000003A3C: D9FE2340 340000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[20:23], a[12:15], v174, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003A44: D3AC7800 180351AE D3AD8C0C 84322974
	ds_read_b32 v170, v188 offset:512                          // 000000003A54: D86C0200 AA0000BC
	s_waitcnt lgkmcnt(5)                                       // 000000003A5C: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[24:27], a[32:35], v174, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003A60: D3AC6000 000353AE D3AD8C20 84823168
	ds_read_b128 v[56:59], v184 offset:12672                   // 000000003A70: D9FE3180 380000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[28:31], a[36:39], v174, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003A78: D3AC7000 000353AE D3AD8C24 84923968
	buffer_load_dwordx4 v[160:163], v191, s[16:19], 0 offen offset:1024// 000000003A88: E05C1400 8004A0BF
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[24:27], a[40:43], v174, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003A90: D3AC6800 000353AE D3AD8C28 84A2316C
	ds_read_b128 v[64:67], v184 offset:12736                   // 000000003AA0: D9FE31C0 400000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[28:31], a[44:47], v174, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003AA8: D3AC7800 000353AE D3AD8C2C 84B2396C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[32:35], a[32:35], v174, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003AB8: D3AC6000 180353AE D3AD8C20 84824170
	ds_read_b128 v[60:63], v184 offset:13184                   // 000000003AC8: D9FE3380 3C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[36:39], a[36:39], v174, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003AD0: D3AC7000 180353AE D3AD8C24 84924970
	buffer_load_dwordx4 v[164:167], v192, s[16:19], 0 offen offset:1024// 000000003AE0: E05C1400 8004A4C0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[32:35], a[40:43], v174, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003AE8: D3AC6800 180353AE D3AD8C28 84A24174
	ds_read_b128 v[68:71], v184 offset:13248                   // 000000003AF8: D9FE33C0 440000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[36:39], a[44:47], v174, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003B00: D3AC7800 180353AE D3AD8C2C 84B24974
	ds_read_b32 v171, v188 offset:768                          // 000000003B10: D86C0300 AB0000BC
	s_waitcnt lgkmcnt(5)                                       // 000000003B18: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[104:107], v[40:43], a[64:67], v174, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003B1C: D3AC6000 000355AE D3AD8C40 85025168
	ds_read_b128 v[72:75], v184 offset:16896                   // 000000003B2C: D9FE4200 480000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[104:107], v[44:47], a[68:71], v174, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003B34: D3AC7000 000355AE D3AD8C44 85125968
	buffer_load_dword v177, v194, s[24:27], 0 offen            // 000000003B44: E0501000 8006B1C2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[40:43], a[72:75], v174, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003B4C: D3AC6800 000355AE D3AD8C48 8522516C
	s_add_u32 s63, 0x200, s60                                  // 000000003B5C: 803F3CFF 00000200
	ds_read_b128 v[80:83], v184 offset:16960                   // 000000003B64: D9FE4240 500000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[44:47], a[76:79], v174, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003B6C: D3AC7800 000355AE D3AD8C4C 8532596C
	s_cmp_lt_u32 s63, s61                                      // 000000003B7C: BF0A3D3F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:115], v[48:51], a[64:67], v174, v170 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003B80: D3AC6000 180355AE D3AD8C40 85026170
	s_cselect_b32 s67, s67, 0                                  // 000000003B90: 85438043
	ds_read_b128 v[76:79], v184 offset:17408                   // 000000003B94: D9FE4400 4C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[112:115], v[52:55], a[68:71], v174, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003B9C: D3AC7000 180355AE D3AD8C44 85126970
	s_cselect_b32 s69, s69, 0                                  // 000000003BAC: 85458045
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[116:119], v[48:51], a[72:75], v174, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003BB0: D3AC6800 180355AE D3AD8C48 85226174
	s_add_u32 s16, s16, s67                                    // 000000003BC0: 80104310
	ds_read_b128 v[84:87], v184 offset:17472                   // 000000003BC4: D9FE4440 540000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:119], v[52:55], a[76:79], v174, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003BCC: D3AC7800 180355AE D3AD8C4C 85326974
	s_addc_u32 s17, 0, s17                                     // 000000003BDC: 82111180
	s_sub_u32 s18, s18, s67                                    // 000000003BE0: 80924312
	ds_read_b32 v172, v188 offset:1024                         // 000000003BE4: D86C0400 AC0000BC
	s_waitcnt lgkmcnt(5)                                       // 000000003BEC: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:107], v[56:59], a[96:99], v174, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003BF0: D3AC6000 000357AE D3AD8C60 85827168
	s_add_u32 s24, s24, s69                                    // 000000003C00: 80184518
	ds_read_b128 v[88:91], v184 offset:21120                   // 000000003C04: D9FE5280 580000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:107], v[60:63], a[100:103], v174, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003C0C: D3AC7000 000357AE D3AD8C64 85927968
	s_addc_u32 s25, 0, s25                                     // 000000003C1C: 82191980
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:111], v[56:59], a[104:107], v174, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003C20: D3AC6800 000357AE D3AD8C68 85A2716C
	s_sub_u32 s26, s26, s69                                    // 000000003C30: 809A451A
	ds_read_b128 v[96:99], v184 offset:21184                   // 000000003C34: D9FE52C0 600000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[60:63], a[108:111], v174, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003C3C: D3AC7800 000357AE D3AD8C6C 85B2796C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:115], v[64:67], a[96:99], v174, v171 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003C4C: D3AC6000 180357AE D3AD8C60 85828170
	ds_read_b128 v[92:95], v184 offset:21632                   // 000000003C5C: D9FE5480 5C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[112:115], v[68:71], a[100:103], v174, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003C64: D3AC7000 180357AE D3AD8C64 85928970
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[116:119], v[64:67], a[104:107], v174, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003C74: D3AC6800 180357AE D3AD8C68 85A28174
	ds_read_b128 v[100:103], v184 offset:21696                 // 000000003C84: D9FE54C0 640000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:119], v[68:71], a[108:111], v174, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003C8C: D3AC7800 180357AE D3AD8C6C 85B28974
	ds_read_b32 v173, v188 offset:1280                         // 000000003C9C: D86C0500 AD0000BC
	s_barrier                                                  // 000000003CA4: BF8A0000
	s_waitcnt lgkmcnt(5)                                       // 000000003CA8: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[104:107], v[72:75], a[128:131], v174, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003CAC: D3AC6000 000359AE D3AD8C80 86029168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[104:107], v[76:79], a[132:135], v174, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003CBC: D3AC7000 000359AE D3AD8C84 86129968
	s_add_u32 m0, 0, s65                                       // 000000003CCC: 807C4180
	buffer_load_dword v186, s[20:23], 0 offen lds              // 000000003CD0: E0511000 800500BA
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:111], v[72:75], a[136:139], v174, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003CD8: D3AC6800 000359AE D3AD8C88 8622916C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[108:111], v[76:79], a[140:143], v174, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003CE8: D3AC7800 000359AE D3AD8C8C 8632996C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:115], v[80:83], a[128:131], v174, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003CF8: D3AC6000 180359AE D3AD8C80 8602A170
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[112:115], v[84:87], a[132:135], v174, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003D08: D3AC7000 180359AE D3AD8C84 8612A970
	s_add_u32 m0, 0x400, s65                                   // 000000003D18: 807C41FF 00000400
	buffer_load_dword v187, s[20:23], 0 offen lds              // 000000003D20: E0511000 800500BB
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[116:119], v[80:83], a[136:139], v174, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003D28: D3AC6800 180359AE D3AD8C88 8622A174
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[116:119], v[84:87], a[140:143], v174, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003D38: D3AC7800 180359AE D3AD8C8C 8632A974
	s_waitcnt lgkmcnt(0)                                       // 000000003D48: BF8CC07F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[104:107], v[88:91], a[160:163], v174, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D4C: D3AC6000 00035BAE D3AD8CA0 8682B168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[104:107], v[92:95], a[164:167], v174, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D5C: D3AC7000 00035BAE D3AD8CA4 8692B968
	s_add_u32 m0, 0, s64                                       // 000000003D6C: 807C4080
	buffer_load_dwordx4 v178, s[12:15], 0 offen lds            // 000000003D70: E05D1000 800300B2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:111], v[88:91], a[168:171], v174, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D78: D3AC6800 00035BAE D3AD8CA8 86A2B16C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[108:111], v[92:95], a[172:175], v174, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D88: D3AC7800 00035BAE D3AD8CAC 86B2B96C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[96:99], a[160:163], v174, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003D98: D3AC6000 18035BAE D3AD8CA0 8682C170
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[112:115], v[100:103], a[164:167], v174, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003DA8: D3AC7000 18035BAE D3AD8CA4 8692C970
	s_add_u32 m0, 0x1080, s64                                  // 000000003DB8: 807C40FF 00001080
	buffer_load_dwordx4 v179, s[12:15], 0 offen lds            // 000000003DC0: E05D1000 800300B3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[96:99], a[168:171], v174, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003DC8: D3AC6800 18035BAE D3AD8CA8 86A2C174
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[100:103], a[172:175], v174, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003DD8: D3AC7800 18035BAE D3AD8CAC 86B2C974
	s_waitcnt vmcnt(18)                                        // 000000003DE8: BF8C4F72
	s_barrier                                                  // 000000003DEC: BF8A0000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[8:11], a[16:19], v175, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003DF0: D3AC6000 000351AF D3AD8C10 84421178
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[12:15], a[20:23], v175, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E00: D3AC7000 000351AF D3AD8C14 84521978
	s_add_u32 m0, 0x2100, s64                                  // 000000003E10: 807C40FF 00002100
	buffer_load_dwordx4 v180, s[12:15], 0 offen lds            // 000000003E18: E05D1000 800300B4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[8:11], a[24:27], v175, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E20: D3AC6800 000351AF D3AD8C18 8462117C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[12:15], a[28:31], v175, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E30: D3AC7800 000351AF D3AD8C1C 8472197C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[16:19], a[16:19], v175, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003E40: D3AC6000 180351AF D3AD8C10 84422180
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[20:23], a[20:23], v175, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003E50: D3AC7000 180351AF D3AD8C14 84522980
	s_add_u32 m0, 0x3180, s64                                  // 000000003E60: 807C40FF 00003180
	buffer_load_dwordx4 v181, s[12:15], 0 offen lds            // 000000003E68: E05D1000 800300B5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[16:19], a[24:27], v175, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003E70: D3AC6800 180351AF D3AD8C18 84622184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[20:23], a[28:31], v175, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003E80: D3AC7800 180351AF D3AD8C1C 84722984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[24:27], a[48:51], v175, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E90: D3AC6000 000353AF D3AD8C30 84C23178
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[28:31], a[52:55], v175, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003EA0: D3AC7000 000353AF D3AD8C34 84D23978
	s_add_u32 m0, 0x4200, s64                                  // 000000003EB0: 807C40FF 00004200
	buffer_load_dwordx4 v182, s[12:15], 0 offen lds            // 000000003EB8: E05D1000 800300B6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[24:27], a[56:59], v175, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003EC0: D3AC6800 000353AF D3AD8C38 84E2317C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[28:31], a[60:63], v175, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003ED0: D3AC7800 000353AF D3AD8C3C 84F2397C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[32:35], a[48:51], v175, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003EE0: D3AC6000 180353AF D3AD8C30 84C24180
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[36:39], a[52:55], v175, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003EF0: D3AC7000 180353AF D3AD8C34 84D24980
	s_add_u32 m0, 0x5280, s64                                  // 000000003F00: 807C40FF 00005280
	buffer_load_dwordx4 v183, s[12:15], 0 offen lds            // 000000003F08: E05D1000 800300B7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[32:35], a[56:59], v175, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003F10: D3AC6800 180353AF D3AD8C38 84E24184
	s_add_u32 s62, 0x300, s60                                  // 000000003F20: 803E3CFF 00000300
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[36:39], a[60:63], v175, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003F28: D3AC5800 180353AF D3AD8C3C 84F24984
	s_cmp_lt_u32 s62, s61                                      // 000000003F38: BF0A3D3E
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[40:43], a[80:83], v175, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003F3C: D3AC6000 000355AF D3AD8C50 85425178
	s_cselect_b32 s66, s66, 0                                  // 000000003F4C: 85428042
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[44:47], a[84:87], v175, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003F50: D3AC5000 000355AF D3AD8C54 85525978
	s_cselect_b32 s68, s68, 0                                  // 000000003F60: 85448044
	buffer_load_dwordx4 v[104:107], v189, s[16:19], 0 offen    // 000000003F64: E05C1000 800468BD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[40:43], a[88:91], v175, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003F6C: D3AC6800 000355AF D3AD8C58 8562517C
	s_add_u32 s12, s12, s66                                    // 000000003F7C: 800C420C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[124:127], v[44:47], a[92:95], v175, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003F80: D3AC5800 000355AF D3AD8C5C 8572597C
	s_addc_u32 s13, 0, s13                                     // 000000003F90: 820D0D80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[128:131], v[48:51], a[80:83], v175, v170 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003F94: D3AC4000 180355AF D3AD8C50 85426180
	s_sub_u32 s14, s14, s66                                    // 000000003FA4: 808E420E
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[128:131], v[52:55], a[84:87], v175, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003FA8: D3AC5000 180355AF D3AD8C54 85526980
	s_add_u32 s20, s20, s68                                    // 000000003FB8: 80144414
	buffer_load_dwordx4 v[108:111], v190, s[16:19], 0 offen    // 000000003FBC: E05C1000 80046CBE
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[48:51], a[88:91], v175, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003FC4: D3AC6800 180355AF D3AD8C58 85626184
	s_addc_u32 s21, 0, s21                                     // 000000003FD4: 82151580
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[52:55], a[92:95], v175, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003FD8: D3AC5800 180355AF D3AD8C5C 85726984
	s_sub_u32 s22, s22, s68                                    // 000000003FE8: 80964416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[56:59], a[112:115], v175, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003FEC: D3AC4000 000357AF D3AD8C70 85C27178
	s_addk_i32 s60, 0x100                                      // 000000003FFC: B73C0100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[60:63], a[116:119], v175, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004000: D3AC7000 000357AF D3AD8C74 85D27978
	s_cmp_lt_i32 s60, s61                                      // 000000004010: BF043D3C
	buffer_load_dwordx4 v[112:115], v189, s[16:19], 0 offen offset:1024// 000000004014: E05C1400 800470BD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[56:59], a[120:123], v175, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000401C: D3AC6800 000357AF D3AD8C78 85E2717C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:127], v[60:63], a[124:127], v175, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000402C: D3AC5800 000357AF D3AD8C7C 85F2797C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[64:67], a[112:115], v175, v171 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000403C: D3AC4000 180357AF D3AD8C70 85C28180
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[68:71], a[116:119], v175, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000404C: D3AC5000 180357AF D3AD8C74 85D28980
	buffer_load_dwordx4 v[116:119], v190, s[16:19], 0 offen offset:1024// 00000000405C: E05C1400 800474BE
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[64:67], a[120:123], v175, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004064: D3AC6800 180357AF D3AD8C78 85E28184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[132:135], v[68:71], a[124:127], v175, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004074: D3AC5800 180357AF D3AD8C7C 85F28984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[120:123], v[72:75], a[144:147], v175, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004084: D3AC4000 000359AF D3AD8C90 86429178
	ds_read_b128 v[8:11], v185                                 // 000000004094: D9FE0000 080000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[120:123], v[76:79], a[148:151], v175, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000409C: D3AC7000 000359AF D3AD8C94 86529978
	buffer_load_dword v174, v193, s[24:27], 0 offen            // 0000000040AC: E0501000 8006AEC1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[72:75], a[152:155], v175, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000040B4: D3AC6800 000359AF D3AD8C98 8662917C
	ds_read_b128 v[16:19], v185 offset:64                      // 0000000040C4: D9FE0040 100000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[124:127], v[76:79], a[156:159], v175, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000040CC: D3AC7800 000359AF D3AD8C9C 8672997C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[128:131], v[80:83], a[144:147], v175, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000040DC: D3AC6000 180359AF D3AD8C90 8642A180
	ds_read_b128 v[12:15], v185 offset:512                     // 0000000040EC: D9FE0200 0C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[128:131], v[84:87], a[148:151], v175, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000040F4: D3AC3000 180359AF D3AD8C94 8652A980
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[132:135], v[80:83], a[152:155], v175, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004104: D3AC6800 180359AF D3AD8C98 8662A184
	ds_read_b128 v[20:23], v185 offset:576                     // 000000004114: D9FE0240 140000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[132:135], v[84:87], a[156:159], v175, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000411C: D3AC7800 180359AF D3AD8C9C 8672A984
	ds_read_b32 v168, v188 offset:2048                         // 00000000412C: D86C0800 A80000BC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[88:91], a[176:179], v175, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004134: D3AC6000 00035BAF D3AD8CB0 86C2B178
	ds_read_b128 v[24:27], v185 offset:4224                    // 000000004144: D9FE1080 180000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[92:95], a[180:183], v175, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000414C: D3AC7000 00035BAF D3AD8CB4 86D2B978
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[88:91], a[184:187], v175, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000415C: D3AC2800 00035BAF D3AD8CB8 86E2B17C
	ds_read_b128 v[32:35], v185 offset:4288                    // 00000000416C: D9FE10C0 200000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[92:95], a[188:191], v175, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004174: D3AC7800 00035BAF D3AD8CBC 86F2B97C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[96:99], a[176:179], v175, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004184: D3AC6000 18035BAF D3AD8CB0 86C2C180
	ds_read_b128 v[28:31], v185 offset:4736                    // 000000004194: D9FE1280 1C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[100:103], a[180:183], v175, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000419C: D3AC7000 18035BAF D3AD8CB4 86D2C980
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[132:135], v[96:99], a[184:187], v175, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000041AC: D3AC6800 18035BAF D3AD8CB8 86E2C184
	ds_read_b128 v[36:39], v185 offset:4800                    // 0000000041BC: D9FE12C0 240000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[132:135], v[100:103], a[188:191], v175, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000041C4: D3AC7800 18035BAF D3AD8CBC 86F2C984
	ds_read_b32 v169, v188 offset:2304                         // 0000000041D4: D86C0900 A90000BC
	s_cbranch_scc0 label_0BB4                                  // 0000000041DC: BF84063C
	s_waitcnt vmcnt(18)                                        // 0000000041E0: BF8C4F72
	s_barrier                                                  // 0000000041E4: BF8A0000
	s_waitcnt lgkmcnt(5)                                       // 0000000041E8: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v176, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000041EC: D3AC6000 000351B0 D3AD8C00 84021188
	ds_read_b128 v[40:43], v185 offset:8448                    // 0000000041FC: D9FE2100 280000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v176, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004204: D3AC7000 000351B0 D3AD8C04 84121988
	buffer_load_dwordx4 v[120:123], v191, s[16:19], 0 offen    // 000000004214: E05C1000 800478BF
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[8:11], a[8:11], v176, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000421C: D3AC6800 000351B0 D3AD8C08 8422118C
	ds_read_b128 v[48:51], v185 offset:8512                    // 00000000422C: D9FE2140 300000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[140:143], v[12:15], a[12:15], v176, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004234: D3AC7800 000351B0 D3AD8C0C 8432198C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:147], v[16:19], a[0:3], v176, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004244: D3AC6000 180351B0 D3AD8C00 84022190
	ds_read_b128 v[44:47], v185 offset:8960                    // 000000004254: D9FE2300 2C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:147], v[20:23], a[4:7], v176, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000425C: D3AC7000 180351B0 D3AD8C04 84122990
	buffer_load_dwordx4 v[124:127], v192, s[16:19], 0 offen    // 00000000426C: E05C1000 80047CC0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:151], v[16:19], a[8:11], v176, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004274: D3AC6800 180351B0 D3AD8C08 84222194
	ds_read_b128 v[52:55], v185 offset:9024                    // 000000004284: D9FE2340 340000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[20:23], a[12:15], v176, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000428C: D3AC7800 180351B0 D3AD8C0C 84322994
	ds_read_b32 v170, v188 offset:2560                         // 00000000429C: D86C0A00 AA0000BC
	s_waitcnt lgkmcnt(5)                                       // 0000000042A4: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[136:139], v[24:27], a[32:35], v176, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000042A8: D3AC6000 000353B0 D3AD8C20 84823188
	ds_read_b128 v[56:59], v185 offset:12672                   // 0000000042B8: D9FE3180 380000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:139], v[28:31], a[36:39], v176, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000042C0: D3AC7000 000353B0 D3AD8C24 84923988
	buffer_load_dwordx4 v[128:131], v191, s[16:19], 0 offen offset:1024// 0000000042D0: E05C1400 800480BF
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[24:27], a[40:43], v176, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000042D8: D3AC6800 000353B0 D3AD8C28 84A2318C
	ds_read_b128 v[64:67], v185 offset:12736                   // 0000000042E8: D9FE31C0 400000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[28:31], a[44:47], v176, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000042F0: D3AC7800 000353B0 D3AD8C2C 84B2398C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[32:35], a[32:35], v176, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004300: D3AC6000 180353B0 D3AD8C20 84824190
	ds_read_b128 v[60:63], v185 offset:13184                   // 000000004310: D9FE3380 3C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[36:39], a[36:39], v176, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004318: D3AC7000 180353B0 D3AD8C24 84924990
	buffer_load_dwordx4 v[132:135], v192, s[16:19], 0 offen offset:1024// 000000004328: E05C1400 800484C0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:151], v[32:35], a[40:43], v176, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004330: D3AC6800 180353B0 D3AD8C28 84A24194
	ds_read_b128 v[68:71], v185 offset:13248                   // 000000004340: D9FE33C0 440000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[148:151], v[36:39], a[44:47], v176, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004348: D3AC7800 180353B0 D3AD8C2C 84B24994
	ds_read_b32 v171, v188 offset:2816                         // 000000004358: D86C0B00 AB0000BC
	s_waitcnt lgkmcnt(5)                                       // 000000004360: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[40:43], a[64:67], v176, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004364: D3AC6000 000355B0 D3AD8C40 85025188
	ds_read_b128 v[72:75], v185 offset:16896                   // 000000004374: D9FE4200 480000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[44:47], a[68:71], v176, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000437C: D3AC7000 000355B0 D3AD8C44 85125988
	buffer_load_dword v175, v194, s[24:27], 0 offen            // 00000000438C: E0501000 8006AFC2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[40:43], a[72:75], v176, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004394: D3AC6800 000355B0 D3AD8C48 8522518C
	s_add_u32 s63, 0x200, s60                                  // 0000000043A4: 803F3CFF 00000200
	ds_read_b128 v[80:83], v185 offset:16960                   // 0000000043AC: D9FE4240 500000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[44:47], a[76:79], v176, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000043B4: D3AC7800 000355B0 D3AD8C4C 8532598C
	s_cmp_lt_u32 s63, s61                                      // 0000000043C4: BF0A3D3F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[48:51], a[64:67], v176, v170 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000043C8: D3AC6000 180355B0 D3AD8C40 85026190
	s_cselect_b32 s67, s67, 0                                  // 0000000043D8: 85438043
	ds_read_b128 v[76:79], v185 offset:17408                   // 0000000043DC: D9FE4400 4C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[52:55], a[68:71], v176, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000043E4: D3AC7000 180355B0 D3AD8C44 85126990
	s_cselect_b32 s69, s69, 0                                  // 0000000043F4: 85458045
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:151], v[48:51], a[72:75], v176, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000043F8: D3AC6800 180355B0 D3AD8C48 85226194
	s_add_u32 s16, s16, s67                                    // 000000004408: 80104310
	ds_read_b128 v[84:87], v185 offset:17472                   // 00000000440C: D9FE4440 540000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[52:55], a[76:79], v176, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004414: D3AC7800 180355B0 D3AD8C4C 85326994
	s_addc_u32 s17, 0, s17                                     // 000000004424: 82111180
	s_sub_u32 s18, s18, s67                                    // 000000004428: 80924312
	ds_read_b32 v172, v188 offset:3072                         // 00000000442C: D86C0C00 AC0000BC
	s_waitcnt lgkmcnt(5)                                       // 000000004434: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[56:59], a[96:99], v176, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004438: D3AC6000 000357B0 D3AD8C60 85827188
	s_add_u32 s24, s24, s69                                    // 000000004448: 80184518
	ds_read_b128 v[88:91], v185 offset:21120                   // 00000000444C: D9FE5280 580000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[136:139], v[60:63], a[100:103], v176, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004454: D3AC7000 000357B0 D3AD8C64 85927988
	s_addc_u32 s25, 0, s25                                     // 000000004464: 82191980
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[56:59], a[104:107], v176, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004468: D3AC6800 000357B0 D3AD8C68 85A2718C
	s_sub_u32 s26, s26, s69                                    // 000000004478: 809A451A
	ds_read_b128 v[96:99], v185 offset:21184                   // 00000000447C: D9FE52C0 600000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[140:143], v[60:63], a[108:111], v176, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004484: D3AC7800 000357B0 D3AD8C6C 85B2798C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[144:147], v[64:67], a[96:99], v176, v171 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004494: D3AC6000 180357B0 D3AD8C60 85828190
	ds_read_b128 v[92:95], v185 offset:21632                   // 0000000044A4: D9FE5480 5C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:147], v[68:71], a[100:103], v176, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000044AC: D3AC7000 180357B0 D3AD8C64 85928990
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[64:67], a[104:107], v176, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000044BC: D3AC6800 180357B0 D3AD8C68 85A28194
	ds_read_b128 v[100:103], v185 offset:21696                 // 0000000044CC: D9FE54C0 640000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:151], v[68:71], a[108:111], v176, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000044D4: D3AC7800 180357B0 D3AD8C6C 85B28994
	ds_read_b32 v173, v188 offset:3328                         // 0000000044E4: D86C0D00 AD0000BC
	s_barrier                                                  // 0000000044EC: BF8A0000
	s_waitcnt lgkmcnt(5)                                       // 0000000044F0: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[136:139], v[72:75], a[128:131], v176, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000044F4: D3AC6000 000359B0 D3AD8C80 86029188
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[136:139], v[76:79], a[132:135], v176, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004504: D3AC7000 000359B0 D3AD8C84 86129988
	s_add_u32 m0, 0x800, s65                                   // 000000004514: 807C41FF 00000800
	buffer_load_dword v186, s[20:23], 0 offen lds              // 00000000451C: E0511000 800500BA
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[140:143], v[72:75], a[136:139], v176, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004524: D3AC6800 000359B0 D3AD8C88 8622918C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[140:143], v[76:79], a[140:143], v176, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004534: D3AC7800 000359B0 D3AD8C8C 8632998C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[144:147], v[80:83], a[128:131], v176, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004544: D3AC6000 180359B0 D3AD8C80 8602A190
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[144:147], v[84:87], a[132:135], v176, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004554: D3AC7000 180359B0 D3AD8C84 8612A990
	s_add_u32 m0, 0xc00, s65                                   // 000000004564: 807C41FF 00000C00
	buffer_load_dword v187, s[20:23], 0 offen lds              // 00000000456C: E0511000 800500BB
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:151], v[80:83], a[136:139], v176, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004574: D3AC6800 180359B0 D3AD8C88 8622A194
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[148:151], v[84:87], a[140:143], v176, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004584: D3AC7800 180359B0 D3AD8C8C 8632A994
	s_waitcnt lgkmcnt(0)                                       // 000000004594: BF8CC07F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[136:139], v[88:91], a[160:163], v176, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004598: D3AC6000 00035BB0 D3AD8CA0 8682B188
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[136:139], v[92:95], a[164:167], v176, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000045A8: D3AC7000 00035BB0 D3AD8CA4 8692B988
	s_add_u32 m0, 0x6300, s64                                  // 0000000045B8: 807C40FF 00006300
	buffer_load_dwordx4 v178, s[12:15], 0 offen lds            // 0000000045C0: E05D1000 800300B2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[88:91], a[168:171], v176, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000045C8: D3AC6800 00035BB0 D3AD8CA8 86A2B18C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[92:95], a[172:175], v176, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000045D8: D3AC7800 00035BB0 D3AD8CAC 86B2B98C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[144:147], v[96:99], a[160:163], v176, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000045E8: D3AC6000 18035BB0 D3AD8CA0 8682C190
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[144:147], v[100:103], a[164:167], v176, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000045F8: D3AC7000 18035BB0 D3AD8CA4 8692C990
	s_add_u32 m0, 0x7380, s64                                  // 000000004608: 807C40FF 00007380
	buffer_load_dwordx4 v179, s[12:15], 0 offen lds            // 000000004610: E05D1000 800300B3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:151], v[96:99], a[168:171], v176, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004618: D3AC6800 18035BB0 D3AD8CA8 86A2C194
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[148:151], v[100:103], a[172:175], v176, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004628: D3AC7800 18035BB0 D3AD8CAC 86B2C994
	s_waitcnt vmcnt(18)                                        // 000000004638: BF8C4F72
	s_barrier                                                  // 00000000463C: BF8A0000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[8:11], a[16:19], v177, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004640: D3AC6000 000351B1 D3AD8C10 84421198
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[152:155], v[12:15], a[20:23], v177, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004650: D3AC7000 000351B1 D3AD8C14 84521998
	s_add_u32 m0, 0x8400, s64                                  // 000000004660: 807C40FF 00008400
	buffer_load_dwordx4 v180, s[12:15], 0 offen lds            // 000000004668: E05D1000 800300B4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[156:159], v[8:11], a[24:27], v177, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004670: D3AC6800 000351B1 D3AD8C18 8462119C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[156:159], v[12:15], a[28:31], v177, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004680: D3AC7800 000351B1 D3AD8C1C 8472199C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:163], v[16:19], a[16:19], v177, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004690: D3AC6000 180351B1 D3AD8C10 844221A0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[160:163], v[20:23], a[20:23], v177, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000046A0: D3AC7000 180351B1 D3AD8C14 845229A0
	s_add_u32 m0, 0x9480, s64                                  // 0000000046B0: 807C40FF 00009480
	buffer_load_dwordx4 v181, s[12:15], 0 offen lds            // 0000000046B8: E05D1000 800300B5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[164:167], v[16:19], a[24:27], v177, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000046C0: D3AC6800 180351B1 D3AD8C18 846221A4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[164:167], v[20:23], a[28:31], v177, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000046D0: D3AC7800 180351B1 D3AD8C1C 847229A4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[24:27], a[48:51], v177, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000046E0: D3AC6000 000353B1 D3AD8C30 84C23198
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[152:155], v[28:31], a[52:55], v177, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000046F0: D3AC7000 000353B1 D3AD8C34 84D23998
	s_add_u32 m0, 0xa500, s64                                  // 000000004700: 807C40FF 0000A500
	buffer_load_dwordx4 v182, s[12:15], 0 offen lds            // 000000004708: E05D1000 800300B6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[24:27], a[56:59], v177, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004710: D3AC6800 000353B1 D3AD8C38 84E2319C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:159], v[28:31], a[60:63], v177, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004720: D3AC7800 000353B1 D3AD8C3C 84F2399C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:163], v[32:35], a[48:51], v177, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004730: D3AC6000 180353B1 D3AD8C30 84C241A0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[36:39], a[52:55], v177, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004740: D3AC7000 180353B1 D3AD8C34 84D249A0
	s_add_u32 m0, 0xb580, s64                                  // 000000004750: 807C40FF 0000B580
	buffer_load_dwordx4 v183, s[12:15], 0 offen lds            // 000000004758: E05D1000 800300B7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[164:167], v[32:35], a[56:59], v177, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004760: D3AC6800 180353B1 D3AD8C38 84E241A4
	s_add_u32 s62, 0x300, s60                                  // 000000004770: 803E3CFF 00000300
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[164:167], v[36:39], a[60:63], v177, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004778: D3AC7800 180353B1 D3AD8C3C 84F249A4
	s_cmp_lt_u32 s62, s61                                      // 000000004788: BF0A3D3E
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[152:155], v[40:43], a[80:83], v177, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000478C: D3AC6000 000355B1 D3AD8C50 85425198
	s_cselect_b32 s66, s66, 0                                  // 00000000479C: 85428042
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[44:47], a[84:87], v177, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000047A0: D3AC7000 000355B1 D3AD8C54 85525998
	s_cselect_b32 s68, s68, 0                                  // 0000000047B0: 85448044
	buffer_load_dwordx4 v[136:139], v189, s[16:19], 0 offen    // 0000000047B4: E05C1000 800488BD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[40:43], a[88:91], v177, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000047BC: D3AC6800 000355B1 D3AD8C58 8562519C
	s_add_u32 s12, s12, s66                                    // 0000000047CC: 800C420C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:159], v[44:47], a[92:95], v177, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000047D0: D3AC7800 000355B1 D3AD8C5C 8572599C
	s_addc_u32 s13, 0, s13                                     // 0000000047E0: 820D0D80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[48:51], a[80:83], v177, v170 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000047E4: D3AC6000 180355B1 D3AD8C50 854261A0
	s_sub_u32 s14, s14, s66                                    // 0000000047F4: 808E420E
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[52:55], a[84:87], v177, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000047F8: D3AC7000 180355B1 D3AD8C54 855269A0
	s_add_u32 s20, s20, s68                                    // 000000004808: 80144414
	buffer_load_dwordx4 v[140:143], v190, s[16:19], 0 offen    // 00000000480C: E05C1000 80048CBE
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[164:167], v[48:51], a[88:91], v177, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004814: D3AC6800 180355B1 D3AD8C58 856261A4
	s_addc_u32 s21, 0, s21                                     // 000000004824: 82151580
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[164:167], v[52:55], a[92:95], v177, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004828: D3AC7800 180355B1 D3AD8C5C 857269A4
	s_sub_u32 s22, s22, s68                                    // 000000004838: 80964416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[152:155], v[56:59], a[112:115], v177, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000483C: D3AC6000 000357B1 D3AD8C70 85C27198
	s_addk_i32 s60, 0x100                                      // 00000000484C: B73C0100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[152:155], v[60:63], a[116:119], v177, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004850: D3AC7000 000357B1 D3AD8C74 85D27998
	s_cmp_lt_i32 s60, s61                                      // 000000004860: BF043D3C
	buffer_load_dwordx4 v[144:147], v189, s[16:19], 0 offen offset:1024// 000000004864: E05C1400 800490BD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[156:159], v[56:59], a[120:123], v177, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000486C: D3AC6800 000357B1 D3AD8C78 85E2719C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[156:159], v[60:63], a[124:127], v177, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000487C: D3AC7800 000357B1 D3AD8C7C 85F2799C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:163], v[64:67], a[112:115], v177, v171 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000488C: D3AC6000 180357B1 D3AD8C70 85C281A0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[160:163], v[68:71], a[116:119], v177, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000489C: D3AC7000 180357B1 D3AD8C74 85D289A0
	buffer_load_dwordx4 v[148:151], v190, s[16:19], 0 offen offset:1024// 0000000048AC: E05C1400 800494BE
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:167], v[64:67], a[120:123], v177, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000048B4: D3AC6800 180357B1 D3AD8C78 85E281A4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[68:71], a[124:127], v177, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000048C4: D3AC7800 180357B1 D3AD8C7C 85F289A4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[152:155], v[72:75], a[144:147], v177, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000048D4: D3AC6000 000359B1 D3AD8C90 86429198
	ds_read_b128 v[8:11], v184                                 // 0000000048E4: D9FE0000 080000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[152:155], v[76:79], a[148:151], v177, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000048EC: D3AC7000 000359B1 D3AD8C94 86529998
	buffer_load_dword v176, v193, s[24:27], 0 offen            // 0000000048FC: E0501000 8006B0C1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[156:159], v[72:75], a[152:155], v177, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004904: D3AC6800 000359B1 D3AD8C98 8662919C
	ds_read_b128 v[16:19], v184 offset:64                      // 000000004914: D9FE0040 100000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[156:159], v[76:79], a[156:159], v177, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000491C: D3AC7800 000359B1 D3AD8C9C 8672999C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[160:163], v[80:83], a[144:147], v177, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000492C: D3AC6000 180359B1 D3AD8C90 8642A1A0
	ds_read_b128 v[12:15], v184 offset:512                     // 00000000493C: D9FE0200 0C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[160:163], v[84:87], a[148:151], v177, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004944: D3AC7000 180359B1 D3AD8C94 8652A9A0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[164:167], v[80:83], a[152:155], v177, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004954: D3AC6800 180359B1 D3AD8C98 8662A1A4
	ds_read_b128 v[20:23], v184 offset:576                     // 000000004964: D9FE0240 140000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[164:167], v[84:87], a[156:159], v177, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000496C: D3AC7800 180359B1 D3AD8C9C 8672A9A4
	ds_read_b32 v168, v188                                     // 00000000497C: D86C0000 A80000BC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[152:155], v[88:91], a[176:179], v177, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004984: D3AC6000 00035BB1 D3AD8CB0 86C2B198
	ds_read_b128 v[24:27], v184 offset:4224                    // 000000004994: D9FE1080 180000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[152:155], v[92:95], a[180:183], v177, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000499C: D3AC7000 00035BB1 D3AD8CB4 86D2B998
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[156:159], v[88:91], a[184:187], v177, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000049AC: D3AC6800 00035BB1 D3AD8CB8 86E2B19C
	ds_read_b128 v[32:35], v184 offset:4288                    // 0000000049BC: D9FE10C0 200000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[156:159], v[92:95], a[188:191], v177, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000049C4: D3AC7800 00035BB1 D3AD8CBC 86F2B99C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:163], v[96:99], a[176:179], v177, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000049D4: D3AC6000 18035BB1 D3AD8CB0 86C2C1A0
	ds_read_b128 v[28:31], v184 offset:4736                    // 0000000049E4: D9FE1280 1C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[160:163], v[100:103], a[180:183], v177, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000049EC: D3AC7000 18035BB1 D3AD8CB4 86D2C9A0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[164:167], v[96:99], a[184:187], v177, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000049FC: D3AC6800 18035BB1 D3AD8CB8 86E2C1A4
	ds_read_b128 v[36:39], v184 offset:4800                    // 000000004A0C: D9FE12C0 240000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[164:167], v[100:103], a[188:191], v177, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004A14: D3AC7800 18035BB1 D3AD8CBC 86F2C9A4
	ds_read_b32 v169, v188 offset:256                          // 000000004A24: D86C0100 A90000BC
	s_cbranch_scc0 label_0BB4                                  // 000000004A2C: BF840428
	s_branch label_0366                                        // 000000004A30: BF82FBD9

0000000000004a34 <label_078D>:
	s_waitcnt vmcnt(18)                                        // 000000004A34: BF8C4F72
	s_barrier                                                  // 000000004A38: BF8A0000
	s_waitcnt lgkmcnt(5)                                       // 000000004A3C: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[8:11], a[0:3], v174, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004A40: D3AC6000 000351AE D3AD8C00 84021168
	buffer_load_dwordx4 v[152:155], v191, s[16:19], 0 offen    // 000000004A50: E05C1000 800498BF
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[12:15], a[4:7], v174, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004A58: D3AC7000 000351AE D3AD8C04 84121968
	ds_read_b128 v[40:43], v184 offset:8448                    // 000000004A68: D9FE2100 280000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[8:11], a[8:11], v174, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004A70: D3AC6800 000351AE D3AD8C08 8422116C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[12:15], a[12:15], v174, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004A80: D3AC7800 000351AE D3AD8C0C 8432196C
	ds_read_b128 v[48:51], v184 offset:8512                    // 000000004A90: D9FE2140 300000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:115], v[16:19], a[0:3], v174, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004A98: D3AC6000 180351AE D3AD8C00 84022170
	buffer_load_dwordx4 v[156:159], v192, s[16:19], 0 offen    // 000000004AA8: E05C1000 80049CC0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:115], v[20:23], a[4:7], v174, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004AB0: D3AC7000 180351AE D3AD8C04 84122970
	ds_read_b128 v[44:47], v184 offset:8960                    // 000000004AC0: D9FE2300 2C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[116:119], v[16:19], a[8:11], v174, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004AC8: D3AC6800 180351AE D3AD8C08 84222174
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[20:23], a[12:15], v174, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004AD8: D3AC7800 180351AE D3AD8C0C 84322974
	ds_read_b128 v[52:55], v184 offset:9024                    // 000000004AE8: D9FE2340 340000B8
	ds_read_b32 v170, v188 offset:512                          // 000000004AF0: D86C0200 AA0000BC
	s_waitcnt lgkmcnt(5)                                       // 000000004AF8: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[24:27], a[32:35], v174, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004AFC: D3AC6000 000353AE D3AD8C20 84823168
	buffer_load_dwordx4 v[160:163], v191, s[16:19], 0 offen offset:1024// 000000004B0C: E05C1400 8004A0BF
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[28:31], a[36:39], v174, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004B14: D3AC7000 000353AE D3AD8C24 84923968
	ds_read_b128 v[56:59], v184 offset:12672                   // 000000004B24: D9FE3180 380000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[24:27], a[40:43], v174, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004B2C: D3AC6800 000353AE D3AD8C28 84A2316C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[28:31], a[44:47], v174, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004B3C: D3AC7800 000353AE D3AD8C2C 84B2396C
	ds_read_b128 v[64:67], v184 offset:12736                   // 000000004B4C: D9FE31C0 400000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[32:35], a[32:35], v174, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004B54: D3AC6000 180353AE D3AD8C20 84824170
	buffer_load_dwordx4 v[164:167], v192, s[16:19], 0 offen offset:1024// 000000004B64: E05C1400 8004A4C0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[36:39], a[36:39], v174, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004B6C: D3AC7000 180353AE D3AD8C24 84924970
	ds_read_b128 v[60:63], v184 offset:13184                   // 000000004B7C: D9FE3380 3C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[32:35], a[40:43], v174, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004B84: D3AC6800 180353AE D3AD8C28 84A24174
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[36:39], a[44:47], v174, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004B94: D3AC7800 180353AE D3AD8C2C 84B24974
	ds_read_b128 v[68:71], v184 offset:13248                   // 000000004BA4: D9FE33C0 440000B8
	ds_read_b32 v171, v188 offset:768                          // 000000004BAC: D86C0300 AB0000BC
	s_waitcnt lgkmcnt(5)                                       // 000000004BB4: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[104:107], v[40:43], a[64:67], v174, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004BB8: D3AC6000 000355AE D3AD8C40 85025168
	buffer_load_dword v177, v194, s[24:27], 0 offen            // 000000004BC8: E0501000 8006B1C2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[104:107], v[44:47], a[68:71], v174, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004BD0: D3AC7000 000355AE D3AD8C44 85125968
	s_add_u32 s63, 0x200, s60                                  // 000000004BE0: 803F3CFF 00000200
	ds_read_b128 v[72:75], v184 offset:16896                   // 000000004BE8: D9FE4200 480000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[40:43], a[72:75], v174, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004BF0: D3AC6800 000355AE D3AD8C48 8522516C
	s_cmp_lt_u32 s63, s61                                      // 000000004C00: BF0A3D3F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[44:47], a[76:79], v174, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004C04: D3AC1800 000355AE D3AD8C4C 8532596C
	s_cselect_b32 s67, s67, 0                                  // 000000004C14: 85438043
	ds_read_b128 v[80:83], v184 offset:16960                   // 000000004C18: D9FE4240 500000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:115], v[48:51], a[64:67], v174, v170 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004C20: D3AC6000 180355AE D3AD8C40 85026170
	s_cselect_b32 s69, s69, 0                                  // 000000004C30: 85458045
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[112:115], v[52:55], a[68:71], v174, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004C34: D3AC7000 180355AE D3AD8C44 85126970
	s_add_u32 s16, s16, s67                                    // 000000004C44: 80104310
	ds_read_b128 v[76:79], v184 offset:17408                   // 000000004C48: D9FE4400 4C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[116:119], v[48:51], a[72:75], v174, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004C50: D3AC6800 180355AE D3AD8C48 85226174
	s_addc_u32 s17, 0, s17                                     // 000000004C60: 82111180
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:119], v[52:55], a[76:79], v174, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004C64: D3AC7800 180355AE D3AD8C4C 85326974
	s_sub_u32 s18, s18, s67                                    // 000000004C74: 80924312
	ds_read_b128 v[84:87], v184 offset:17472                   // 000000004C78: D9FE4440 540000B8
	ds_read_b32 v172, v188 offset:1024                         // 000000004C80: D86C0400 AC0000BC
	s_waitcnt lgkmcnt(5)                                       // 000000004C88: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:107], v[56:59], a[96:99], v174, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004C8C: D3AC6000 000357AE D3AD8C60 85827168
	s_add_u32 s24, s24, s69                                    // 000000004C9C: 80184518
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:107], v[60:63], a[100:103], v174, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004CA0: D3AC7000 000357AE D3AD8C64 85927968
	s_addc_u32 s25, 0, s25                                     // 000000004CB0: 82191980
	ds_read_b128 v[88:91], v184 offset:21120                   // 000000004CB4: D9FE5280 580000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:111], v[56:59], a[104:107], v174, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004CBC: D3AC6800 000357AE D3AD8C68 85A2716C
	s_sub_u32 s26, s26, s69                                    // 000000004CCC: 809A451A
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[60:63], a[108:111], v174, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004CD0: D3AC7800 000357AE D3AD8C6C 85B2796C
	ds_read_b128 v[96:99], v184 offset:21184                   // 000000004CE0: D9FE52C0 600000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:115], v[64:67], a[96:99], v174, v171 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004CE8: D3AC6000 180357AE D3AD8C60 85828170
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[112:115], v[68:71], a[100:103], v174, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004CF8: D3AC7000 180357AE D3AD8C64 85928970
	ds_read_b128 v[92:95], v184 offset:21632                   // 000000004D08: D9FE5480 5C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[116:119], v[64:67], a[104:107], v174, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004D10: D3AC6800 180357AE D3AD8C68 85A28174
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:119], v[68:71], a[108:111], v174, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004D20: D3AC7800 180357AE D3AD8C6C 85B28974
	ds_read_b128 v[100:103], v184 offset:21696                 // 000000004D30: D9FE54C0 640000B8
	ds_read_b32 v173, v188 offset:1280                         // 000000004D38: D86C0500 AD0000BC
	s_barrier                                                  // 000000004D40: BF8A0000
	s_waitcnt lgkmcnt(5)                                       // 000000004D44: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[104:107], v[72:75], a[128:131], v174, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004D48: D3AC6000 000359AE D3AD8C80 86029168
	s_add_u32 m0, 0, s65                                       // 000000004D58: 807C4180
	buffer_load_dword v186, s[20:23], 0 offen lds              // 000000004D5C: E0511000 800500BA
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[104:107], v[76:79], a[132:135], v174, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004D64: D3AC7000 000359AE D3AD8C84 86129968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:111], v[72:75], a[136:139], v174, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004D74: D3AC6800 000359AE D3AD8C88 8622916C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[108:111], v[76:79], a[140:143], v174, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004D84: D3AC7800 000359AE D3AD8C8C 8632996C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:115], v[80:83], a[128:131], v174, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004D94: D3AC6000 180359AE D3AD8C80 8602A170
	s_add_u32 m0, 0x400, s65                                   // 000000004DA4: 807C41FF 00000400
	buffer_load_dword v187, s[20:23], 0 offen lds              // 000000004DAC: E0511000 800500BB
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[112:115], v[84:87], a[132:135], v174, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004DB4: D3AC7000 180359AE D3AD8C84 8612A970
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[116:119], v[80:83], a[136:139], v174, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004DC4: D3AC6800 180359AE D3AD8C88 8622A174
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[116:119], v[84:87], a[140:143], v174, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004DD4: D3AC7800 180359AE D3AD8C8C 8632A974
	s_waitcnt lgkmcnt(0)                                       // 000000004DE4: BF8CC07F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[104:107], v[88:91], a[160:163], v174, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004DE8: D3AC6000 00035BAE D3AD8CA0 8682B168
	s_add_u32 m0, 0, s64                                       // 000000004DF8: 807C4080
	buffer_load_dwordx4 v178, s[12:15], 0 offen lds            // 000000004DFC: E05D1000 800300B2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[104:107], v[92:95], a[164:167], v174, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004E04: D3AC7000 00035BAE D3AD8CA4 8692B968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:111], v[88:91], a[168:171], v174, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004E14: D3AC6800 00035BAE D3AD8CA8 86A2B16C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[108:111], v[92:95], a[172:175], v174, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004E24: D3AC7800 00035BAE D3AD8CAC 86B2B96C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[96:99], a[160:163], v174, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004E34: D3AC6000 18035BAE D3AD8CA0 8682C170
	s_add_u32 m0, 0x1080, s64                                  // 000000004E44: 807C40FF 00001080
	buffer_load_dwordx4 v179, s[12:15], 0 offen lds            // 000000004E4C: E05D1000 800300B3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[112:115], v[100:103], a[164:167], v174, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004E54: D3AC7000 18035BAE D3AD8CA4 8692C970
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[96:99], a[168:171], v174, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004E64: D3AC6800 18035BAE D3AD8CA8 86A2C174
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[100:103], a[172:175], v174, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004E74: D3AC7800 18035BAE D3AD8CAC 86B2C974
	s_waitcnt vmcnt(18)                                        // 000000004E84: BF8C4F72
	s_barrier                                                  // 000000004E88: BF8A0000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[8:11], a[16:19], v175, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004E8C: D3AC6000 000351AF D3AD8C10 84421178
	s_add_u32 m0, 0x2100, s64                                  // 000000004E9C: 807C40FF 00002100
	buffer_load_dwordx4 v180, s[12:15], 0 offen lds            // 000000004EA4: E05D1000 800300B4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[12:15], a[20:23], v175, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004EAC: D3AC7000 000351AF D3AD8C14 84521978
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[8:11], a[24:27], v175, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004EBC: D3AC6800 000351AF D3AD8C18 8462117C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[12:15], a[28:31], v175, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004ECC: D3AC7800 000351AF D3AD8C1C 8472197C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[16:19], a[16:19], v175, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004EDC: D3AC6000 180351AF D3AD8C10 84422180
	s_add_u32 m0, 0x3180, s64                                  // 000000004EEC: 807C40FF 00003180
	buffer_load_dwordx4 v181, s[12:15], 0 offen lds            // 000000004EF4: E05D1000 800300B5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[20:23], a[20:23], v175, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004EFC: D3AC7000 180351AF D3AD8C14 84522980
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[16:19], a[24:27], v175, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004F0C: D3AC6800 180351AF D3AD8C18 84622184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[20:23], a[28:31], v175, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004F1C: D3AC7800 180351AF D3AD8C1C 84722984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[24:27], a[48:51], v175, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004F2C: D3AC6000 000353AF D3AD8C30 84C23178
	s_add_u32 m0, 0x4200, s64                                  // 000000004F3C: 807C40FF 00004200
	buffer_load_dwordx4 v182, s[12:15], 0 offen lds            // 000000004F44: E05D1000 800300B6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[28:31], a[52:55], v175, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004F4C: D3AC7000 000353AF D3AD8C34 84D23978
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[24:27], a[56:59], v175, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004F5C: D3AC6800 000353AF D3AD8C38 84E2317C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[28:31], a[60:63], v175, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004F6C: D3AC7800 000353AF D3AD8C3C 84F2397C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[32:35], a[48:51], v175, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004F7C: D3AC6000 180353AF D3AD8C30 84C24180
	s_add_u32 m0, 0x5280, s64                                  // 000000004F8C: 807C40FF 00005280
	buffer_load_dwordx4 v183, s[12:15], 0 offen lds            // 000000004F94: E05D1000 800300B7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[36:39], a[52:55], v175, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004F9C: D3AC7000 180353AF D3AD8C34 84D24980
	s_add_u32 s62, 0x300, s60                                  // 000000004FAC: 803E3CFF 00000300
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[32:35], a[56:59], v175, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004FB4: D3AC6800 180353AF D3AD8C38 84E24184
	s_cmp_lt_u32 s62, s61                                      // 000000004FC4: BF0A3D3E
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[36:39], a[60:63], v175, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004FC8: D3AC7800 180353AF D3AD8C3C 84F24984
	s_cselect_b32 s66, s66, 0                                  // 000000004FD8: 85428042
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[40:43], a[80:83], v175, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004FDC: D3AC6000 000355AF D3AD8C50 85425178
	s_cselect_b32 s68, s68, 0                                  // 000000004FEC: 85448044
	buffer_load_dwordx4 v[104:107], v189, s[16:19], 0 offen    // 000000004FF0: E05C1000 800468BD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[44:47], a[84:87], v175, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004FF8: D3AC7000 000355AF D3AD8C54 85525978
	s_add_u32 s12, s12, s66                                    // 000000005008: 800C420C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[40:43], a[88:91], v175, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000500C: D3AC6800 000355AF D3AD8C58 8562517C
	s_addc_u32 s13, 0, s13                                     // 00000000501C: 820D0D80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[124:127], v[44:47], a[92:95], v175, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005020: D3AC7800 000355AF D3AD8C5C 8572597C
	s_sub_u32 s14, s14, s66                                    // 000000005030: 808E420E
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[128:131], v[48:51], a[80:83], v175, v170 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005034: D3AC6000 180355AF D3AD8C50 85426180
	s_add_u32 s20, s20, s68                                    // 000000005044: 80144414
	buffer_load_dwordx4 v[108:111], v190, s[16:19], 0 offen    // 000000005048: E05C1000 80046CBE
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[128:131], v[52:55], a[84:87], v175, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005050: D3AC7000 180355AF D3AD8C54 85526980
	s_addc_u32 s21, 0, s21                                     // 000000005060: 82151580
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[48:51], a[88:91], v175, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005064: D3AC6800 180355AF D3AD8C58 85626184
	s_sub_u32 s22, s22, s68                                    // 000000005074: 80964416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[52:55], a[92:95], v175, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005078: D3AC7800 180355AF D3AD8C5C 85726984
	s_addk_i32 s60, 0x100                                      // 000000005088: B73C0100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[56:59], a[112:115], v175, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000508C: D3AC6000 000357AF D3AD8C70 85C27178
	s_cmp_lt_i32 s60, s61                                      // 00000000509C: BF043D3C
	buffer_load_dwordx4 v[112:115], v189, s[16:19], 0 offen offset:1024// 0000000050A0: E05C1400 800470BD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[60:63], a[116:119], v175, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000050A8: D3AC7000 000357AF D3AD8C74 85D27978
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[56:59], a[120:123], v175, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000050B8: D3AC6800 000357AF D3AD8C78 85E2717C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:127], v[60:63], a[124:127], v175, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000050C8: D3AC7800 000357AF D3AD8C7C 85F2797C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[64:67], a[112:115], v175, v171 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000050D8: D3AC6000 180357AF D3AD8C70 85C28180
	buffer_load_dwordx4 v[116:119], v190, s[16:19], 0 offen offset:1024// 0000000050E8: E05C1400 800474BE
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[68:71], a[116:119], v175, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000050F0: D3AC7000 180357AF D3AD8C74 85D28980
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[64:67], a[120:123], v175, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005100: D3AC6800 180357AF D3AD8C78 85E28184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[132:135], v[68:71], a[124:127], v175, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005110: D3AC7800 180357AF D3AD8C7C 85F28984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[120:123], v[72:75], a[144:147], v175, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005120: D3AC6000 000359AF D3AD8C90 86429178
	buffer_load_dword v174, v193, s[24:27], 0 offen            // 000000005130: E0501000 8006AEC1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[120:123], v[76:79], a[148:151], v175, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005138: D3AC7000 000359AF D3AD8C94 86529978
	ds_read_b128 v[8:11], v185                                 // 000000005148: D9FE0000 080000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[72:75], a[152:155], v175, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005150: D3AC6800 000359AF D3AD8C98 8662917C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[124:127], v[76:79], a[156:159], v175, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005160: D3AC7800 000359AF D3AD8C9C 8672997C
	ds_read_b128 v[16:19], v185 offset:64                      // 000000005170: D9FE0040 100000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[128:131], v[80:83], a[144:147], v175, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005178: D3AC6000 180359AF D3AD8C90 8642A180
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[128:131], v[84:87], a[148:151], v175, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005188: D3AC7000 180359AF D3AD8C94 8652A980
	ds_read_b128 v[12:15], v185 offset:512                     // 000000005198: D9FE0200 0C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[132:135], v[80:83], a[152:155], v175, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000051A0: D3AC6800 180359AF D3AD8C98 8662A184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[132:135], v[84:87], a[156:159], v175, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000051B0: D3AC7800 180359AF D3AD8C9C 8672A984
	ds_read_b128 v[20:23], v185 offset:576                     // 0000000051C0: D9FE0240 140000B9
	ds_read_b32 v168, v188 offset:2048                         // 0000000051C8: D86C0800 A80000BC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[88:91], a[176:179], v175, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000051D0: D3AC6000 00035BAF D3AD8CB0 86C2B178
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[92:95], a[180:183], v175, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000051E0: D3AC7000 00035BAF D3AD8CB4 86D2B978
	ds_read_b128 v[24:27], v185 offset:4224                    // 0000000051F0: D9FE1080 180000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[88:91], a[184:187], v175, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000051F8: D3AC6800 00035BAF D3AD8CB8 86E2B17C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[92:95], a[188:191], v175, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005208: D3AC7800 00035BAF D3AD8CBC 86F2B97C
	ds_read_b128 v[32:35], v185 offset:4288                    // 000000005218: D9FE10C0 200000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[96:99], a[176:179], v175, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005220: D3AC6000 18035BAF D3AD8CB0 86C2C180
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[100:103], a[180:183], v175, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005230: D3AC7000 18035BAF D3AD8CB4 86D2C980
	ds_read_b128 v[28:31], v185 offset:4736                    // 000000005240: D9FE1280 1C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[132:135], v[96:99], a[184:187], v175, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005248: D3AC6800 18035BAF D3AD8CB8 86E2C184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[132:135], v[100:103], a[188:191], v175, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005258: D3AC7800 18035BAF D3AD8CBC 86F2C984
	ds_read_b128 v[36:39], v185 offset:4800                    // 000000005268: D9FE12C0 240000B9
	ds_read_b32 v169, v188 offset:2304                         // 000000005270: D86C0900 A90000BC
	s_cbranch_scc0 label_0BB4                                  // 000000005278: BF840215
	s_waitcnt vmcnt(18)                                        // 00000000527C: BF8C4F72
	s_barrier                                                  // 000000005280: BF8A0000
	s_waitcnt lgkmcnt(5)                                       // 000000005284: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v176, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005288: D3AC6000 000351B0 D3AD8C00 84021188
	buffer_load_dwordx4 v[120:123], v191, s[16:19], 0 offen    // 000000005298: E05C1000 800478BF
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v176, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000052A0: D3AC7000 000351B0 D3AD8C04 84121988
	ds_read_b128 v[40:43], v185 offset:8448                    // 0000000052B0: D9FE2100 280000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[8:11], a[8:11], v176, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000052B8: D3AC6800 000351B0 D3AD8C08 8422118C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[140:143], v[12:15], a[12:15], v176, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000052C8: D3AC7800 000351B0 D3AD8C0C 8432198C
	ds_read_b128 v[48:51], v185 offset:8512                    // 0000000052D8: D9FE2140 300000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:147], v[16:19], a[0:3], v176, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000052E0: D3AC6000 180351B0 D3AD8C00 84022190
	buffer_load_dwordx4 v[124:127], v192, s[16:19], 0 offen    // 0000000052F0: E05C1000 80047CC0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:147], v[20:23], a[4:7], v176, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000052F8: D3AC7000 180351B0 D3AD8C04 84122990
	ds_read_b128 v[44:47], v185 offset:8960                    // 000000005308: D9FE2300 2C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:151], v[16:19], a[8:11], v176, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005310: D3AC6800 180351B0 D3AD8C08 84222194
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[20:23], a[12:15], v176, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005320: D3AC7800 180351B0 D3AD8C0C 84322994
	ds_read_b128 v[52:55], v185 offset:9024                    // 000000005330: D9FE2340 340000B9
	ds_read_b32 v170, v188 offset:2560                         // 000000005338: D86C0A00 AA0000BC
	s_waitcnt lgkmcnt(5)                                       // 000000005340: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[136:139], v[24:27], a[32:35], v176, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005344: D3AC6000 000353B0 D3AD8C20 84823188
	buffer_load_dwordx4 v[128:131], v191, s[16:19], 0 offen offset:1024// 000000005354: E05C1400 800480BF
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:139], v[28:31], a[36:39], v176, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000535C: D3AC7000 000353B0 D3AD8C24 84923988
	ds_read_b128 v[56:59], v185 offset:12672                   // 00000000536C: D9FE3180 380000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[24:27], a[40:43], v176, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005374: D3AC6800 000353B0 D3AD8C28 84A2318C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[28:31], a[44:47], v176, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005384: D3AC7800 000353B0 D3AD8C2C 84B2398C
	ds_read_b128 v[64:67], v185 offset:12736                   // 000000005394: D9FE31C0 400000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[32:35], a[32:35], v176, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000539C: D3AC6000 180353B0 D3AD8C20 84824190
	buffer_load_dwordx4 v[132:135], v192, s[16:19], 0 offen offset:1024// 0000000053AC: E05C1400 800484C0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[36:39], a[36:39], v176, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000053B4: D3AC7000 180353B0 D3AD8C24 84924990
	ds_read_b128 v[60:63], v185 offset:13184                   // 0000000053C4: D9FE3380 3C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:151], v[32:35], a[40:43], v176, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000053CC: D3AC6800 180353B0 D3AD8C28 84A24194
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[148:151], v[36:39], a[44:47], v176, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000053DC: D3AC7800 180353B0 D3AD8C2C 84B24994
	ds_read_b128 v[68:71], v185 offset:13248                   // 0000000053EC: D9FE33C0 440000B9
	ds_read_b32 v171, v188 offset:2816                         // 0000000053F4: D86C0B00 AB0000BC
	s_waitcnt lgkmcnt(5)                                       // 0000000053FC: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[40:43], a[64:67], v176, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005400: D3AC6000 000355B0 D3AD8C40 85025188
	buffer_load_dword v175, v194, s[24:27], 0 offen            // 000000005410: E0501000 8006AFC2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[44:47], a[68:71], v176, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005418: D3AC7000 000355B0 D3AD8C44 85125988
	s_add_u32 s63, 0x200, s60                                  // 000000005428: 803F3CFF 00000200
	ds_read_b128 v[72:75], v185 offset:16896                   // 000000005430: D9FE4200 480000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[40:43], a[72:75], v176, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005438: D3AC6800 000355B0 D3AD8C48 8522518C
	s_cmp_lt_u32 s63, s61                                      // 000000005448: BF0A3D3F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[44:47], a[76:79], v176, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000544C: D3AC7800 000355B0 D3AD8C4C 8532598C
	s_cselect_b32 s67, s67, 0                                  // 00000000545C: 85438043
	ds_read_b128 v[80:83], v185 offset:16960                   // 000000005460: D9FE4240 500000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[48:51], a[64:67], v176, v170 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005468: D3AC6000 180355B0 D3AD8C40 85026190
	s_cselect_b32 s69, s69, 0                                  // 000000005478: 85458045
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[52:55], a[68:71], v176, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000547C: D3AC7000 180355B0 D3AD8C44 85126990
	s_add_u32 s16, s16, s67                                    // 00000000548C: 80104310
	ds_read_b128 v[76:79], v185 offset:17408                   // 000000005490: D9FE4400 4C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:151], v[48:51], a[72:75], v176, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005498: D3AC6800 180355B0 D3AD8C48 85226194
	s_addc_u32 s17, 0, s17                                     // 0000000054A8: 82111180
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[52:55], a[76:79], v176, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000054AC: D3AC7800 180355B0 D3AD8C4C 85326994
	s_sub_u32 s18, s18, s67                                    // 0000000054BC: 80924312
	ds_read_b128 v[84:87], v185 offset:17472                   // 0000000054C0: D9FE4440 540000B9
	ds_read_b32 v172, v188 offset:3072                         // 0000000054C8: D86C0C00 AC0000BC
	s_waitcnt lgkmcnt(5)                                       // 0000000054D0: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[56:59], a[96:99], v176, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000054D4: D3AC6000 000357B0 D3AD8C60 85827188
	s_add_u32 s24, s24, s69                                    // 0000000054E4: 80184518
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[136:139], v[60:63], a[100:103], v176, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000054E8: D3AC7000 000357B0 D3AD8C64 85927988
	s_addc_u32 s25, 0, s25                                     // 0000000054F8: 82191980
	ds_read_b128 v[88:91], v185 offset:21120                   // 0000000054FC: D9FE5280 580000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[56:59], a[104:107], v176, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005504: D3AC6800 000357B0 D3AD8C68 85A2718C
	s_sub_u32 s26, s26, s69                                    // 000000005514: 809A451A
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[140:143], v[60:63], a[108:111], v176, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005518: D3AC7800 000357B0 D3AD8C6C 85B2798C
	ds_read_b128 v[96:99], v185 offset:21184                   // 000000005528: D9FE52C0 600000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[144:147], v[64:67], a[96:99], v176, v171 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005530: D3AC6000 180357B0 D3AD8C60 85828190
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:147], v[68:71], a[100:103], v176, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005540: D3AC7000 180357B0 D3AD8C64 85928990
	ds_read_b128 v[92:95], v185 offset:21632                   // 000000005550: D9FE5480 5C0000B9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[64:67], a[104:107], v176, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005558: D3AC6800 180357B0 D3AD8C68 85A28194
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:151], v[68:71], a[108:111], v176, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005568: D3AC7800 180357B0 D3AD8C6C 85B28994
	ds_read_b128 v[100:103], v185 offset:21696                 // 000000005578: D9FE54C0 640000B9
	ds_read_b32 v173, v188 offset:3328                         // 000000005580: D86C0D00 AD0000BC
	s_barrier                                                  // 000000005588: BF8A0000
	s_waitcnt lgkmcnt(5)                                       // 00000000558C: BF8CC57F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[136:139], v[72:75], a[128:131], v176, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005590: D3AC6000 000359B0 D3AD8C80 86029188
	s_add_u32 m0, 0x800, s65                                   // 0000000055A0: 807C41FF 00000800
	buffer_load_dword v186, s[20:23], 0 offen lds              // 0000000055A8: E0511000 800500BA
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[136:139], v[76:79], a[132:135], v176, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000055B0: D3AC7000 000359B0 D3AD8C84 86129988
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[140:143], v[72:75], a[136:139], v176, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000055C0: D3AC6800 000359B0 D3AD8C88 8622918C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[140:143], v[76:79], a[140:143], v176, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000055D0: D3AC7800 000359B0 D3AD8C8C 8632998C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[144:147], v[80:83], a[128:131], v176, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000055E0: D3AC6000 180359B0 D3AD8C80 8602A190
	s_add_u32 m0, 0xc00, s65                                   // 0000000055F0: 807C41FF 00000C00
	buffer_load_dword v187, s[20:23], 0 offen lds              // 0000000055F8: E0511000 800500BB
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[144:147], v[84:87], a[132:135], v176, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005600: D3AC7000 180359B0 D3AD8C84 8612A990
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:151], v[80:83], a[136:139], v176, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005610: D3AC6800 180359B0 D3AD8C88 8622A194
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[148:151], v[84:87], a[140:143], v176, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005620: D3AC7800 180359B0 D3AD8C8C 8632A994
	s_waitcnt lgkmcnt(0)                                       // 000000005630: BF8CC07F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[136:139], v[88:91], a[160:163], v176, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005634: D3AC6000 00035BB0 D3AD8CA0 8682B188
	s_add_u32 m0, 0x6300, s64                                  // 000000005644: 807C40FF 00006300
	buffer_load_dwordx4 v178, s[12:15], 0 offen lds            // 00000000564C: E05D1000 800300B2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[136:139], v[92:95], a[164:167], v176, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005654: D3AC7000 00035BB0 D3AD8CA4 8692B988
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[88:91], a[168:171], v176, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005664: D3AC6800 00035BB0 D3AD8CA8 86A2B18C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[92:95], a[172:175], v176, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005674: D3AC7800 00035BB0 D3AD8CAC 86B2B98C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[144:147], v[96:99], a[160:163], v176, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005684: D3AC6000 18035BB0 D3AD8CA0 8682C190
	s_add_u32 m0, 0x7380, s64                                  // 000000005694: 807C40FF 00007380
	buffer_load_dwordx4 v179, s[12:15], 0 offen lds            // 00000000569C: E05D1000 800300B3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[144:147], v[100:103], a[164:167], v176, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000056A4: D3AC7000 18035BB0 D3AD8CA4 8692C990
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:151], v[96:99], a[168:171], v176, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000056B4: D3AC6800 18035BB0 D3AD8CA8 86A2C194
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[148:151], v[100:103], a[172:175], v176, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000056C4: D3AC7800 18035BB0 D3AD8CAC 86B2C994
	s_waitcnt vmcnt(18)                                        // 0000000056D4: BF8C4F72
	s_barrier                                                  // 0000000056D8: BF8A0000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[8:11], a[16:19], v177, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000056DC: D3AC6000 000351B1 D3AD8C10 84421198
	s_add_u32 m0, 0x8400, s64                                  // 0000000056EC: 807C40FF 00008400
	buffer_load_dwordx4 v180, s[12:15], 0 offen lds            // 0000000056F4: E05D1000 800300B4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[152:155], v[12:15], a[20:23], v177, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000056FC: D3AC7000 000351B1 D3AD8C14 84521998
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[156:159], v[8:11], a[24:27], v177, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000570C: D3AC6800 000351B1 D3AD8C18 8462119C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[156:159], v[12:15], a[28:31], v177, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000571C: D3AC7800 000351B1 D3AD8C1C 8472199C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:163], v[16:19], a[16:19], v177, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000572C: D3AC6000 180351B1 D3AD8C10 844221A0
	s_add_u32 m0, 0x9480, s64                                  // 00000000573C: 807C40FF 00009480
	buffer_load_dwordx4 v181, s[12:15], 0 offen lds            // 000000005744: E05D1000 800300B5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[160:163], v[20:23], a[20:23], v177, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000574C: D3AC7000 180351B1 D3AD8C14 845229A0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[164:167], v[16:19], a[24:27], v177, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000575C: D3AC6800 180351B1 D3AD8C18 846221A4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[164:167], v[20:23], a[28:31], v177, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000576C: D3AC7800 180351B1 D3AD8C1C 847229A4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[24:27], a[48:51], v177, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000577C: D3AC6000 000353B1 D3AD8C30 84C23198
	s_add_u32 m0, 0xa500, s64                                  // 00000000578C: 807C40FF 0000A500
	buffer_load_dwordx4 v182, s[12:15], 0 offen lds            // 000000005794: E05D1000 800300B6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[152:155], v[28:31], a[52:55], v177, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000579C: D3AC7000 000353B1 D3AD8C34 84D23998
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[24:27], a[56:59], v177, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000057AC: D3AC6800 000353B1 D3AD8C38 84E2319C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:159], v[28:31], a[60:63], v177, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000057BC: D3AC7800 000353B1 D3AD8C3C 84F2399C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:163], v[32:35], a[48:51], v177, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000057CC: D3AC6000 180353B1 D3AD8C30 84C241A0
	s_add_u32 m0, 0xb580, s64                                  // 0000000057DC: 807C40FF 0000B580
	buffer_load_dwordx4 v183, s[12:15], 0 offen lds            // 0000000057E4: E05D1000 800300B7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[36:39], a[52:55], v177, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000057EC: D3AC7000 180353B1 D3AD8C34 84D249A0
	s_add_u32 s62, 0x300, s60                                  // 0000000057FC: 803E3CFF 00000300
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[164:167], v[32:35], a[56:59], v177, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005804: D3AC6800 180353B1 D3AD8C38 84E241A4
	s_cmp_lt_u32 s62, s61                                      // 000000005814: BF0A3D3E
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[164:167], v[36:39], a[60:63], v177, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005818: D3AC7800 180353B1 D3AD8C3C 84F249A4
	s_cselect_b32 s66, s66, 0                                  // 000000005828: 85428042
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[152:155], v[40:43], a[80:83], v177, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000582C: D3AC6000 000355B1 D3AD8C50 85425198
	s_cselect_b32 s68, s68, 0                                  // 00000000583C: 85448044
	buffer_load_dwordx4 v[136:139], v189, s[16:19], 0 offen    // 000000005840: E05C1000 800488BD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[44:47], a[84:87], v177, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005848: D3AC7000 000355B1 D3AD8C54 85525998
	s_add_u32 s12, s12, s66                                    // 000000005858: 800C420C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[40:43], a[88:91], v177, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000585C: D3AC6800 000355B1 D3AD8C58 8562519C
	s_addc_u32 s13, 0, s13                                     // 00000000586C: 820D0D80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:159], v[44:47], a[92:95], v177, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005870: D3AC7800 000355B1 D3AD8C5C 8572599C
	s_sub_u32 s14, s14, s66                                    // 000000005880: 808E420E
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[48:51], a[80:83], v177, v170 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005884: D3AC6000 180355B1 D3AD8C50 854261A0
	s_add_u32 s20, s20, s68                                    // 000000005894: 80144414
	buffer_load_dwordx4 v[140:143], v190, s[16:19], 0 offen    // 000000005898: E05C1000 80048CBE
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[52:55], a[84:87], v177, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000058A0: D3AC7000 180355B1 D3AD8C54 855269A0
	s_addc_u32 s21, 0, s21                                     // 0000000058B0: 82151580
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[164:167], v[48:51], a[88:91], v177, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000058B4: D3AC6800 180355B1 D3AD8C58 856261A4
	s_sub_u32 s22, s22, s68                                    // 0000000058C4: 80964416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[164:167], v[52:55], a[92:95], v177, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000058C8: D3AC7800 180355B1 D3AD8C5C 857269A4
	s_addk_i32 s60, 0x100                                      // 0000000058D8: B73C0100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[152:155], v[56:59], a[112:115], v177, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000058DC: D3AC6000 000357B1 D3AD8C70 85C27198
	s_cmp_lt_i32 s60, s61                                      // 0000000058EC: BF043D3C
	buffer_load_dwordx4 v[144:147], v189, s[16:19], 0 offen offset:1024// 0000000058F0: E05C1400 800490BD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[152:155], v[60:63], a[116:119], v177, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000058F8: D3AC7000 000357B1 D3AD8C74 85D27998
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[156:159], v[56:59], a[120:123], v177, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005908: D3AC6800 000357B1 D3AD8C78 85E2719C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[156:159], v[60:63], a[124:127], v177, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005918: D3AC7800 000357B1 D3AD8C7C 85F2799C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:163], v[64:67], a[112:115], v177, v171 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005928: D3AC6000 180357B1 D3AD8C70 85C281A0
	buffer_load_dwordx4 v[148:151], v190, s[16:19], 0 offen offset:1024// 000000005938: E05C1400 800494BE
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[160:163], v[68:71], a[116:119], v177, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005940: D3AC7000 180357B1 D3AD8C74 85D289A0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:167], v[64:67], a[120:123], v177, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005950: D3AC6800 180357B1 D3AD8C78 85E281A4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[68:71], a[124:127], v177, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005960: D3AC7800 180357B1 D3AD8C7C 85F289A4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[152:155], v[72:75], a[144:147], v177, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005970: D3AC6000 000359B1 D3AD8C90 86429198
	buffer_load_dword v176, v193, s[24:27], 0 offen            // 000000005980: E0501000 8006B0C1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[152:155], v[76:79], a[148:151], v177, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005988: D3AC7000 000359B1 D3AD8C94 86529998
	ds_read_b128 v[8:11], v184                                 // 000000005998: D9FE0000 080000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[156:159], v[72:75], a[152:155], v177, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000059A0: D3AC6800 000359B1 D3AD8C98 8662919C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[156:159], v[76:79], a[156:159], v177, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000059B0: D3AC7800 000359B1 D3AD8C9C 8672999C
	ds_read_b128 v[16:19], v184 offset:64                      // 0000000059C0: D9FE0040 100000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[160:163], v[80:83], a[144:147], v177, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000059C8: D3AC6000 180359B1 D3AD8C90 8642A1A0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[160:163], v[84:87], a[148:151], v177, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000059D8: D3AC7000 180359B1 D3AD8C94 8652A9A0
	ds_read_b128 v[12:15], v184 offset:512                     // 0000000059E8: D9FE0200 0C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[164:167], v[80:83], a[152:155], v177, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000059F0: D3AC6800 180359B1 D3AD8C98 8662A1A4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[164:167], v[84:87], a[156:159], v177, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005A00: D3AC7800 180359B1 D3AD8C9C 8672A9A4
	ds_read_b128 v[20:23], v184 offset:576                     // 000000005A10: D9FE0240 140000B8
	ds_read_b32 v168, v188                                     // 000000005A18: D86C0000 A80000BC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[152:155], v[88:91], a[176:179], v177, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005A20: D3AC6000 00035BB1 D3AD8CB0 86C2B198
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[152:155], v[92:95], a[180:183], v177, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005A30: D3AC7000 00035BB1 D3AD8CB4 86D2B998
	ds_read_b128 v[24:27], v184 offset:4224                    // 000000005A40: D9FE1080 180000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[156:159], v[88:91], a[184:187], v177, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005A48: D3AC6800 00035BB1 D3AD8CB8 86E2B19C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[156:159], v[92:95], a[188:191], v177, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000005A58: D3AC7800 00035BB1 D3AD8CBC 86F2B99C
	ds_read_b128 v[32:35], v184 offset:4288                    // 000000005A68: D9FE10C0 200000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:163], v[96:99], a[176:179], v177, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005A70: D3AC6000 18035BB1 D3AD8CB0 86C2C1A0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[160:163], v[100:103], a[180:183], v177, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005A80: D3AC7000 18035BB1 D3AD8CB4 86D2C9A0
	ds_read_b128 v[28:31], v184 offset:4736                    // 000000005A90: D9FE1280 1C0000B8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[164:167], v[96:99], a[184:187], v177, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005A98: D3AC6800 18035BB1 D3AD8CB8 86E2C1A4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[164:167], v[100:103], a[188:191], v177, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000005AA8: D3AC7800 18035BB1 D3AD8CBC 86F2C9A4
	ds_read_b128 v[36:39], v184 offset:4800                    // 000000005AB8: D9FE12C0 240000B8
	ds_read_b32 v169, v188 offset:256                          // 000000005AC0: D86C0100 A90000BC
	s_cbranch_scc0 label_0BB4                                  // 000000005AC8: BF840001
	s_branch label_078D                                        // 000000005ACC: BF82FBD9

0000000000005ad0 <label_0BB4>:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 000000005AD0: BF8C0000
	s_barrier                                                  // 000000005AD4: BF8A0000
	v_lshrrev_b32_e32 v4, 5, v0                                // 000000005AD8: 20080085
	v_mul_i32_i24_e32 v4, 16, v4                               // 000000005ADC: 0C080890
	v_lshrrev_b32_e32 v5, 4, v0                                // 000000005AE0: 200A0084
	v_and_b32_e32 v5, 1, v5                                    // 000000005AE4: 260A0A81
	v_mul_i32_i24_e32 v5, 32, v5                               // 000000005AE8: 0C0A0AA0
	v_add_u32_e32 v4, v4, v5                                   // 000000005AEC: 68080B04
	v_and_b32_e32 v5, 15, v0                                   // 000000005AF0: 260A008F
	v_mul_i32_i24_e32 v5, 0x80, v5                             // 000000005AF4: 0C0A0AFF 00000080
	v_add_u32_e32 v4, v4, v5                                   // 000000005AFC: 68080B04
	s_mul_i32 s62, s46, 0x6000                                 // 000000005B00: 923EFF2E 00006000
	s_add_i32 s62, s62, 0                                      // 000000005B08: 813E803E
	v_add_i32 v4, v4, s62                                      // 000000005B0C: D29C0004 00007D04
	v_accvgpr_read_b32 v8, a0                                  // 000000005B14: D3D84008 18000100
	v_accvgpr_read_b32 v9, a1                                  // 000000005B1C: D3D84009 18000101
	v_accvgpr_read_b32 v10, a2                                 // 000000005B24: D3D8400A 18000102
	v_accvgpr_read_b32 v11, a3                                 // 000000005B2C: D3D8400B 18000103
	v_accvgpr_read_b32 v12, a8                                 // 000000005B34: D3D8400C 18000108
	v_accvgpr_read_b32 v13, a9                                 // 000000005B3C: D3D8400D 18000109
	v_accvgpr_read_b32 v14, a10                                // 000000005B44: D3D8400E 1800010A
	v_accvgpr_read_b32 v15, a11                                // 000000005B4C: D3D8400F 1800010B
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000005B54: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000005B5C: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000005B64: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000005B6C: D2680013 00021F0E
	s_nop 1                                                    // 000000005B74: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000005B78: 7E20B312
	s_nop 1                                                    // 000000005B7C: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000005B80: 7E22B313
	s_nop 1                                                    // 000000005B84: BF800001
	ds_write_b128 v4, v[16:19]                                 // 000000005B88: D9BE0000 00001004
	v_accvgpr_read_b32 v8, a16                                 // 000000005B90: D3D84008 18000110
	v_accvgpr_read_b32 v9, a17                                 // 000000005B98: D3D84009 18000111
	v_accvgpr_read_b32 v10, a18                                // 000000005BA0: D3D8400A 18000112
	v_accvgpr_read_b32 v11, a19                                // 000000005BA8: D3D8400B 18000113
	v_accvgpr_read_b32 v12, a24                                // 000000005BB0: D3D8400C 18000118
	v_accvgpr_read_b32 v13, a25                                // 000000005BB8: D3D8400D 18000119
	v_accvgpr_read_b32 v14, a26                                // 000000005BC0: D3D8400E 1800011A
	v_accvgpr_read_b32 v15, a27                                // 000000005BC8: D3D8400F 1800011B
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000005BD0: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000005BD8: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000005BE0: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000005BE8: D2680013 00021F0E
	s_nop 1                                                    // 000000005BF0: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000005BF4: 7E20B312
	s_nop 1                                                    // 000000005BF8: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000005BFC: 7E22B313
	s_nop 1                                                    // 000000005C00: BF800001
	ds_write_b128 v4, v[16:19] offset:64                       // 000000005C04: D9BE0040 00001004
	v_accvgpr_read_b32 v8, a4                                  // 000000005C0C: D3D84008 18000104
	v_accvgpr_read_b32 v9, a5                                  // 000000005C14: D3D84009 18000105
	v_accvgpr_read_b32 v10, a6                                 // 000000005C1C: D3D8400A 18000106
	v_accvgpr_read_b32 v11, a7                                 // 000000005C24: D3D8400B 18000107
	v_accvgpr_read_b32 v12, a12                                // 000000005C2C: D3D8400C 1800010C
	v_accvgpr_read_b32 v13, a13                                // 000000005C34: D3D8400D 1800010D
	v_accvgpr_read_b32 v14, a14                                // 000000005C3C: D3D8400E 1800010E
	v_accvgpr_read_b32 v15, a15                                // 000000005C44: D3D8400F 1800010F
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000005C4C: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000005C54: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000005C5C: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000005C64: D2680013 00021F0E
	s_nop 1                                                    // 000000005C6C: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000005C70: 7E20B312
	s_nop 1                                                    // 000000005C74: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000005C78: 7E22B313
	s_nop 1                                                    // 000000005C7C: BF800001
	ds_write_b128 v4, v[16:19] offset:2048                     // 000000005C80: D9BE0800 00001004
	v_accvgpr_read_b32 v8, a20                                 // 000000005C88: D3D84008 18000114
	v_accvgpr_read_b32 v9, a21                                 // 000000005C90: D3D84009 18000115
	v_accvgpr_read_b32 v10, a22                                // 000000005C98: D3D8400A 18000116
	v_accvgpr_read_b32 v11, a23                                // 000000005CA0: D3D8400B 18000117
	v_accvgpr_read_b32 v12, a28                                // 000000005CA8: D3D8400C 1800011C
	v_accvgpr_read_b32 v13, a29                                // 000000005CB0: D3D8400D 1800011D
	v_accvgpr_read_b32 v14, a30                                // 000000005CB8: D3D8400E 1800011E
	v_accvgpr_read_b32 v15, a31                                // 000000005CC0: D3D8400F 1800011F
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000005CC8: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000005CD0: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000005CD8: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000005CE0: D2680013 00021F0E
	s_nop 1                                                    // 000000005CE8: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000005CEC: 7E20B312
	s_nop 1                                                    // 000000005CF0: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000005CF4: 7E22B313
	s_nop 1                                                    // 000000005CF8: BF800001
	ds_write_b128 v4, v[16:19] offset:2112                     // 000000005CFC: D9BE0840 00001004
	v_accvgpr_read_b32 v8, a32                                 // 000000005D04: D3D84008 18000120
	v_accvgpr_read_b32 v9, a33                                 // 000000005D0C: D3D84009 18000121
	v_accvgpr_read_b32 v10, a34                                // 000000005D14: D3D8400A 18000122
	v_accvgpr_read_b32 v11, a35                                // 000000005D1C: D3D8400B 18000123
	v_accvgpr_read_b32 v12, a40                                // 000000005D24: D3D8400C 18000128
	v_accvgpr_read_b32 v13, a41                                // 000000005D2C: D3D8400D 18000129
	v_accvgpr_read_b32 v14, a42                                // 000000005D34: D3D8400E 1800012A
	v_accvgpr_read_b32 v15, a43                                // 000000005D3C: D3D8400F 1800012B
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000005D44: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000005D4C: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000005D54: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000005D5C: D2680013 00021F0E
	s_nop 1                                                    // 000000005D64: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000005D68: 7E20B312
	s_nop 1                                                    // 000000005D6C: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000005D70: 7E22B313
	s_nop 1                                                    // 000000005D74: BF800001
	ds_write_b128 v4, v[16:19] offset:4096                     // 000000005D78: D9BE1000 00001004
	v_accvgpr_read_b32 v8, a48                                 // 000000005D80: D3D84008 18000130
	v_accvgpr_read_b32 v9, a49                                 // 000000005D88: D3D84009 18000131
	v_accvgpr_read_b32 v10, a50                                // 000000005D90: D3D8400A 18000132
	v_accvgpr_read_b32 v11, a51                                // 000000005D98: D3D8400B 18000133
	v_accvgpr_read_b32 v12, a56                                // 000000005DA0: D3D8400C 18000138
	v_accvgpr_read_b32 v13, a57                                // 000000005DA8: D3D8400D 18000139
	v_accvgpr_read_b32 v14, a58                                // 000000005DB0: D3D8400E 1800013A
	v_accvgpr_read_b32 v15, a59                                // 000000005DB8: D3D8400F 1800013B
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000005DC0: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000005DC8: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000005DD0: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000005DD8: D2680013 00021F0E
	s_nop 1                                                    // 000000005DE0: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000005DE4: 7E20B312
	s_nop 1                                                    // 000000005DE8: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000005DEC: 7E22B313
	s_nop 1                                                    // 000000005DF0: BF800001
	ds_write_b128 v4, v[16:19] offset:4160                     // 000000005DF4: D9BE1040 00001004
	v_accvgpr_read_b32 v8, a36                                 // 000000005DFC: D3D84008 18000124
	v_accvgpr_read_b32 v9, a37                                 // 000000005E04: D3D84009 18000125
	v_accvgpr_read_b32 v10, a38                                // 000000005E0C: D3D8400A 18000126
	v_accvgpr_read_b32 v11, a39                                // 000000005E14: D3D8400B 18000127
	v_accvgpr_read_b32 v12, a44                                // 000000005E1C: D3D8400C 1800012C
	v_accvgpr_read_b32 v13, a45                                // 000000005E24: D3D8400D 1800012D
	v_accvgpr_read_b32 v14, a46                                // 000000005E2C: D3D8400E 1800012E
	v_accvgpr_read_b32 v15, a47                                // 000000005E34: D3D8400F 1800012F
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000005E3C: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000005E44: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000005E4C: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000005E54: D2680013 00021F0E
	s_nop 1                                                    // 000000005E5C: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000005E60: 7E20B312
	s_nop 1                                                    // 000000005E64: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000005E68: 7E22B313
	s_nop 1                                                    // 000000005E6C: BF800001
	ds_write_b128 v4, v[16:19] offset:6144                     // 000000005E70: D9BE1800 00001004
	v_accvgpr_read_b32 v8, a52                                 // 000000005E78: D3D84008 18000134
	v_accvgpr_read_b32 v9, a53                                 // 000000005E80: D3D84009 18000135
	v_accvgpr_read_b32 v10, a54                                // 000000005E88: D3D8400A 18000136
	v_accvgpr_read_b32 v11, a55                                // 000000005E90: D3D8400B 18000137
	v_accvgpr_read_b32 v12, a60                                // 000000005E98: D3D8400C 1800013C
	v_accvgpr_read_b32 v13, a61                                // 000000005EA0: D3D8400D 1800013D
	v_accvgpr_read_b32 v14, a62                                // 000000005EA8: D3D8400E 1800013E
	v_accvgpr_read_b32 v15, a63                                // 000000005EB0: D3D8400F 1800013F
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000005EB8: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000005EC0: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000005EC8: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000005ED0: D2680013 00021F0E
	s_nop 1                                                    // 000000005ED8: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000005EDC: 7E20B312
	s_nop 1                                                    // 000000005EE0: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000005EE4: 7E22B313
	s_nop 1                                                    // 000000005EE8: BF800001
	ds_write_b128 v4, v[16:19] offset:6208                     // 000000005EEC: D9BE1840 00001004
	v_accvgpr_read_b32 v8, a64                                 // 000000005EF4: D3D84008 18000140
	v_accvgpr_read_b32 v9, a65                                 // 000000005EFC: D3D84009 18000141
	v_accvgpr_read_b32 v10, a66                                // 000000005F04: D3D8400A 18000142
	v_accvgpr_read_b32 v11, a67                                // 000000005F0C: D3D8400B 18000143
	v_accvgpr_read_b32 v12, a72                                // 000000005F14: D3D8400C 18000148
	v_accvgpr_read_b32 v13, a73                                // 000000005F1C: D3D8400D 18000149
	v_accvgpr_read_b32 v14, a74                                // 000000005F24: D3D8400E 1800014A
	v_accvgpr_read_b32 v15, a75                                // 000000005F2C: D3D8400F 1800014B
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000005F34: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000005F3C: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000005F44: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000005F4C: D2680013 00021F0E
	s_nop 1                                                    // 000000005F54: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000005F58: 7E20B312
	s_nop 1                                                    // 000000005F5C: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000005F60: 7E22B313
	s_nop 1                                                    // 000000005F64: BF800001
	ds_write_b128 v4, v[16:19] offset:8192                     // 000000005F68: D9BE2000 00001004
	v_accvgpr_read_b32 v8, a80                                 // 000000005F70: D3D84008 18000150
	v_accvgpr_read_b32 v9, a81                                 // 000000005F78: D3D84009 18000151
	v_accvgpr_read_b32 v10, a82                                // 000000005F80: D3D8400A 18000152
	v_accvgpr_read_b32 v11, a83                                // 000000005F88: D3D8400B 18000153
	v_accvgpr_read_b32 v12, a88                                // 000000005F90: D3D8400C 18000158
	v_accvgpr_read_b32 v13, a89                                // 000000005F98: D3D8400D 18000159
	v_accvgpr_read_b32 v14, a90                                // 000000005FA0: D3D8400E 1800015A
	v_accvgpr_read_b32 v15, a91                                // 000000005FA8: D3D8400F 1800015B
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000005FB0: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000005FB8: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000005FC0: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000005FC8: D2680013 00021F0E
	s_nop 1                                                    // 000000005FD0: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000005FD4: 7E20B312
	s_nop 1                                                    // 000000005FD8: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000005FDC: 7E22B313
	s_nop 1                                                    // 000000005FE0: BF800001
	ds_write_b128 v4, v[16:19] offset:8256                     // 000000005FE4: D9BE2040 00001004
	v_accvgpr_read_b32 v8, a68                                 // 000000005FEC: D3D84008 18000144
	v_accvgpr_read_b32 v9, a69                                 // 000000005FF4: D3D84009 18000145
	v_accvgpr_read_b32 v10, a70                                // 000000005FFC: D3D8400A 18000146
	v_accvgpr_read_b32 v11, a71                                // 000000006004: D3D8400B 18000147
	v_accvgpr_read_b32 v12, a76                                // 00000000600C: D3D8400C 1800014C
	v_accvgpr_read_b32 v13, a77                                // 000000006014: D3D8400D 1800014D
	v_accvgpr_read_b32 v14, a78                                // 00000000601C: D3D8400E 1800014E
	v_accvgpr_read_b32 v15, a79                                // 000000006024: D3D8400F 1800014F
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 00000000602C: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000006034: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 00000000603C: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000006044: D2680013 00021F0E
	s_nop 1                                                    // 00000000604C: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000006050: 7E20B312
	s_nop 1                                                    // 000000006054: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000006058: 7E22B313
	s_nop 1                                                    // 00000000605C: BF800001
	ds_write_b128 v4, v[16:19] offset:10240                    // 000000006060: D9BE2800 00001004
	v_accvgpr_read_b32 v8, a84                                 // 000000006068: D3D84008 18000154
	v_accvgpr_read_b32 v9, a85                                 // 000000006070: D3D84009 18000155
	v_accvgpr_read_b32 v10, a86                                // 000000006078: D3D8400A 18000156
	v_accvgpr_read_b32 v11, a87                                // 000000006080: D3D8400B 18000157
	v_accvgpr_read_b32 v12, a92                                // 000000006088: D3D8400C 1800015C
	v_accvgpr_read_b32 v13, a93                                // 000000006090: D3D8400D 1800015D
	v_accvgpr_read_b32 v14, a94                                // 000000006098: D3D8400E 1800015E
	v_accvgpr_read_b32 v15, a95                                // 0000000060A0: D3D8400F 1800015F
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 0000000060A8: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 0000000060B0: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 0000000060B8: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 0000000060C0: D2680013 00021F0E
	s_nop 1                                                    // 0000000060C8: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 0000000060CC: 7E20B312
	s_nop 1                                                    // 0000000060D0: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 0000000060D4: 7E22B313
	s_nop 1                                                    // 0000000060D8: BF800001
	ds_write_b128 v4, v[16:19] offset:10304                    // 0000000060DC: D9BE2840 00001004
	v_accvgpr_read_b32 v8, a96                                 // 0000000060E4: D3D84008 18000160
	v_accvgpr_read_b32 v9, a97                                 // 0000000060EC: D3D84009 18000161
	v_accvgpr_read_b32 v10, a98                                // 0000000060F4: D3D8400A 18000162
	v_accvgpr_read_b32 v11, a99                                // 0000000060FC: D3D8400B 18000163
	v_accvgpr_read_b32 v12, a104                               // 000000006104: D3D8400C 18000168
	v_accvgpr_read_b32 v13, a105                               // 00000000610C: D3D8400D 18000169
	v_accvgpr_read_b32 v14, a106                               // 000000006114: D3D8400E 1800016A
	v_accvgpr_read_b32 v15, a107                               // 00000000611C: D3D8400F 1800016B
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000006124: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 00000000612C: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000006134: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 00000000613C: D2680013 00021F0E
	s_nop 1                                                    // 000000006144: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000006148: 7E20B312
	s_nop 1                                                    // 00000000614C: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000006150: 7E22B313
	s_nop 1                                                    // 000000006154: BF800001
	ds_write_b128 v4, v[16:19] offset:12288                    // 000000006158: D9BE3000 00001004
	v_accvgpr_read_b32 v8, a112                                // 000000006160: D3D84008 18000170
	v_accvgpr_read_b32 v9, a113                                // 000000006168: D3D84009 18000171
	v_accvgpr_read_b32 v10, a114                               // 000000006170: D3D8400A 18000172
	v_accvgpr_read_b32 v11, a115                               // 000000006178: D3D8400B 18000173
	v_accvgpr_read_b32 v12, a120                               // 000000006180: D3D8400C 18000178
	v_accvgpr_read_b32 v13, a121                               // 000000006188: D3D8400D 18000179
	v_accvgpr_read_b32 v14, a122                               // 000000006190: D3D8400E 1800017A
	v_accvgpr_read_b32 v15, a123                               // 000000006198: D3D8400F 1800017B
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 0000000061A0: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 0000000061A8: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 0000000061B0: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 0000000061B8: D2680013 00021F0E
	s_nop 1                                                    // 0000000061C0: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 0000000061C4: 7E20B312
	s_nop 1                                                    // 0000000061C8: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 0000000061CC: 7E22B313
	s_nop 1                                                    // 0000000061D0: BF800001
	ds_write_b128 v4, v[16:19] offset:12352                    // 0000000061D4: D9BE3040 00001004
	v_accvgpr_read_b32 v8, a100                                // 0000000061DC: D3D84008 18000164
	v_accvgpr_read_b32 v9, a101                                // 0000000061E4: D3D84009 18000165
	v_accvgpr_read_b32 v10, a102                               // 0000000061EC: D3D8400A 18000166
	v_accvgpr_read_b32 v11, a103                               // 0000000061F4: D3D8400B 18000167
	v_accvgpr_read_b32 v12, a108                               // 0000000061FC: D3D8400C 1800016C
	v_accvgpr_read_b32 v13, a109                               // 000000006204: D3D8400D 1800016D
	v_accvgpr_read_b32 v14, a110                               // 00000000620C: D3D8400E 1800016E
	v_accvgpr_read_b32 v15, a111                               // 000000006214: D3D8400F 1800016F
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 00000000621C: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000006224: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 00000000622C: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000006234: D2680013 00021F0E
	s_nop 1                                                    // 00000000623C: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000006240: 7E20B312
	s_nop 1                                                    // 000000006244: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000006248: 7E22B313
	s_nop 1                                                    // 00000000624C: BF800001
	ds_write_b128 v4, v[16:19] offset:14336                    // 000000006250: D9BE3800 00001004
	v_accvgpr_read_b32 v8, a116                                // 000000006258: D3D84008 18000174
	v_accvgpr_read_b32 v9, a117                                // 000000006260: D3D84009 18000175
	v_accvgpr_read_b32 v10, a118                               // 000000006268: D3D8400A 18000176
	v_accvgpr_read_b32 v11, a119                               // 000000006270: D3D8400B 18000177
	v_accvgpr_read_b32 v12, a124                               // 000000006278: D3D8400C 1800017C
	v_accvgpr_read_b32 v13, a125                               // 000000006280: D3D8400D 1800017D
	v_accvgpr_read_b32 v14, a126                               // 000000006288: D3D8400E 1800017E
	v_accvgpr_read_b32 v15, a127                               // 000000006290: D3D8400F 1800017F
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000006298: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 0000000062A0: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 0000000062A8: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 0000000062B0: D2680013 00021F0E
	s_nop 1                                                    // 0000000062B8: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 0000000062BC: 7E20B312
	s_nop 1                                                    // 0000000062C0: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 0000000062C4: 7E22B313
	s_nop 1                                                    // 0000000062C8: BF800001
	ds_write_b128 v4, v[16:19] offset:14400                    // 0000000062CC: D9BE3840 00001004
	v_accvgpr_read_b32 v8, a128                                // 0000000062D4: D3D84008 18000180
	v_accvgpr_read_b32 v9, a129                                // 0000000062DC: D3D84009 18000181
	v_accvgpr_read_b32 v10, a130                               // 0000000062E4: D3D8400A 18000182
	v_accvgpr_read_b32 v11, a131                               // 0000000062EC: D3D8400B 18000183
	v_accvgpr_read_b32 v12, a136                               // 0000000062F4: D3D8400C 18000188
	v_accvgpr_read_b32 v13, a137                               // 0000000062FC: D3D8400D 18000189
	v_accvgpr_read_b32 v14, a138                               // 000000006304: D3D8400E 1800018A
	v_accvgpr_read_b32 v15, a139                               // 00000000630C: D3D8400F 1800018B
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000006314: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 00000000631C: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000006324: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 00000000632C: D2680013 00021F0E
	s_nop 1                                                    // 000000006334: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000006338: 7E20B312
	s_nop 1                                                    // 00000000633C: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000006340: 7E22B313
	s_nop 1                                                    // 000000006344: BF800001
	ds_write_b128 v4, v[16:19] offset:16384                    // 000000006348: D9BE4000 00001004
	v_accvgpr_read_b32 v8, a144                                // 000000006350: D3D84008 18000190
	v_accvgpr_read_b32 v9, a145                                // 000000006358: D3D84009 18000191
	v_accvgpr_read_b32 v10, a146                               // 000000006360: D3D8400A 18000192
	v_accvgpr_read_b32 v11, a147                               // 000000006368: D3D8400B 18000193
	v_accvgpr_read_b32 v12, a152                               // 000000006370: D3D8400C 18000198
	v_accvgpr_read_b32 v13, a153                               // 000000006378: D3D8400D 18000199
	v_accvgpr_read_b32 v14, a154                               // 000000006380: D3D8400E 1800019A
	v_accvgpr_read_b32 v15, a155                               // 000000006388: D3D8400F 1800019B
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000006390: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000006398: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 0000000063A0: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 0000000063A8: D2680013 00021F0E
	s_nop 1                                                    // 0000000063B0: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 0000000063B4: 7E20B312
	s_nop 1                                                    // 0000000063B8: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 0000000063BC: 7E22B313
	s_nop 1                                                    // 0000000063C0: BF800001
	ds_write_b128 v4, v[16:19] offset:16448                    // 0000000063C4: D9BE4040 00001004
	v_accvgpr_read_b32 v8, a132                                // 0000000063CC: D3D84008 18000184
	v_accvgpr_read_b32 v9, a133                                // 0000000063D4: D3D84009 18000185
	v_accvgpr_read_b32 v10, a134                               // 0000000063DC: D3D8400A 18000186
	v_accvgpr_read_b32 v11, a135                               // 0000000063E4: D3D8400B 18000187
	v_accvgpr_read_b32 v12, a140                               // 0000000063EC: D3D8400C 1800018C
	v_accvgpr_read_b32 v13, a141                               // 0000000063F4: D3D8400D 1800018D
	v_accvgpr_read_b32 v14, a142                               // 0000000063FC: D3D8400E 1800018E
	v_accvgpr_read_b32 v15, a143                               // 000000006404: D3D8400F 1800018F
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 00000000640C: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000006414: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 00000000641C: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000006424: D2680013 00021F0E
	s_nop 1                                                    // 00000000642C: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000006430: 7E20B312
	s_nop 1                                                    // 000000006434: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000006438: 7E22B313
	s_nop 1                                                    // 00000000643C: BF800001
	ds_write_b128 v4, v[16:19] offset:18432                    // 000000006440: D9BE4800 00001004
	v_accvgpr_read_b32 v8, a148                                // 000000006448: D3D84008 18000194
	v_accvgpr_read_b32 v9, a149                                // 000000006450: D3D84009 18000195
	v_accvgpr_read_b32 v10, a150                               // 000000006458: D3D8400A 18000196
	v_accvgpr_read_b32 v11, a151                               // 000000006460: D3D8400B 18000197
	v_accvgpr_read_b32 v12, a156                               // 000000006468: D3D8400C 1800019C
	v_accvgpr_read_b32 v13, a157                               // 000000006470: D3D8400D 1800019D
	v_accvgpr_read_b32 v14, a158                               // 000000006478: D3D8400E 1800019E
	v_accvgpr_read_b32 v15, a159                               // 000000006480: D3D8400F 1800019F
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000006488: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000006490: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000006498: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 0000000064A0: D2680013 00021F0E
	s_nop 1                                                    // 0000000064A8: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 0000000064AC: 7E20B312
	s_nop 1                                                    // 0000000064B0: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 0000000064B4: 7E22B313
	s_nop 1                                                    // 0000000064B8: BF800001
	ds_write_b128 v4, v[16:19] offset:18496                    // 0000000064BC: D9BE4840 00001004
	v_accvgpr_read_b32 v8, a160                                // 0000000064C4: D3D84008 180001A0
	v_accvgpr_read_b32 v9, a161                                // 0000000064CC: D3D84009 180001A1
	v_accvgpr_read_b32 v10, a162                               // 0000000064D4: D3D8400A 180001A2
	v_accvgpr_read_b32 v11, a163                               // 0000000064DC: D3D8400B 180001A3
	v_accvgpr_read_b32 v12, a168                               // 0000000064E4: D3D8400C 180001A8
	v_accvgpr_read_b32 v13, a169                               // 0000000064EC: D3D8400D 180001A9
	v_accvgpr_read_b32 v14, a170                               // 0000000064F4: D3D8400E 180001AA
	v_accvgpr_read_b32 v15, a171                               // 0000000064FC: D3D8400F 180001AB
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000006504: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 00000000650C: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000006514: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 00000000651C: D2680013 00021F0E
	s_nop 1                                                    // 000000006524: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000006528: 7E20B312
	s_nop 1                                                    // 00000000652C: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000006530: 7E22B313
	s_nop 1                                                    // 000000006534: BF800001
	ds_write_b128 v4, v[16:19] offset:20480                    // 000000006538: D9BE5000 00001004
	v_accvgpr_read_b32 v8, a176                                // 000000006540: D3D84008 180001B0
	v_accvgpr_read_b32 v9, a177                                // 000000006548: D3D84009 180001B1
	v_accvgpr_read_b32 v10, a178                               // 000000006550: D3D8400A 180001B2
	v_accvgpr_read_b32 v11, a179                               // 000000006558: D3D8400B 180001B3
	v_accvgpr_read_b32 v12, a184                               // 000000006560: D3D8400C 180001B8
	v_accvgpr_read_b32 v13, a185                               // 000000006568: D3D8400D 180001B9
	v_accvgpr_read_b32 v14, a186                               // 000000006570: D3D8400E 180001BA
	v_accvgpr_read_b32 v15, a187                               // 000000006578: D3D8400F 180001BB
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000006580: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000006588: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000006590: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000006598: D2680013 00021F0E
	s_nop 1                                                    // 0000000065A0: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 0000000065A4: 7E20B312
	s_nop 1                                                    // 0000000065A8: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 0000000065AC: 7E22B313
	s_nop 1                                                    // 0000000065B0: BF800001
	ds_write_b128 v4, v[16:19] offset:20544                    // 0000000065B4: D9BE5040 00001004
	v_accvgpr_read_b32 v8, a164                                // 0000000065BC: D3D84008 180001A4
	v_accvgpr_read_b32 v9, a165                                // 0000000065C4: D3D84009 180001A5
	v_accvgpr_read_b32 v10, a166                               // 0000000065CC: D3D8400A 180001A6
	v_accvgpr_read_b32 v11, a167                               // 0000000065D4: D3D8400B 180001A7
	v_accvgpr_read_b32 v12, a172                               // 0000000065DC: D3D8400C 180001AC
	v_accvgpr_read_b32 v13, a173                               // 0000000065E4: D3D8400D 180001AD
	v_accvgpr_read_b32 v14, a174                               // 0000000065EC: D3D8400E 180001AE
	v_accvgpr_read_b32 v15, a175                               // 0000000065F4: D3D8400F 180001AF
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 0000000065FC: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000006604: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 00000000660C: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000006614: D2680013 00021F0E
	s_nop 1                                                    // 00000000661C: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 000000006620: 7E20B312
	s_nop 1                                                    // 000000006624: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 000000006628: 7E22B313
	s_nop 1                                                    // 00000000662C: BF800001
	ds_write_b128 v4, v[16:19] offset:22528                    // 000000006630: D9BE5800 00001004
	v_accvgpr_read_b32 v8, a180                                // 000000006638: D3D84008 180001B4
	v_accvgpr_read_b32 v9, a181                                // 000000006640: D3D84009 180001B5
	v_accvgpr_read_b32 v10, a182                               // 000000006648: D3D8400A 180001B6
	v_accvgpr_read_b32 v11, a183                               // 000000006650: D3D8400B 180001B7
	v_accvgpr_read_b32 v12, a188                               // 000000006658: D3D8400C 180001BC
	v_accvgpr_read_b32 v13, a189                               // 000000006660: D3D8400D 180001BD
	v_accvgpr_read_b32 v14, a190                               // 000000006668: D3D8400E 180001BE
	v_accvgpr_read_b32 v15, a191                               // 000000006670: D3D8400F 180001BF
	v_cvt_pk_bf16_f32 v16, v8, v9                              // 000000006678: D2680010 00021308
	v_cvt_pk_bf16_f32 v17, v10, v11                            // 000000006680: D2680011 0002170A
	v_cvt_pk_bf16_f32 v18, v12, v13                            // 000000006688: D2680012 00021B0C
	v_cvt_pk_bf16_f32 v19, v14, v15                            // 000000006690: D2680013 00021F0E
	s_nop 1                                                    // 000000006698: BF800001
	v_permlane16_swap_b32_e32 v16, v18                         // 00000000669C: 7E20B312
	s_nop 1                                                    // 0000000066A0: BF800001
	v_permlane16_swap_b32_e32 v17, v19                         // 0000000066A4: 7E22B313
	s_nop 1                                                    // 0000000066A8: BF800001
	ds_write_b128 v4, v[16:19] offset:22592                    // 0000000066AC: D9BE5840 00001004
	s_waitcnt lgkmcnt(0)                                       // 0000000066B4: BF8CC07F
	v_mul_i32_i24_e64 v4, v0, 16                               // 0000000066B8: D1060004 00012100
	v_add_i32 v4, v4, s62                                      // 0000000066C0: D29C0004 00007D04
	ds_read_b128 v[16:19], v4                                  // 0000000066C8: D9FE0000 10000004
	s_waitcnt lgkmcnt(0)                                       // 0000000066D0: BF8CC07F
	buffer_store_dwordx4 v[16:19], v195, s[4:7], 0 offen       // 0000000066D4: E07C1000 800110C3
	ds_read_b128 v[16:19], v4 offset:1024                      // 0000000066DC: D9FE0400 10000004
	s_waitcnt lgkmcnt(0)                                       // 0000000066E4: BF8CC07F
	buffer_store_dwordx4 v[16:19], v196, s[4:7], 0 offen       // 0000000066E8: E07C1000 800110C4
	ds_read_b128 v[16:19], v4 offset:2048                      // 0000000066F0: D9FE0800 10000004
	s_waitcnt lgkmcnt(0)                                       // 0000000066F8: BF8CC07F
	buffer_store_dwordx4 v[16:19], v197, s[4:7], 0 offen       // 0000000066FC: E07C1000 800110C5
	ds_read_b128 v[16:19], v4 offset:3072                      // 000000006704: D9FE0C00 10000004
	s_waitcnt lgkmcnt(0)                                       // 00000000670C: BF8CC07F
	buffer_store_dwordx4 v[16:19], v198, s[4:7], 0 offen       // 000000006710: E07C1000 800110C6
	ds_read_b128 v[16:19], v4 offset:4096                      // 000000006718: D9FE1000 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006720: BF8CC07F
	buffer_store_dwordx4 v[16:19], v199, s[4:7], 0 offen       // 000000006724: E07C1000 800110C7
	ds_read_b128 v[16:19], v4 offset:5120                      // 00000000672C: D9FE1400 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006734: BF8CC07F
	buffer_store_dwordx4 v[16:19], v200, s[4:7], 0 offen       // 000000006738: E07C1000 800110C8
	ds_read_b128 v[16:19], v4 offset:6144                      // 000000006740: D9FE1800 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006748: BF8CC07F
	buffer_store_dwordx4 v[16:19], v201, s[4:7], 0 offen       // 00000000674C: E07C1000 800110C9
	ds_read_b128 v[16:19], v4 offset:7168                      // 000000006754: D9FE1C00 10000004
	s_waitcnt lgkmcnt(0)                                       // 00000000675C: BF8CC07F
	buffer_store_dwordx4 v[16:19], v202, s[4:7], 0 offen       // 000000006760: E07C1000 800110CA
	ds_read_b128 v[16:19], v4 offset:8192                      // 000000006768: D9FE2000 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006770: BF8CC07F
	buffer_store_dwordx4 v[16:19], v203, s[4:7], 0 offen       // 000000006774: E07C1000 800110CB
	ds_read_b128 v[16:19], v4 offset:9216                      // 00000000677C: D9FE2400 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006784: BF8CC07F
	buffer_store_dwordx4 v[16:19], v204, s[4:7], 0 offen       // 000000006788: E07C1000 800110CC
	ds_read_b128 v[16:19], v4 offset:10240                     // 000000006790: D9FE2800 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006798: BF8CC07F
	buffer_store_dwordx4 v[16:19], v205, s[4:7], 0 offen       // 00000000679C: E07C1000 800110CD
	ds_read_b128 v[16:19], v4 offset:11264                     // 0000000067A4: D9FE2C00 10000004
	s_waitcnt lgkmcnt(0)                                       // 0000000067AC: BF8CC07F
	buffer_store_dwordx4 v[16:19], v206, s[4:7], 0 offen       // 0000000067B0: E07C1000 800110CE
	ds_read_b128 v[16:19], v4 offset:12288                     // 0000000067B8: D9FE3000 10000004
	s_waitcnt lgkmcnt(0)                                       // 0000000067C0: BF8CC07F
	buffer_store_dwordx4 v[16:19], v207, s[4:7], 0 offen       // 0000000067C4: E07C1000 800110CF
	ds_read_b128 v[16:19], v4 offset:13312                     // 0000000067CC: D9FE3400 10000004
	s_waitcnt lgkmcnt(0)                                       // 0000000067D4: BF8CC07F
	buffer_store_dwordx4 v[16:19], v208, s[4:7], 0 offen       // 0000000067D8: E07C1000 800110D0
	ds_read_b128 v[16:19], v4 offset:14336                     // 0000000067E0: D9FE3800 10000004
	s_waitcnt lgkmcnt(0)                                       // 0000000067E8: BF8CC07F
	buffer_store_dwordx4 v[16:19], v209, s[4:7], 0 offen       // 0000000067EC: E07C1000 800110D1
	ds_read_b128 v[16:19], v4 offset:15360                     // 0000000067F4: D9FE3C00 10000004
	s_waitcnt lgkmcnt(0)                                       // 0000000067FC: BF8CC07F
	buffer_store_dwordx4 v[16:19], v210, s[4:7], 0 offen       // 000000006800: E07C1000 800110D2
	ds_read_b128 v[16:19], v4 offset:16384                     // 000000006808: D9FE4000 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006810: BF8CC07F
	buffer_store_dwordx4 v[16:19], v211, s[4:7], 0 offen       // 000000006814: E07C1000 800110D3
	ds_read_b128 v[16:19], v4 offset:17408                     // 00000000681C: D9FE4400 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006824: BF8CC07F
	buffer_store_dwordx4 v[16:19], v212, s[4:7], 0 offen       // 000000006828: E07C1000 800110D4
	ds_read_b128 v[16:19], v4 offset:18432                     // 000000006830: D9FE4800 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006838: BF8CC07F
	buffer_store_dwordx4 v[16:19], v213, s[4:7], 0 offen       // 00000000683C: E07C1000 800110D5
	ds_read_b128 v[16:19], v4 offset:19456                     // 000000006844: D9FE4C00 10000004
	s_waitcnt lgkmcnt(0)                                       // 00000000684C: BF8CC07F
	buffer_store_dwordx4 v[16:19], v214, s[4:7], 0 offen       // 000000006850: E07C1000 800110D6
	ds_read_b128 v[16:19], v4 offset:20480                     // 000000006858: D9FE5000 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006860: BF8CC07F
	buffer_store_dwordx4 v[16:19], v215, s[4:7], 0 offen       // 000000006864: E07C1000 800110D7
	ds_read_b128 v[16:19], v4 offset:21504                     // 00000000686C: D9FE5400 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006874: BF8CC07F
	buffer_store_dwordx4 v[16:19], v216, s[4:7], 0 offen       // 000000006878: E07C1000 800110D8
	ds_read_b128 v[16:19], v4 offset:22528                     // 000000006880: D9FE5800 10000004
	s_waitcnt lgkmcnt(0)                                       // 000000006888: BF8CC07F
	buffer_store_dwordx4 v[16:19], v217, s[4:7], 0 offen       // 00000000688C: E07C1000 800110D9
	ds_read_b128 v[16:19], v4 offset:23552                     // 000000006894: D9FE5C00 10000004
	s_waitcnt lgkmcnt(0)                                       // 00000000689C: BF8CC07F
	buffer_store_dwordx4 v[16:19], v218, s[4:7], 0 offen       // 0000000068A0: E07C1000 800110DA
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 0000000068A8: BF8C0000
	s_endpgm                                                   // 0000000068AC: BF810000
