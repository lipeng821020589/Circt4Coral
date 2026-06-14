// Vector benchmark: 4-element i32 add via RVV
// (one vle8 loads 4 i32s, one vadd adds them, one vse8 stores)
func.func @vector_add_bench() -> i32 {
  %addr1 = coralnpu.li 0x40000000 : i32
  %addr2 = coralnpu.li 0x40000020 : i32
  %addr3 = coralnpu.li 0x40000040 : i32
  %n = coralnpu.li 16 : i32
  %v1 = coralnpu.vle8 %addr1, %n : i32
  %v2 = coralnpu.vle8 %addr2, %n : i32
  %v3 = coralnpu.vadd %v1, %v2 : i32
  coralnpu.vse8 %addr3, %v3, %n : i32
  %zero = coralnpu.li 0 : i32
  coralnpu.return %zero : i32
}
