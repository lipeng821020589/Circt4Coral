// Vector benchmark: 16-element i32 add
// Chain ensures DCE doesn't kill: each result feeds next op
func.func @vector_add16_bench() -> i32 {
  %addr1 = coralnpu.li 0x40000000 : i32
  %addr2 = coralnpu.li 0x40000020 : i32
  %addr3 = coralnpu.li 0x40000040 : i32
  %addr4 = coralnpu.li 0x40000060 : i32
  %n = coralnpu.li 16 : i32
  // Vop 1: a+b -> c
  %v1a = coralnpu.vle8 %addr1, %n : i32
  %v1b = coralnpu.vle8 %addr2, %n : i32
  %v1c = coralnpu.vadd %v1a, %v1b : i32
  // Vop 2: c+d -> a (use v1c result)
  %v2a = coralnpu.vle8 %addr3, %n : i32
  %v2b = coralnpu.vadd %v1c, %v2a : i32
  // Vop 3: b+v2b -> b
  %v3a = coralnpu.vle8 %addr2, %n : i32
  %v3b = coralnpu.vadd %v3a, %v2b : i32
  // Vop 4: v3b+v3b -> final
  %v4a = coralnpu.vadd %v3b, %v3b : i32
  coralnpu.return %v4a : i32
}
