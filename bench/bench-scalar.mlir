// Scalar benchmark: 8-element i32 add via RV32I
// Returns a value in %sum so it's not DCE'd
func.func @scalar_add_bench() -> i32 {
  %a0 = coralnpu.li 1 : i32
  %a1 = coralnpu.li 2 : i32
  %a2 = coralnpu.li 3 : i32
  %a3 = coralnpu.li 4 : i32
  %b0 = coralnpu.li 5 : i32
  %b1 = coralnpu.li 6 : i32
  %b2 = coralnpu.li 7 : i32
  %b3 = coralnpu.li 8 : i32
  %c0 = coralnpu.add %a0, %b0 : i32
  %c1 = coralnpu.add %a1, %b1 : i32
  %c2 = coralnpu.add %a2, %b2 : i32
  %c3 = coralnpu.add %a3, %b3 : i32
  %d0 = coralnpu.add %c0, %c1 : i32
  %d1 = coralnpu.add %c2, %c3 : i32
  %e = coralnpu.add %d0, %d1 : i32
  coralnpu.return %e : i32
}
