// Hello CoralNPU: add two constants and return
func.func @hello() -> i32 {
  %a = coralnpu.li 42 : i32
  %b = coralnpu.li 7 : i32
  %c = coralnpu.add %a, %b : i32
  coralnpu.return %c : i32
}
