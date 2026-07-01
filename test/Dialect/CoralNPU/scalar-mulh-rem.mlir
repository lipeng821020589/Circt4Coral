// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// CoralNPU scalar mulh / mulhsu / rem — round-trip + assembly emission.
//
// mulh   : high 32 bits of signed×signed 64-bit product (RV32M mulh)
// mulhsu : high 32 bits of signed×unsigned 64-bit product (RV32M mulhsu)
// rem    : signed remainder (RV32M rem)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_mulh_rem
// CHECK: coralnpu.mulh
// CHECK: coralnpu.mulhsu
// CHECK: coralnpu.rem
// ASM: mulh
// ASM: mulhsu
// ASM: rem
func.func @test_mulh_rem(%a: i32, %b: i32) -> i32 {
  %h  = coralnpu.mulh   %a, %b : i32
  %hu = coralnpu.mulhsu %a, %b : i32
  %r  = coralnpu.rem    %h, %b : i32
  %s  = coralnpu.add    %r, %hu : i32
  coralnpu.return %s : i32
}
