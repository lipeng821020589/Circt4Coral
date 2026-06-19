// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s --check-prefix=ASM

//===----------------------------------------------------------------------===//
// Chained element-wise lowering with real data flow.
//
// Computes out = (a + b) + c. The first add's result must flow directly into
// the second add through a vector register (no reload from memory), proving
// the lowering carries real SSA data flow rather than zero placeholders.
//===----------------------------------------------------------------------===//

// The first add loads both operands; the second add reuses the first add's
// result (%6) as an operand and only loads the third operand.
// The 8xi32 tensors carry 8 elements; at e32 a vector register holds 4, so the
// real stripmine factor is ceil(8/4) = 2 (derived from the tile size, not a
// hardcoded constant).
// CHECK: %[[A:.*]] = coralnpu.vle32
// CHECK: %[[B:.*]] = coralnpu.vle32
// CHECK: %[[ADD0:.*]] = coralnpu.vadd stripmine = 2 %[[A]], %[[B]]
// CHECK: coralnpu.vse32 %[[ADD0]]
// CHECK: %[[C:.*]] = coralnpu.vle32
// CHECK: %[[ADD1:.*]] = coralnpu.vadd stripmine = 2 %[[ADD0]], %[[C]]
// CHECK: coralnpu.vse32 %[[ADD1]]

// In assembly, the second vadd consumes the first vadd's destination register
// directly (chained in registers, only the new operand is reloaded).
// ASM: vle32.v
// ASM: vle32.v
// ASM: vadd.vv v2, v0, v1
// ASM: vse32.v v2
// ASM: vle32.v v0
// ASM: vadd.vv v1, v2, v0
// ASM: vse32.v v1

func.func @chain(%a: tensor<8xi32>, %b: tensor<8xi32>, %c: tensor<8xi32>) -> tensor<8xi32> {
  %0 = tosa.add %a, %b : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  %1 = tosa.add %0, %c : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  func.return %1 : tensor<8xi32>
}
