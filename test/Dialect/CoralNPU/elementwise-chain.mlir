// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s --check-prefix=ASM

//===----------------------------------------------------------------------===//
// Chained element-wise lowering with real, tiled data flow.
//
// Computes out = (a + b) + c on tensor<8xi32>. At e32 a vector register holds
// 4 elements, so each tensor is split into k = ceil(8/4) = 2 register-sized
// tiles. The first add's per-tile results must flow directly into the second
// add through vector registers (no reload from memory), proving the lowering
// carries real SSA data flow at *tile* granularity — not a single fake
// register pretending to hold all 8 elements, and not zero placeholders.
//===----------------------------------------------------------------------===//

// add0 loads both operands tile-by-tile (2 tiles each), issues one vadd per
// tile, and stores both result tiles. Tiling replaces stripmine, so each vadd
// processes exactly one register and carries no `stripmine` clause (factor 1,
// elided). add1 reuses add0's two result vregs as its lhs tiles and only loads
// the two tiles of c.
// CHECK: %[[A0:.*]] = coralnpu.vle32
// CHECK: %[[A1:.*]] = coralnpu.vle32
// CHECK: %[[B0:.*]] = coralnpu.vle32
// CHECK: %[[B1:.*]] = coralnpu.vle32
// CHECK: %[[S0:.*]] = coralnpu.vadd %[[A0]], %[[B0]] : i32
// CHECK: %[[S1:.*]] = coralnpu.vadd %[[A1]], %[[B1]] : i32
// CHECK: coralnpu.vse32 %[[S0]]
// CHECK: coralnpu.vse32 %[[S1]]
// CHECK: %[[C0:.*]] = coralnpu.vle32
// CHECK: %[[C1:.*]] = coralnpu.vle32
// CHECK: %[[T0:.*]] = coralnpu.vadd %[[S0]], %[[C0]] : i32
// CHECK: %[[T1:.*]] = coralnpu.vadd %[[S1]], %[[C1]] : i32
// CHECK: coralnpu.vse32 %[[T0]]
// CHECK: coralnpu.vse32 %[[T1]]

// In assembly, add0 produces two destination registers; add1 consumes those
// same two registers as its lhs operands (chained in registers per tile, only
// c's tiles are reloaded).
// ASM: vadd.vv v4, v0, v2
// ASM: vadd.vv v0, v1, v3
// ASM: vadd.vv v3, v4, v1
// ASM: vadd.vv v1, v0, v2

func.func @chain(%a: tensor<8xi32>, %b: tensor<8xi32>, %c: tensor<8xi32>) -> tensor<8xi32> {
  %0 = tosa.add %a, %b : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  %1 = tosa.add %0, %c : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  func.return %1 : tensor<8xi32>
}
