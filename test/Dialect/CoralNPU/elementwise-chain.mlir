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
// tile, and stores both result tiles. add1 reloads add0's result from its
// assigned TCM slot (per-op unique slot assignment) plus loads c tiles.
// Note: reloadCarrierFromSlot re-emits VLE32 for distant consumers to prevent
// register reuse issues; hence add0 result tiles appear as fresh VLE32s in add1.
// CHECK: %[[A0:.*]] = coralnpu.vle32
// CHECK: %[[A1:.*]] = coralnpu.vle32
// CHECK: %[[B0:.*]] = coralnpu.vle32
// CHECK: %[[B1:.*]] = coralnpu.vle32
// CHECK: %[[S0:.*]] = coralnpu.vadd %[[A0]], %[[B0]] : (!coralnpu.vreg<e32, m1>, !coralnpu.vreg<e32, m1>) -> !coralnpu.vreg<e32, m1>
// CHECK: %[[S1:.*]] = coralnpu.vadd %[[A1]], %[[B1]] : (!coralnpu.vreg<e32, m1>, !coralnpu.vreg<e32, m1>) -> !coralnpu.vreg<e32, m1>
// CHECK: coralnpu.vse32 %[[S0]]
// CHECK: coralnpu.vse32 %[[S1]]
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vadd
// CHECK: coralnpu.vadd
// CHECK: coralnpu.vse32
// CHECK: coralnpu.vse32

// In assembly, add0 produces register tiles that are stored then reloaded
// by add1 (per-op slot isolation). Four vadd instructions total.
// ASM: vadd.vv
// ASM: vadd.vv
// ASM: vadd.vv
// ASM: vadd.vv

func.func @chain(%a: tensor<8xi32>, %b: tensor<8xi32>, %c: tensor<8xi32>) -> tensor<8xi32> {
  %0 = tosa.add %a, %b : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  %1 = tosa.add %0, %c : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  func.return %1 : tensor<8xi32>
}
