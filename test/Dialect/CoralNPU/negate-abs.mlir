// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// tosa.negate → per-tile vrsub.vx vD, vS, x0  (vD[i] = 0 - vS[i])
// tosa.abs    → per-tile relu(x) + relu(-x)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_abs
// CHECK: coralnpu.vsetvl
// CHECK: %[[T:.*]] = coralnpu.vle32
// CHECK: %[[ZERO:.*]] = coralnpu.li 0
// CHECK: coralnpu.vsub_vx %[[ZERO]], %[[T]]
// CHECK: coralnpu.vse32
// ASM-LABEL: # CoralNPU Assembly
// ASM: vle32.v
// ASM: add
func.func @test_abs(%a: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tosa.abs %a : (tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}
