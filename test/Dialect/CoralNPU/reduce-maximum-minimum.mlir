// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// tosa.reduce_sum, tosa.reduce_max, tosa.maximum, tosa.minimum lowering.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_minimum
// CHECK: coralnpu.vsetvl e32, m1
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vmin_vv
// CHECK: coralnpu.vse32
// ASM-LABEL: # CoralNPU Assembly
// ASM: vle32.v
func.func @test_minimum(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tosa.minimum %a, %b : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}
