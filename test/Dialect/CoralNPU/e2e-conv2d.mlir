// RUN: circt-opt --tosa-to-coralnpu --coralnpu-stripmine --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s

// CHECK: # Coral NPU Assembly
// CHECK: li     x1, 0
// CHECK: vadd.vv  # stripmine=4
// CHECK: ret
// CHECK: # End of assembly

func.func @e2e_add(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  func.return %0 : tensor<4x4xi32>
}
