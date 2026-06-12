// RUN: circt-opt --tosa-to-coralnpu %s --verify-diagnostics 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// TOSA → CoralNPU lowering tests
// Verifies CoralNPU ops are produced from TOSA ops.
// Note: tensor→scalar type mismatches are expected at this lowering stage.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: test_tosa_add
// CHECK: coralnpu.li
// CHECK: coralnpu.vadd
func.func @test_tosa_add(%arg0: tensor<8x8xi32>, %arg1: tensor<8x8xi32>) -> tensor<8x8xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi32>
  func.return %0 : tensor<8x8xi32>
}

// CHECK-LABEL: test_tosa_transpose
// CHECK: coralnpu.li
// CHECK: coralnpu.dma_load
// CHECK: coralnpu.dma_store
func.func @test_tosa_transpose(%arg0: tensor<2x4xi32>) -> tensor<4x2xi32> {
  %0 = tosa.transpose %arg0 {perms = array<i32: 1, 0>} : (tensor<2x4xi32>) -> tensor<4x2xi32>
  func.return %0 : tensor<4x2xi32>
}
