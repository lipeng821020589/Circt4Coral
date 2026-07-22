// RUN: circt-opt --one-shot-bufferize='allow-unknown-ops bufferize-function-boundaries' --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s

// Validates that the bufferization → CoralNPU lowering pipeline works:
// 1. one-shot-bufferize converts tensor function boundaries to memref
// 2. tosa-to-coralnpu still handles the TOSA ops (via bufferization.to_tensor)
//
// The function signature changes from tensor<4xi32> to memref<4xi32, strided<[?]>>
// which is the MLIR-standard representation. This addresses RFC Q2 (lowering
// too steep) by inserting a bufferization layer between TOSA and CoralNPU.

// CHECK-LABEL: func.func @bufferized_add(
// CHECK-SAME:    %arg0: memref<4xi32
// CHECK-SAME:    %arg1: memref<4xi32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vadd
// CHECK: coralnpu.vse32

func.func @bufferized_add(%a: tensor<4xi32>, %b: tensor<4xi32>) -> i32 {
  %0 = tosa.add %a, %b : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = tosa.reduce_sum %0 { axis = 0 : i32 } : (tensor<4xi32>) -> tensor<1xi32>
  %c0 = arith.constant 0 : i32
  func.return %c0 : i32
}
