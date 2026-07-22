// RUN: circt-opt --pass-pipeline='builtin.module(func.func(tosa-to-linalg),one-shot-bufferize{bufferize-function-boundaries})' %s 2>/dev/null | FileCheck %s

// Tests the standard MLIR TOSA→linalg→bufferize pipeline.
// This is the community-standard path addressing RFC Q2 (lowering too steep):
//   TOSA → linalg.generic (tosa-to-linalg)
//        → memref-based (one-shot-bufferize)
// The result is a proper memref IR suitable for further lowering.

// CHECK-LABEL: func.func @add(
// CHECK-SAME:    %arg0: memref<4xi32
// CHECK-SAME:    %arg1: memref<4xi32
// CHECK:  memref.alloc
// CHECK:  linalg.generic
// CHECK-SAME:  ins(%arg0, %arg1
// CHECK-SAME:  outs(%alloc
// CHECK:    arith.addi
// CHECK:    linalg.yield

func.func @add(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tosa.add %a, %b : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}
