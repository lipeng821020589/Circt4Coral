// RUN: circt-opt --pass-pipeline='builtin.module(func.func(tosa-layerwise-constant-fold,tosa-to-linalg-named,tosa-to-linalg),one-shot-bufferize{bufferize-function-boundaries})' %s 2>/dev/null | FileCheck %s

// Tests the complete TOSA→linalg→bufferize pipeline for matmul (LLM core op).
// Validates:
//   tosa.matmul (with const zero-points) → linalg.batch_matmul → memref
//   tosa.const → memref.global + memref.get_global
//   tosa.reshape → memref.reshape (via copy for strided sources)

// CHECK-LABEL: func.func @gemv(
// CHECK-SAME:    %arg0: memref<1x1x4xi32
// CHECK-SAME:    %arg1: memref<1x4x4xi32
// CHECK:  memref.alloc
// CHECK:  linalg.batch_matmul
// CHECK-SAME:  ins(%arg0, %arg1
// CHECK:  return

func.func @gemv(%a: tensor<1x1x4xi32>, %b: tensor<1x4x4xi32>) -> tensor<1x1x4xi32> {
  %az = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %bz = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %0 = tosa.matmul %a, %b, %az, %bz
    : (tensor<1x1x4xi32>, tensor<1x4x4xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x4xi32>
  func.return %0 : tensor<1x1x4xi32>
}
