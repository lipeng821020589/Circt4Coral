// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s

// This test verifies that the stripmine factor is derived from the tile size
// and SEW as ceil(numElements / capacity), clamped to {1, 2, 4}. The factor
// is only printed when it is not equal to 1. The vadd carrier is always i32
// (the vector-register value), independent of the tensor element type.

// CHECK-LABEL: func.func @add_4xi32
// CHECK: coralnpu.vadd %{{.*}}, %{{.*}} : i32
func.func @add_4xi32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: func.func @add_8xi32
// CHECK: coralnpu.vadd stripmine = 2 %{{.*}}, %{{.*}} : i32
func.func @add_8xi32(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// CHECK-LABEL: func.func @add_16xi32
// CHECK: coralnpu.vadd stripmine = 4 %{{.*}}, %{{.*}} : i32
func.func @add_16xi32(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> tensor<16xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
  func.return %0 : tensor<16xi32>
}

// CHECK-LABEL: func.func @add_16xi8
// CHECK: coralnpu.vadd %{{.*}}, %{{.*}} : i32
func.func @add_16xi8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xi8>, tensor<16xi8>) -> tensor<16xi8>
  func.return %0 : tensor<16xi8>
}

// CHECK-LABEL: func.func @add_64xi8
// CHECK: coralnpu.vadd stripmine = 4 %{{.*}}, %{{.*}} : i32
func.func @add_64xi8(%arg0: tensor<64xi8>, %arg1: tensor<64xi8>) -> tensor<64xi8> {
  %0 = tosa.add %arg0, %arg1 : (tensor<64xi8>, tensor<64xi8>) -> tensor<64xi8>
  func.return %0 : tensor<64xi8>
}
