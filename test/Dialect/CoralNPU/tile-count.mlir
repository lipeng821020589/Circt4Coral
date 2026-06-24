// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s

// This test asserts that the --tosa-to-coralnpu pass lowers tosa.add into
// exactly ceil(N / vregCap) coralnpu.vadd instructions, where vregCap is
// the number of elements per 128-bit vector register for the element type.

// CHECK-LABEL: func.func @add_4xi32
// CHECK-COUNT-1: coralnpu.vadd %{{.*}}, %{{.*}} : i32
// CHECK-NOT: coralnpu.vadd
func.func @add_4xi32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: func.func @add_8xi32
// CHECK-COUNT-2: coralnpu.vadd %{{.*}}, %{{.*}} : i32
// CHECK-NOT: coralnpu.vadd
func.func @add_8xi32(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// CHECK-LABEL: func.func @add_16xi32
// CHECK-COUNT-4: coralnpu.vadd %{{.*}}, %{{.*}} : i32
// CHECK-NOT: coralnpu.vadd
func.func @add_16xi32(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> tensor<16xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
  func.return %0 : tensor<16xi32>
}

// CHECK-LABEL: func.func @add_16xi8
// CHECK-COUNT-1: coralnpu.vadd %{{.*}}, %{{.*}} : i32
// CHECK-NOT: coralnpu.vadd
func.func @add_16xi8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xi8>, tensor<16xi8>) -> tensor<16xi8>
  func.return %0 : tensor<16xi8>
}

// CHECK-LABEL: func.func @add_64xi8
// CHECK-COUNT-4: coralnpu.vadd %{{.*}}, %{{.*}} : i32
// CHECK-NOT: coralnpu.vadd
func.func @add_64xi8(%arg0: tensor<64xi8>, %arg1: tensor<64xi8>) -> tensor<64xi8> {
  %0 = tosa.add %arg0, %arg1 : (tensor<64xi8>, tensor<64xi8>) -> tensor<64xi8>
  func.return %0 : tensor<64xi8>
}
