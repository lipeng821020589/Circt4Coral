// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// CHECK-LABEL: @rmsnorm
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vmul
// CHECK: coralnpu.vredsum
// CHECK: coralnpu.vmul

func.func @rmsnorm(%x: tensor<4xi32>, %gamma: tensor<4xi32>) -> tensor<4xi32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %x_sq = tosa.mul %x, %x, %shift : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
  %sum  = tosa.reduce_sum %x_sq { axis = 0 : i32 } : (tensor<4xi32>) -> tensor<1xi32>
  %eps  = "tosa.const"() <{values = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  %seps = tosa.add %sum, %eps : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %rsq  = tosa.rsqrt %seps : (tensor<1xi32>) -> tensor<1xi32>
  %norm = tosa.mul %x, %rsq, %shift : (tensor<4xi32>, tensor<1xi32>, tensor<1xi8>) -> tensor<4xi32>
  %out  = tosa.mul %norm, %gamma, %shift : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
  func.return %out : tensor<4xi32>
}
