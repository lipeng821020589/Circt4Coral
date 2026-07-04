// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s

//===----------------------------------------------------------------------===//
// arithmetic_right_shift / equal / greater / greater_equal /
// logical_and / logical_or / logical_not / select
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_ars
// CHECK: coralnpu.vredsum
// CHECK: coralnpu.vredsum
// CHECK: coralnpu.sra
// CHECK: coralnpu.sw
func.func @test_ars(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tosa.arithmetic_right_shift %a, %b {round = false}
      : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @test_equal
// CHECK: coralnpu.slt
// CHECK: coralnpu.slt
// CHECK: coralnpu.sub
// CHECK: coralnpu.sw
func.func @test_equal(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi1> {
  %0 = tosa.equal %a, %b : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// CHECK-LABEL: @test_greater
// CHECK: coralnpu.slt
// CHECK: coralnpu.sw
func.func @test_greater(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi1> {
  %0 = tosa.greater %a, %b : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// CHECK-LABEL: @test_greater_equal
// CHECK: coralnpu.slt
// CHECK: coralnpu.sub
// CHECK: coralnpu.sw
func.func @test_greater_equal(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi1> {
  %0 = tosa.greater_equal %a, %b : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// tosa.logical_not requires i1 input
// CHECK-LABEL: @test_logical_not
// CHECK: coralnpu.slt
// CHECK: coralnpu.or
// CHECK: coralnpu.sub
func.func @test_logical_not(%a: tensor<4xi1>) -> tensor<4xi1> {
  %0 = tosa.logical_not %a : (tensor<4xi1>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// tosa.logical_and / or require i1 inputs
// CHECK-LABEL: @test_logical_and
// CHECK: coralnpu.and
// CHECK: coralnpu.sw
func.func @test_logical_and(%a: tensor<4xi1>, %b: tensor<4xi1>) -> tensor<4xi1> {
  %0 = tosa.logical_and %a, %b : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// tosa.select: condition must be i1
// CHECK-LABEL: @test_select
// CHECK: coralnpu.sub
// CHECK: coralnpu.and
// CHECK: coralnpu.add
// CHECK: coralnpu.sw
func.func @test_select(%cond: tensor<1xi1>, %a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  %0 = tosa.select %cond, %a, %b : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}
