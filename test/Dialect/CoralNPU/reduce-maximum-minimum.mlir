// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// tosa.reduce_sum, tosa.reduce_max, tosa.maximum, tosa.minimum lowering.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_reduce_sum
// CHECK: coralnpu.vsetvl e32, m1
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vredsum
// CHECK: coralnpu.sw
// ASM-LABEL: # CoralNPU Assembly
// ASM: vredsum.vs
// ASM: vmv.x.s
// ASM: sw
func.func @test_reduce_sum(%a: tensor<4xi32>) -> tensor<1xi32> {
  %0 = tosa.reduce_sum %a {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}

// CHECK-LABEL: @test_reduce_max
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vredmax
// CHECK: coralnpu.sw
// ASM-LABEL: # CoralNPU Assembly
// ASM: vredmax.vs
// ASM: vmv.x.s
// ASM: sw
func.func @test_reduce_max(%a: tensor<4xi32>) -> tensor<1xi32> {
  %0 = tosa.reduce_max %a {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}

// CHECK-LABEL: @test_maximum
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vmax_vv
// CHECK: coralnpu.vse32
// ASM-LABEL: # CoralNPU Assembly
// ASM: vmax.vv
func.func @test_maximum(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tosa.maximum %a, %b : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @test_minimum
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vmin_vv
// CHECK: coralnpu.vse32
// ASM-LABEL: # CoralNPU Assembly
// ASM: vmin.vv
func.func @test_minimum(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tosa.minimum %a, %b : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}
