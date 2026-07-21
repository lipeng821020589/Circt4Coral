// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s --check-prefix=ASM

// GEMV: M=1, K=4, N=4 — i32 path: vle+vmul+vredsum per output channel.
// CHECK-LABEL: @gemv_i32_4x4
// CHECK-NOT: coralnpu.outer_product
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vmul
// CHECK: coralnpu.vredsum
// ASM-LABEL: _start:
// ASM: vle32.v
// ASM: vmul

func.func @gemv_i32_4x4(%a: tensor<1x1x4xi32>, %b: tensor<1x4x4xi32>,
                         %az: tensor<1xi32>, %bz: tensor<1xi32>) -> tensor<1x1x4xi32> {
  %0 = tosa.matmul %a, %b, %az, %bz
    : (tensor<1x1x4xi32>, tensor<1x4x4xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x4xi32>
  func.return %0 : tensor<1x1x4xi32>
}

// GEMV K=8 — still vfmacc path (M=1 regardless of K size).
// CHECK-LABEL: @gemv_i32_8x8
// CHECK-NOT: coralnpu.outer_product
// CHECK: coralnpu.vle32
func.func @gemv_i32_8x8(%a: tensor<1x1x8xi32>, %b: tensor<1x8x8xi32>,
                         %az: tensor<1xi32>, %bz: tensor<1xi32>) -> tensor<1x1x8xi32> {
  %0 = tosa.matmul %a, %b, %az, %bz
    : (tensor<1x1x8xi32>, tensor<1x8x8xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x8xi32>
  func.return %0 : tensor<1x1x8xi32>
}
