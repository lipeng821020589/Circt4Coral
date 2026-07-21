// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s --check-prefix=ASM

// int8 GEMV M=1: uses MAC CSR path (outer_product), not vle+vmul+vredsum.
// CHECK-LABEL: @gemv_int8
// CHECK: coralnpu.outer_product
// CHECK: coralnpu.accread
// CHECK-NOT: coralnpu.vmul
// ASM-LABEL: _start:
// ASM: csrw

func.func @gemv_int8(%a: tensor<1x1x8xi8>, %b: tensor<1x8x8xi8>,
                     %az: tensor<1xi8>, %bz: tensor<1xi8>) -> tensor<1x1x8xi32> {
  %0 = tosa.matmul %a, %b, %az, %bz
    : (tensor<1x1x8xi8>, tensor<1x8x8xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x8xi32>
  func.return %0 : tensor<1x1x8xi32>
}
