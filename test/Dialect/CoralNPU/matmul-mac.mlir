// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s --check-prefix=ASM

//===----------------------------------------------------------------------===//
// tosa.matmul / tosa.conv2d real outer-product MAC lowering.
//
// The matrix ops now carry real data flow: operands are loaded from their
// TCM slots, fed to the 8x8 outer-product accumulator, the result column is
// read back, and stored. This replaces the previous "li 0" placeholder.
//===----------------------------------------------------------------------===//

// MatMul: load A and B, MAC into the accumulator, read it back, store.
// CHECK-LABEL: @mm
// CHECK: %[[A:.*]] = coralnpu.vle32
// CHECK: %[[B:.*]] = coralnpu.vle32
// CHECK: %[[ACC:.*]] = coralnpu.outer_product %[[A]], %[[B]], %{{.*}} stripmine = 4
// CHECK: %[[R:.*]] = coralnpu.accread %[[ACC]]
// CHECK: coralnpu.vse32 %[[R]]

// In assembly the MAC is realized via the KSCM/KISA custom CSRs.
// ASM-LABEL: _start:
// ASM: vle32.v
// ASM: vle32.v
// ASM: csrw{{.*}}KSCM0
// ASM: csrw{{.*}}KSCM1
// ASM: csrw{{.*}}KISA
// ASM: csrr{{.*}}accread
// ASM: vse32.v

func.func @mm(%a: tensor<1x8x8xi8>, %b: tensor<1x8x8xi8>, %azp: tensor<1xi8>, %bzp: tensor<1xi8>) -> tensor<1x8x8xi32> {
  %0 = tosa.matmul %a, %b, %azp, %bzp : (tensor<1x8x8xi8>, tensor<1x8x8xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8xi32>
  func.return %0 : tensor<1x8x8xi32>
}
