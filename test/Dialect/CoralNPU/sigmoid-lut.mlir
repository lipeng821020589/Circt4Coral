// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// tosa.sigmoid → LUT lookup in TCM slot 9 (kLutSlot = 0x19000).
//
// LUT index = input + 128; address = 0x19000 + idx * 4.
// Matches coralnpu/sw/opt/litert-micro/logistic.cc LUT-based approach.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sigmoid_lut
// CHECK: coralnpu.vsetvl e32, m1
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vredsum
// CHECK: coralnpu.add
// CHECK: coralnpu.mul
// CHECK: coralnpu.lw
// CHECK: coralnpu.vse32

// ASM-LABEL: # CoralNPU Assembly
// ASM: vle32.v
// ASM: vredsum.vs
// ASM: vmv.x.s
// ASM: lw
// ASM: vse32.v

func.func @sigmoid_lut(%in: tensor<4xi8>) -> tensor<4xi8> {
  %out = tosa.sigmoid %in : (tensor<4xi8>) -> tensor<4xi8>
  func.return %out : tensor<4xi8>
}
