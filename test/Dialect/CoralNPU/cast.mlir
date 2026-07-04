// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// tosa.cast lowering.
//
// Widening (i8->i32): load with source SEW, passthrough — sign-extension is
//   implicit in the i32 register file.
// Narrowing (i32->i8): load tiles, mask with 0xFF via scalar and, store.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cast_widen_i8_to_i32
// CHECK: coralnpu.vsetvl e8, m1
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vse32
// ASM-LABEL: # CoralNPU Assembly
// ASM: vle32.v
// ASM: vse32.v
func.func @cast_widen_i8_to_i32(%a: tensor<4xi8>) -> tensor<4xi32> {
  %0 = tosa.cast %a : (tensor<4xi8>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @cast_narrow_i32_to_i8
// CHECK: coralnpu.vsetvl e32, m1
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vredsum
// CHECK: coralnpu.and
// CHECK: coralnpu.sw
// ASM-LABEL: # CoralNPU Assembly
// ASM: vredsum.vs
// ASM: andi
// ASM: sw
func.func @cast_narrow_i32_to_i8(%a: tensor<4xi32>) -> tensor<4xi8> {
  %0 = tosa.cast %a : (tensor<4xi32>) -> tensor<4xi8>
  func.return %0 : tensor<4xi8>
}
