// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// tosa.rescale → (x - in_zp) * mult >> shift + out_zp, saturate to int8
//
// Implements the TOSA quantization rescale operation:
//   out = saturate_int8((in - in_zp) * multiplier >> shift + out_zp)
//
// Uses ScalarLwOp to load mult/shift/zp from TCM slots, VRedSumOp + scalar
// arithmetic for the per-tile calculation, ScalarMulhOp for the high-32-bit
// multiply, then branchless int8 clamp.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @rescale_scalar
// CHECK: coralnpu.lw
// CHECK: coralnpu.lw
// CHECK: coralnpu.lw
// CHECK: coralnpu.sub
// CHECK: coralnpu.mulh
// CHECK: coralnpu.sra
// CHECK: coralnpu.add
// CHECK: coralnpu.sw
// ASM-LABEL: # CoralNPU Assembly
// ASM: vle32.v
// ASM: mul
// ASM: sw
// ASM: add
func.func @rescale_scalar(%in: tensor<4xi32>, %mult: tensor<1xi32>, %shift: tensor<1xi8>, %izp: tensor<1xi32>, %ozp: tensor<1xi8>) -> tensor<4xi8> {
  %out = tosa.rescale %in, %mult, %shift, %izp, %ozp {
    input_unsigned = false,
    output_unsigned = false,
    per_channel = false,
    rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>,
    scale32 = true
  } : (tensor<4xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<4xi8>
  func.return %out : tensor<4xi8>
}
