// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// depthwise_conv2d + tosa.rescale chain: quantized conv pipeline.
//
// depthwise_conv2d produces int32 accumulator output, then tosa.rescale
// applies the per-layer quantization: (acc - in_zp) * mult >> shift + out_zp
// with int8 saturation. This is the standard quantized inference pattern
// matching what coralnpu/sw/opt/litert-micro does via PostprocessAcc().
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @dw_rescale
// Depthwise part:
// CHECK: coralnpu.lw
// CHECK: coralnpu.mul
// CHECK: coralnpu.add
// CHECK: coralnpu.sw
// Rescale part (must follow the sw from depthwise):
// CHECK: coralnpu.lw
// CHECK: coralnpu.mulh
// CHECK: coralnpu.sra
// CHECK: coralnpu.sw

// ASM-LABEL: # CoralNPU Assembly
// ASM: vle32.v
// ASM: mul
// ASM: sw
// ASM: add

func.func @dw_rescale(%in: tensor<1x1x1x4xi8>, %wt: tensor<1x1x4x1xi8>,
                      %bias: tensor<4xi32>, %mult: tensor<1xi32>, %shift: tensor<1xi8>,
                      %izp: tensor<1xi8>, %wzp: tensor<1xi8>,
                      %izp32: tensor<1xi32>, %ozp: tensor<1xi8>) -> tensor<1x1x1x4xi8> {
  %dw = tosa.depthwise_conv2d %in, %wt, %bias, %izp, %wzp {
    acc_type = i32,
    dilation = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>,
    stride = array<i64: 1, 1>
  } : (tensor<1x1x1x4xi8>, tensor<1x1x4x1xi8>, tensor<4xi32>, tensor<1xi8>, tensor<1xi8>)
    -> tensor<1x1x1x4xi32>
  %out = tosa.rescale %dw, %mult, %shift, %izp32, %ozp {
    input_unsigned = false,
    output_unsigned = false,
    per_channel = false,
    rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>,
    scale32 = true
  } : (tensor<1x1x1x4xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x1x4xi8>
  func.return %out : tensor<1x1x1x4xi8>
}
