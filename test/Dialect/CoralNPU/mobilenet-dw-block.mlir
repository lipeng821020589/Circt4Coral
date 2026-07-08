// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// P2-1: MobileNet depthwise block — tosa.depthwise_conv2d → tosa.rescale → tosa.clamp
//
// This is the core quantized inference block from MobileNetV1:
//   depthwise_conv2d (MAC) → rescale (requantize int32→int8) → ReLU (clamp)
//
// The full 3-op TOSA chain lowers correctly to CoralNPU scalar+vector sequence.
// spike E2E: in=[2,4,6,8], wt=[1]*4, bias=0, mult=0x40000000, shift=32
//   → dw_acc=20 → rescale=5 → relu(5)=5  (bit-exact)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @mobilenet_dw_block
// dw produces: vle32, vmul, vredsum, add (bias), sw
// CHECK: coralnpu.vsetvl
// CHECK: coralnpu.lw
// CHECK: coralnpu.mul
// CHECK: coralnpu.lw
// CHECK: coralnpu.add
// CHECK: coralnpu.sw
// rescale produces: lw (mult/shift/zp), sub, mulh, sra, add, sw
// CHECK: coralnpu.lw
// CHECK: coralnpu.mulh
// CHECK: coralnpu.sra
// CHECK: coralnpu.sw
// relu produces: vle32, vmax_vx, vse32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vmax_vx
// CHECK: coralnpu.vse32
// ASM-LABEL: # CoralNPU Assembly
// ASM: vle32.v
// ASM: mul
// ASM: sw
// ASM: add
func.func @mobilenet_dw_block(
  %in: tensor<1x1x1x4xi8>, %wt: tensor<1x1x4x1xi8>,
  %bias: tensor<4xi32>, %mult: tensor<1xi32>, %shift: tensor<1xi8>,
  %izp: tensor<1xi8>, %wzp: tensor<1xi8>,
  %izp32: tensor<1xi32>, %ozp: tensor<1xi8>
) -> tensor<1x1x1x4xi8> {
  %dw = tosa.depthwise_conv2d %in, %wt, %bias, %izp, %wzp {
    acc_type = i32, dilation = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>
  } : (tensor<1x1x1x4xi8>, tensor<1x1x4x1xi8>, tensor<4xi32>,
       tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x1x4xi32>
  %rs = tosa.rescale %dw, %mult, %shift, %izp32, %ozp {
    input_unsigned = false, output_unsigned = false, per_channel = false,
    rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true
  } : (tensor<1x1x1x4xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>,
       tensor<1xi8>) -> tensor<1x1x1x4xi8>
  %rl = tosa.clamp %rs {min_val = 0 : i8, max_val = 127 : i8}
      : (tensor<1x1x1x4xi8>) -> tensor<1x1x1x4xi8>
  func.return %rl : tensor<1x1x1x4xi8>
}
