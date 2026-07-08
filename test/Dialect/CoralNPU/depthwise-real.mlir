// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// tosa.depthwise_conv2d real data-flow lowering.
//
// Each channel: out[c] = sum_{kh,kw}(in[c,kh,kw] * wt[kh,kw,c,0]) + bias[c]
// Lowering: per-tile VMulOp + VRedSumOp → scalar fold → bias load → sw.
//===----------------------------------------------------------------------===//

// 1x1x4x4 input, 1x1x4x1 weight (pointwise depthwise): single tile pair.
// CHECK-LABEL: @dw_pointwise
// CHECK: coralnpu.vsetvl e32, m1
// CHECK: coralnpu.lw
// CHECK: coralnpu.lw
// CHECK: coralnpu.mul
// CHECK: coralnpu.add
// CHECK: coralnpu.lw
// CHECK: coralnpu.add
// CHECK: coralnpu.sw
// ASM-LABEL: # CoralNPU Assembly
// ASM: vle32.v
// ASM: mul
// ASM: sw
// ASM: add
func.func @dw_pointwise(%in: tensor<1x4x4x4xi8>, %wt: tensor<1x1x4x1xi8>,
                         %bias: tensor<4xi32>, %izp: tensor<1xi8>, %wzp: tensor<1xi8>)
    -> tensor<1x4x4x4xi32> {
  %0 = tosa.depthwise_conv2d %in, %wt, %bias, %izp, %wzp {
    acc_type = i32,
    dilation = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>,
    stride = array<i64: 1, 1>
  } : (tensor<1x4x4x4xi8>, tensor<1x1x4x1xi8>, tensor<4xi32>,
       tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x4xi32>
  func.return %0 : tensor<1x4x4x4xi32>
}
