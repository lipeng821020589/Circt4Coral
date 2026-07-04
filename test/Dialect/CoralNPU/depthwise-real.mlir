// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// tosa.depthwise_conv2d real data-flow lowering.
//
// Each channel is processed as a dot product:
//   out[c] = sum_{kh,kw}(in[c,kh,kw] * wt[kh,kw,c,0]) + bias[c]
//
// Lowering (v0.4.0+): per-tile VWMACCOp (vwmacc.vv, int8 widening MAC)
// + VRedSumOp → scalar fold → bias load → sw.
//===----------------------------------------------------------------------===//

// 1x1x4x4 input, 1x1x4x1 weight (pointwise depthwise): single tile pair.
// CHECK-LABEL: @dw_pointwise
// CHECK: coralnpu.vsetvl e32, m1
// CHECK: %[[IN0:.*]] = coralnpu.vle32
// CHECK: %[[WT0:.*]] = coralnpu.vle32
// CHECK: %[[ACC0:.*]] = coralnpu.vwmacc %[[IN0]], %[[WT0]]
// CHECK: %[[S0:.*]] = coralnpu.vredsum %[[ACC0]]
// CHECK: %{{.*}} = coralnpu.lw
// CHECK: %{{.*}} = coralnpu.add %[[S0]]
// CHECK: coralnpu.sw
// ASM-LABEL: # CoralNPU Assembly
// ASM: vle32.v
// ASM: vle32.v
// ASM: vwmacc.vv
// ASM: vmv.v.i
// ASM: vredsum.vs
// ASM: vmv.x.s
// ASM: lw
// ASM: add
// ASM: sw
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
