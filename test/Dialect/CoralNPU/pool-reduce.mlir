// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s

//===----------------------------------------------------------------------===//
// tosa.avg_pool2d / max_pool2d real data flow.
//
// Pooling loads the real input window (vle32) and reduce-sums it (vredsum).
// avg_pool now TILES the input into k = ceil(N / vregCap) register-sized
// loads, gives each tile its own vredsum partial sum, folds the partials
// with a scalar add chain, then divides by the kernel area. max_pool stays a
// single-vredsum placeholder (the dialect has no vredmax).
//===----------------------------------------------------------------------===//

// avg_pool2d over N=8 (k=2 tiles at e32): two vle32 + two vredsum, folded by
// a scalar add chain, divided by the 2x2 kernel area (4), then stored.
// CHECK-LABEL: @ap
// CHECK: %[[T0:.*]] = coralnpu.vle32
// CHECK: %[[T1:.*]] = coralnpu.vle32
// CHECK: %[[P0:.*]] = coralnpu.vredsum %[[T0]]
// CHECK: %[[P1:.*]] = coralnpu.vredsum %[[T1]]
// CHECK: %[[SUM:.*]] = coralnpu.add %[[P0]], %[[P1]]
// CHECK: %[[N:.*]] = coralnpu.li 4
// CHECK: %[[AVG:.*]] = coralnpu.div %[[SUM]], %[[N]]
// CHECK: coralnpu.sw %[[AVG]]
func.func @ap(%in: tensor<1x2x2x2xi32>, %izp: tensor<1xi32>, %ozp: tensor<1xi32>) -> tensor<1x1x1x2xi32> {
  %0 = tosa.avg_pool2d %in, %izp, %ozp {acc_type = i32, kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>} : (tensor<1x2x2x2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x1x2xi32>
  func.return %0 : tensor<1x1x1x2xi32>
}

// max_pool2d: load real input, reduce, store. Unchanged placeholder form.
// CHECK-LABEL: @mp
// CHECK: %[[IN2:.*]] = coralnpu.vle32
// CHECK: %[[R2:.*]] = coralnpu.vredsum %[[IN2]]
// CHECK: coralnpu.vse32 %[[R2]]
func.func @mp(%in: tensor<1x8x8x4xi32>) -> tensor<1x4x4x4xi32> {
  %0 = tosa.max_pool2d %in {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, nan_mode = "PROPAGATE"} : (tensor<1x8x8x4xi32>) -> tensor<1x4x4x4xi32>
  func.return %0 : tensor<1x4x4x4xi32>
}
