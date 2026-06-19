// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s

//===----------------------------------------------------------------------===//
// tosa.avg_pool2d / max_pool2d real data flow.
//
// Pooling now loads the real input window (vle32), reduces it (vredsum), and
// stores the result (vse32). avg_pool divides the sum by the kernel area read
// from the op's kernel attribute. Previously these were zero placeholders.
//===----------------------------------------------------------------------===//

// avg_pool2d over a 2x2 window: load, reduce-sum, divide by 4, store.
// CHECK-LABEL: @ap
// CHECK: %[[IN:.*]] = coralnpu.vle32
// CHECK: %[[SUM:.*]] = coralnpu.vredsum %[[IN]]
// CHECK: %[[N:.*]] = coralnpu.li 4
// CHECK: %[[AVG:.*]] = coralnpu.div %[[SUM]], %[[N]]
// CHECK: coralnpu.vse32 %[[AVG]]
func.func @ap(%in: tensor<1x8x8x4xi32>, %izp: tensor<1xi32>, %ozp: tensor<1xi32>) -> tensor<1x4x4x4xi32> {
  %0 = tosa.avg_pool2d %in, %izp, %ozp {acc_type = i32, kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>} : (tensor<1x8x8x4xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x4x4x4xi32>
  func.return %0 : tensor<1x4x4x4xi32>
}

// max_pool2d: load real input, reduce, store.
// CHECK-LABEL: @mp
// CHECK: %[[IN2:.*]] = coralnpu.vle32
// CHECK: %[[R2:.*]] = coralnpu.vredsum %[[IN2]]
// CHECK: coralnpu.vse32 %[[R2]]
func.func @mp(%in: tensor<1x8x8x4xi32>) -> tensor<1x4x4x4xi32> {
  %0 = tosa.max_pool2d %in {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, nan_mode = "PROPAGATE"} : (tensor<1x8x8x4xi32>) -> tensor<1x4x4x4xi32>
  func.return %0 : tensor<1x4x4x4xi32>
}
