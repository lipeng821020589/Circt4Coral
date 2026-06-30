// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s

//===----------------------------------------------------------------------===//
// tosa.avg_pool2d cross-tile tiling.
//
// For an input with N elements the lowering emits k = ceil(N / vregCap)
// register-sized vle32 loads at PROGRESSIVE addresses (tile i at base + i*16
// bytes), one vredsum per tile, then folds the k partial sums with a scalar
// add chain before dividing by the kernel area. This asserts the k>=2 path
// with explicit progressive load addresses.
//===----------------------------------------------------------------------===//

// N=8 at e32 (vregCap=4) => k=2. Tile 0 loads from 0x10000 (65536), tile 1
// from 0x10010 (65552 = base + 16 bytes). Two vredsum partials, folded by a
// scalar add, then div.
// CHECK-LABEL: @avgpool_tiled
// CHECK: %{{.*}} = coralnpu.li 65536
// CHECK: %[[T0:.*]] = coralnpu.vle32
// CHECK: %{{.*}} = coralnpu.li 65552
// CHECK: %[[T1:.*]] = coralnpu.vle32
// CHECK: %[[P0:.*]] = coralnpu.vredsum %[[T0]]
// CHECK: %[[P1:.*]] = coralnpu.vredsum %[[T1]]
// CHECK: %[[SUM:.*]] = coralnpu.add %[[P0]], %[[P1]]
// CHECK: %{{.*}} = coralnpu.div %[[SUM]]
// CHECK: coralnpu.sw
func.func @avgpool_tiled(%in: tensor<1x2x2x2xi32>, %izp: tensor<1xi32>, %ozp: tensor<1xi32>) -> tensor<1x1x1x2xi32> {
  %0 = tosa.avg_pool2d %in, %izp, %ozp {acc_type = i32, kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>} : (tensor<1x2x2x2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x1x2xi32>
  func.return %0 : tensor<1x1x1x2xi32>
}
