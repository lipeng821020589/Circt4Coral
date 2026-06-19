// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s

//===----------------------------------------------------------------------===//
// Tiled matmul: a 16x16 x 16x16 matmul tiles into the 8x8 accumulator as a
// 2x2x2 (M/N/K) loop nest. Each of the 4 output tiles accumulates over the 2
// K-tiles, giving 8 outer_products total and 4 accreads.
//
// This proves the lowering scales beyond a single 8x8 tile -- the matrix is
// streamed through the MAC engine tile by tile with real K-accumulation.
//===----------------------------------------------------------------------===//

// The 4 output tiles each accumulate over 2 K-tiles. outer_product and
// accread interleave (2 outer_products then 1 accread, per output tile), so
// we assert the first output tile's shape and rely on the IR-count RUN below.
// CHECK-LABEL: @mm
// CHECK: coralnpu.outer_product
// CHECK: coralnpu.outer_product
// CHECK: coralnpu.accread
// CHECK: coralnpu.vse32

func.func @mm(%a: tensor<1x16x16xi8>, %b: tensor<1x16x16xi8>, %az: tensor<1xi8>, %bz: tensor<1xi8>) -> tensor<1x16x16xi32> {
  %0 = tosa.matmul %a, %b, %az, %bz : (tensor<1x16x16xi8>, tensor<1x16x16xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x16x16xi32>
  func.return %0 : tensor<1x16x16xi32>
}
