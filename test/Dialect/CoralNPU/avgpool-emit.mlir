// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s

//===----------------------------------------------------------------------===//
// avg_pool2d cross-tile reduction → real RISC-V assembly.
//
// vredsum.vs leaves its reduction result in vector-register lane 0; crossing
// into the scalar domain (where the partial-sum add chain and the area div
// live) requires a vmv.x.s move. This asserts that each vredsum is followed
// by a vmv.x.s into a scalar register, and that the scalar add folding the
// two partial sums reads those moved scalar registers (NOT the default x5).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: # CoralNPU Assembly
// CHECK: _start:
// CHECK: vle32.v
// CHECK: vle32.v
// CHECK: vredsum.vs [[VD0:v[0-9]+]]
// CHECK: vmv.x.s [[XS0:x[0-9]+]], [[VD0]]
// CHECK: vredsum.vs [[VD1:v[0-9]+]]
// CHECK: vmv.x.s [[XS1:x[0-9]+]], [[VD1]]
// CHECK: add    [[SUM:x[0-9]+]], [[XS0]], [[XS1]]
// CHECK: div    {{x[0-9]+}}, [[SUM]]
// CHECK: ret
func.func @avgpool_emit(%in: tensor<1x2x2x2xi32>, %izp: tensor<1xi32>, %ozp: tensor<1xi32>) -> tensor<1x1x1x2xi32> {
  %0 = tosa.avg_pool2d %in, %izp, %ozp {acc_type = i32, kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>} : (tensor<1x2x2x2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x1x2xi32>
  func.return %0 : tensor<1x1x1x2xi32>
}
