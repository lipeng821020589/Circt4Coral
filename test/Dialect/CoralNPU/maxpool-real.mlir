// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// tosa.max_pool2d real data flow with VRedMaxOp.
//
// Uses vredmax.vs (signed integer max reduction) instead of the old
// vredsum placeholder. Per-tile max scalars are folded with a branchless
// scalar max chain: max(a,b) = a + ((b-a) & ~((b-a)>>31)).
// The final result is stored via coralnpu.sw (scalar word store).
//===----------------------------------------------------------------------===//

// k=1 (N=4, vregCap=4): single vredmax, no scalar fold needed.
// CHECK-LABEL: @maxpool_k1
// CHECK: %[[T0:.*]] = coralnpu.vle32
// CHECK: %[[M0:.*]] = coralnpu.vredmax %[[T0]]
// CHECK: coralnpu.sw %[[M0]]
// ASM-LABEL: # CoralNPU Assembly
// ASM: vredmax.vs
// ASM: vmv.x.s
// ASM: sw
func.func @maxpool_k1(%in: tensor<1x2x2x1xi32>) -> tensor<1x1x1x1xi32> {
  %0 = tosa.max_pool2d %in {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, nan_mode = "PROPAGATE"} : (tensor<1x2x2x1xi32>) -> tensor<1x1x1x1xi32>
  func.return %0 : tensor<1x1x1x1xi32>
}

// k=2 (N=8, vregCap=4 at e32): two vredmax, one branchless-max fold.
// CHECK-LABEL: @maxpool_k2
// CHECK: %[[T0:.*]] = coralnpu.vle32
// CHECK: %[[T1:.*]] = coralnpu.vle32
// CHECK: %[[M0:.*]] = coralnpu.vredmax %[[T0]]
// CHECK: %[[M1:.*]] = coralnpu.vredmax %[[T1]]
// CHECK: coralnpu.sub
// CHECK: coralnpu.sra
// CHECK: coralnpu.xor
// CHECK: coralnpu.and
// CHECK: coralnpu.add
// CHECK: coralnpu.sw
// ASM: vredmax.vs
// ASM: vredmax.vs
// ASM: sub
// ASM: sra
// ASM: xor
// ASM: and
// ASM: add
// ASM: sw
func.func @maxpool_k2(%in: tensor<1x2x2x2xi32>) -> tensor<1x1x1x2xi32> {
  %0 = tosa.max_pool2d %in {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, nan_mode = "PROPAGATE"} : (tensor<1x2x2x2xi32>) -> tensor<1x1x1x2xi32>
  func.return %0 : tensor<1x1x1x2xi32>
}
