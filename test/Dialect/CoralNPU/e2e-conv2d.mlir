// RUN: circt-opt --tosa-to-coralnpu --coralnpu-stripmine --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s

// End-to-end element-wise add through the full CoralNPU pipeline. The TOSA
// add lowers to a real load / compute / store sequence rather than a
// placeholder, so the emitted assembly contains the full vle/vadd/vse chain.

// CHECK: # CoralNPU Assembly (target: real google-coral/coralnpu ISA)
// CHECK: _start:
// CHECK: vle32.v
// CHECK: vadd.vv
// CHECK: vse32.v
// CHECK: ret
// CHECK: # End of assembly

func.func @e2e_add(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  func.return %0 : tensor<4x4xi32>
}
