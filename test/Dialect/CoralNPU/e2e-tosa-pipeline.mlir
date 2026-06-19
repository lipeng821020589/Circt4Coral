// RUN: circt-opt --tosa-to-coralnpu --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s

//===----------------------------------------------------------------------===//
// Full ML Compiler Pipeline: TOSA → CoralNPU → RISC-V Assembly
//
// Input:  TOSA element-wise add (result is returned, so it is live)
// Output: Real RISC-V assembly with V extension — the add lowers to a real
//         vle32 / vadd.vv / vse32 sequence over the operand TCM slots.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: # CoralNPU Assembly
// CHECK: _start:
// CHECK: vsetivli
// CHECK: vle32.v
// CHECK: vadd.vv
// CHECK: vse32.v
// CHECK: ret

func.func @tosa_add(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  func.return %0 : tensor<4x4xi32>
}
