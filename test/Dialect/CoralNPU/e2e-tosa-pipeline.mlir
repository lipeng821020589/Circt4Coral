// RUN: circt-opt --tosa-to-coralnpu --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s

//===----------------------------------------------------------------------===//
// Full ML Compiler Pipeline: TOSA → CoralNPU → RISC-V Assembly
//
// Input:  TOSA element-wise add
// Output: Real RISC-V assembly with V extension
//===----------------------------------------------------------------------===//

// CHECK-LABEL: # CoralNPU Assembly
// CHECK: _start:
// CHECK: addi
// CHECK: vsetivli
// CHECK: vadd.vv
// CHECK: ret

func.func @tosa_add(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> i32 {
  %0 = tosa.add %arg0, %arg1 : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  %ret = coralnpu.li 0 : i32
  func.return %ret : i32
}
