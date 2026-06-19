// RUN: circt-opt --tosa-to-coralnpu --coralnpu-stripmine --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s

// Test: TOSA add lowering → real RISC-V emit.
// The lowering now carries real data flow: each tensor operand is loaded
// from its TCM slot (vle32), added (vadd.vv), and the result stored back
// (vse32). The output is real RISC-V assembly runnable on spike+CoralNPU.

// CHECK: # CoralNPU Assembly (target: real google-coral/coralnpu ISA)
// CHECK: ISA: rv32imf_zve32f_zvl128b_zicsr_zifencei_zbb_zfbfmin_zvfbfa
// CHECK: _start:
// CHECK: vle32.v
// CHECK: vle32.v
// CHECK: vadd.vv
// CHECK: vse32.v
// CHECK: ret

func.func @e2e_add(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  func.return %0 : tensor<4x4xi32>
}
