// RUN: circt-opt --tosa-to-coralnpu --coralnpu-stripmine --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s

// Test: TOSA add lowering → real RISC-V emit
// The output should be real RISC-V assembly runnable on spike+CoralNPU

// CHECK: # CoralNPU Assembly (target: real google-coral/coralnpu ISA)
// CHECK: ISA: rv32imf_zve32f_zvl128b_zicsr_zifencei_zbb_zfbfmin_zvfbfa
// CHECK: _start:
// CHECK: mul
// CHECK: vadd.vv
// CHECK: ret

func.func @e2e_add(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  func.return %0 : tensor<4x4xi32>
}
