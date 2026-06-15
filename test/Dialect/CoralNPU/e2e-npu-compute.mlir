// RUN: circt-opt --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck %s

//===----------------------------------------------------------------------===//
// End-to-End NPU Compute Pipeline Test
//
// Models: out[i] = (in1[i] * in2[i]) + in3[i]  (vector element-wise)
//
// Pipeline: parse → regalloc → emit assembly
//===----------------------------------------------------------------------===//

// CHECK-LABEL: # CoralNPU Assembly (target: real google-coral/coralnpu ISA)
// CHECK: _start:
// CHECK: .L0:
// CHECK: addi
// CHECK: vsetivli
// CHECK: vle32.v
// CHECK: vmul.vv
// CHECK: vadd.vv
// CHECK: vse32.v
// CHECK: ret
// CHECK: .Lexit:

module {
  func.func @npu_compute() -> i32 {
    // --- Setup ---
    %addr = coralnpu.li 65536 : i32         // 0x10000 base address
    %n    = coralnpu.li 8 : i32             // 8 elements per vector

    // Configure vector unit: 32-bit elements, LMUL=1
    coralnpu.vsetvl e32, m1

    // --- Load inputs ---
    %in1 = coralnpu.vle32 %addr, %n : i32

    %off1  = coralnpu.li 32 : i32
    %addr2 = coralnpu.add %addr, %off1 : i32
    %in2   = coralnpu.vle32 %addr2, %n : i32

    %off2  = coralnpu.li 64 : i32
    %addr3 = coralnpu.add %addr, %off2 : i32
    %in3   = coralnpu.vle32 %addr3, %n : i32

    // --- Compute: out = (in1 * in2) + in3 ---
    %mul_result = coralnpu.vmul %in1, %in2 : i32
    %add_result = coralnpu.vadd %mul_result, %in3 : i32

    // --- Store result ---
    %off3     = coralnpu.li 96 : i32
    %addr_out = coralnpu.add %addr, %off3 : i32
    coralnpu.vse32 %add_result, %addr_out, %n : i32

    // --- Return success ---
    %zero = coralnpu.li 0 : i32
    coralnpu.return %zero : i32
  }
}
