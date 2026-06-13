// RUN: circt-opt --check-coralnpu-hw %s 2>&1 | FileCheck %s

// CHECK: === Coral NPU Hardware Verification ===
// CHECK: [1/3] AXI4 Protocol Compliance
// CHECK: [2/3] Pipeline Structure Analysis
// CHECK: [3/3] Clock/Reset Domain Check
// CHECK: === Verification Summary ===
// CHECK: Status: PASSED

func.func @axi_dma_test(%awvalid: i32, %awready: i32) -> i32 {
  coralnpu.vsetvl e32, m1
  %addr = coralnpu.li 0x1000 : i32
  %size = coralnpu.li 256 : i32
  coralnpu.dma_load %addr, %addr, %size
  %data = coralnpu.vle32 %addr, %size : i32
  %result = coralnpu.vadd %data, %data : i32
  coralnpu.vse32 %result, %addr, %size : i32
  coralnpu.dma_store %addr, %addr, %size
  coralnpu.return %result : i32
}
