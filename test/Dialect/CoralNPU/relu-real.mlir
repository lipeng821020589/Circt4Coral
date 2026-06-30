// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// tosa.clamp (ReLU) → VMaxVXOp → vmax.vx v, v, x0
//
// ReLU = max(x, 0) per element. The lowering uses VMaxVXOp which emits
// vmax.vx v_out, v_in, x0, exploiting RISC-V's x0 zero register as the
// scalar operand for a free "max with zero" per element.
//===----------------------------------------------------------------------===//

// tensor<4xi32>, k=1: single vle32 → vmax.vx → vse32.
// CHECK-LABEL: @relu_k1
// CHECK: %[[T0:.*]] = coralnpu.vle32
// CHECK: %[[R0:.*]] = coralnpu.vmax_vx %[[T0]]
// CHECK: coralnpu.vse32 %[[R0]]
// ASM-LABEL: # CoralNPU Assembly
// ASM: vle32.v
// ASM: vmax.vx {{.*}}, x0
// ASM: vse32.v
func.func @relu_k1(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tosa.clamp %arg0 {min_val = 0 : i32, max_val = 2147483647 : i32} : (tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// tensor<8xi32>, k=2 tiles: two vle32 → two vmax.vx → two vse32.
// CHECK-LABEL: @relu_k2
// CHECK: %[[T0:.*]] = coralnpu.vle32
// CHECK: %[[T1:.*]] = coralnpu.vle32
// CHECK: %[[R0:.*]] = coralnpu.vmax_vx %[[T0]]
// CHECK: %[[R1:.*]] = coralnpu.vmax_vx %[[T1]]
// CHECK: coralnpu.vse32 %[[R0]]
// CHECK: coralnpu.vse32 %[[R1]]
// ASM: vmax.vx
// ASM: vmax.vx
func.func @relu_k2(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = tosa.clamp %arg0 {min_val = 0 : i32, max_val = 2147483647 : i32} : (tensor<8xi32>) -> tensor<8xi32>
  func.return %0 : tensor<8xi32>
}
