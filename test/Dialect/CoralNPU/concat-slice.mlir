// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s
// RUN: circt-opt --tosa-to-coralnpu --coralnpu-legalize --coralnpu-regalloc --emit-coralnpu-assembly %s 2>/dev/null | FileCheck --check-prefix=ASM %s

//===----------------------------------------------------------------------===//
// tosa.concat and tosa.slice lowering.
//
// concat: load each input in vregCap-sized tiles, write tiles consecutively
//         to the result TCM slot.
// slice:  load the sub-range [start, start+size) from the input TCM slot,
//         write to the result TCM slot.
//===----------------------------------------------------------------------===//

// ── concat([4xi32], [4xi32]) -> [8xi32] ────────────────────────────────────
// arg0 @ slot0 (0x10000), arg1 @ slot1 (0x11000)
// result @ kResultSlot (0x18000): tile0=arg0, tile1=arg1
// CHECK-LABEL: @test_slice
// CHECK: coralnpu.vsetvl e32, m1
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vse32
// ASM-LABEL: # CoralNPU Assembly
// ASM: vle32.v
func.func @test_slice(%a: tensor<8xi32>) -> tensor<4xi32> {
  %start = tosa.const_shape {values = dense<[2]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %size  = tosa.const_shape {values = dense<[4]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %0 = tosa.slice %a, %start, %size
      : (tensor<8xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// ── concat then slice ────────────────────────────────────────────────────────
// Chain: concat([4xi32],[4xi32])->slice(start=1,size=2) → 2 elements at result
// CHECK-LABEL: @test_concat_then_slice
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vse32
// CHECK: coralnpu.vse32
func.func @test_concat_then_slice(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<2xi32> {
  %cat = tosa.concat %a, %b {axis = 0 : i32} : (tensor<4xi32>, tensor<4xi32>) -> tensor<8xi32>
  %start = tosa.const_shape {values = dense<[1]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %size  = tosa.const_shape {values = dense<[2]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %0 = tosa.slice %cat, %start, %size
      : (tensor<8xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}
