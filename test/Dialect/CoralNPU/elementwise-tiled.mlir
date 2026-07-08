// RUN: circt-opt --tosa-to-coralnpu %s 2>/dev/null | FileCheck %s

// This test proves that the tosa-to-coralnpu pass tiles a 16-element i32
// tosa.add into 4 register-sized tiles and uses progressive byte addresses
// for each tile (tile i starts at base + i*16 bytes).

// CHECK: coralnpu.li 65536 : i32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.li 65552 : i32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.li 65568 : i32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.li 65584 : i32
// CHECK: coralnpu.vle32

// CHECK-COUNT-4: coralnpu.vadd %{{.*}}, %{{.*}} : (!coralnpu.vreg<e32, m1>, !coralnpu.vreg<e32, m1>) -> !coralnpu.vreg<e32, m1>

// CHECK: coralnpu.li 98304 : i32
// CHECK: coralnpu.vse32
// CHECK: coralnpu.li 98320 : i32
// CHECK: coralnpu.vse32
// CHECK: coralnpu.li 98336 : i32
// CHECK: coralnpu.vse32
// CHECK: coralnpu.li 98352 : i32
// CHECK: coralnpu.vse32

func.func @tiled_add(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> tensor<16xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
  func.return %0 : tensor<16xi32>
}
