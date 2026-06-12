// RUN: circt-opt --coralnpu-stripmine %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Stripmine scheduling tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_stripmine_vadd
func.func @test_stripmine_vadd(%a: i32, %b: i32) -> i32 {
  coralnpu.vsetvl e32, m4
  // CHECK: coralnpu.vadd stripmine = 4 {{.*}} : i32
  %r = coralnpu.vadd %a, %b : i32
  coralnpu.return %r : i32
}

// CHECK-LABEL: func.func @test_stripmine_vsub
func.func @test_stripmine_vsub(%a: i32, %b: i32) -> i32 {
  coralnpu.vsetvl e8, m2
  // CHECK: coralnpu.vsub stripmine = 4 {{.*}} : i32
  %r = coralnpu.vsub %a, %b : i32
  coralnpu.return %r : i32
}

// CHECK-LABEL: func.func @test_stripmine_vdot
func.func @test_stripmine_vdot(%a: i32, %b: i32) -> i32 {
  coralnpu.vsetvl e8, m1
  // CHECK: coralnpu.vdot stripmine = 4 {{.*}} : i32
  %r = coralnpu.vdot %a, %b : i32
  coralnpu.return %r : i32
}

// CHECK-LABEL: func.func @test_stripmine_outer_product
func.func @test_stripmine_outer_product(%inp: i32, %wgt: i32, %acc: i32) -> i32 {
  coralnpu.vsetvl e8, m1
  // CHECK: coralnpu.outer_product {{.*}} stripmine = 4 : i32
  %r = coralnpu.outer_product %inp, %wgt, %acc : i32
  coralnpu.return %r : i32
}

// CHECK-LABEL: func.func @test_stripmine_already_set
func.func @test_stripmine_already_set(%a: i32, %b: i32) -> i32 {
  coralnpu.vsetvl e32, m1
  // CHECK: coralnpu.vadd stripmine = 2 {{.*}} : i32
  %r = coralnpu.vadd stripmine = 2 %a, %b : i32
  coralnpu.return %r : i32
}
