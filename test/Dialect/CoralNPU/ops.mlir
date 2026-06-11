// RUN: circt-opt %s | circt-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// CoralNPU Dialect - Round-trip test
// Verifies that all CoralNPU ops parse and print correctly.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_scalar_ops
func.func @test_scalar_ops() {
  // CHECK: coralnpu.li 42 : i32
  %0 = coralnpu.li 42 : i32
  // CHECK: coralnpu.add %0, %0 : i32
  %1 = coralnpu.add %0, %0 : i32
  // CHECK: coralnpu.sub %1, %0 : i32
  %2 = coralnpu.sub %1, %0 : i32
  // CHECK: coralnpu.mul %2, %2 : i32
  %3 = coralnpu.mul %2, %2 : i32
  // CHECK: coralnpu.return %3 : i32
  coralnpu.return %3 : i32
}

// CHECK-LABEL: func.func @test_vector_ops
func.func @test_vector_ops(%addr: i32) {
  // CHECK: coralnpu.vsetvl e32, m1
  coralnpu.vsetvl e32, m1
  // CHECK: coralnpu.vle32 %addr, 8 : i32
  %v = coralnpu.vle32 %addr, 8 : i32
  // CHECK: coralnpu.vadd %v, %v : i32
  %r = coralnpu.vadd %v, %v : i32
  // CHECK: coralnpu.vse32 %r, %addr, 8 : i32
  coralnpu.vse32 %r, %addr, 8 : i32
  coralnpu.return
}

// CHECK-LABEL: func.func @test_stripmine
func.func @test_stripmine(%addr: i32) {
  coralnpu.vsetvl e8, m2
  %v = coralnpu.vle8 %addr, 16 : i32
  // CHECK: coralnpu.vadd stripmine = 4 %v, %v : i32
  %r = coralnpu.vadd stripmine = 4 %v, %v : i32
  // CHECK: coralnpu.vdot stripmine = 2 %r, %r : i32
  %d = coralnpu.vdot stripmine = 2 %r, %r : i32
  coralnpu.vse8 %d, %addr, 16 : i32
  coralnpu.return
}

// CHECK-LABEL: func.func @test_matrix_ops
func.func @test_matrix_ops(%inp: i32, %wgt: i32, %acc: i32) {
  coralnpu.vsetvl e8, m1
  // CHECK: coralnpu.outer_product %arg0, %arg1, %arg2 : i32
  %new_acc = coralnpu.outer_product %inp, %wgt, %acc : i32
  // CHECK: coralnpu.outer_product %arg0, %arg1, %new_acc stripmine = 4 : i32
  %new_acc2 = coralnpu.outer_product %inp, %wgt, %new_acc stripmine = 4 : i32
  coralnpu.return %new_acc2 : i32
}

// CHECK-LABEL: func.func @test_memory_ops
func.func @test_memory_ops(%addr: i32) {
  // CHECK: coralnpu.lw %arg0 : i32
  %v = coralnpu.lw %addr : i32
  // CHECK: coralnpu.add %v, %v : i32
  %r = coralnpu.add %v, %v : i32
  // CHECK: coralnpu.sw %r, %addr : i32
  coralnpu.sw %r, %addr : i32
  coralnpu.return
}

// CHECK-LABEL: func.func @test_control_flow
func.func @test_control_flow(%a: i32, %b: i32) {
  // CHECK: coralnpu.cond_br eq = true %arg0, %arg1, ^[[BB1:.*]], ^[[BB2:.*]]
  coralnpu.cond_br eq = true %a, %b, ^bb1, ^bb2

^bb1:  // CHECK: ^[[BB1]]:
  %r1 = coralnpu.add %a, %b : i32
  coralnpu.return %r1 : i32

^bb2:  // CHECK: ^[[BB2]]:
  %r2 = coralnpu.sub %a, %b : i32
  coralnpu.return %r2 : i32
}
