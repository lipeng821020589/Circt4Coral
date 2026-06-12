// RUN: circt-opt %s | circt-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// CoralNPU Dialect - Round-trip test
// Verifies that all CoralNPU ops parse and print correctly.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_scalar_ops
func.func @test_scalar_ops() {
  // CHECK: coralnpu.li 42 : i32
  %0 = coralnpu.li 42 : i32
  // CHECK: coralnpu.li 7 : i32
  %shamt = coralnpu.li 7 : i32
  // CHECK: coralnpu.add %0, %0 : i32
  %1 = coralnpu.add %0, %0 : i32
  // CHECK: coralnpu.sub %1, %0 : i32
  %2 = coralnpu.sub %1, %0 : i32
  // CHECK: coralnpu.mul %2, %2 : i32
  %3 = coralnpu.mul %2, %2 : i32
  // CHECK: coralnpu.div %3, %0 : i32
  %4 = coralnpu.div %3, %0 : i32
  // CHECK: coralnpu.and %4, %0 : i32
  %5 = coralnpu.and %4, %0 : i32
  // CHECK: coralnpu.or %5, %0 : i32
  %6 = coralnpu.or %5, %0 : i32
  // CHECK: coralnpu.xor %6, %0 : i32
  %7 = coralnpu.xor %6, %0 : i32
  // CHECK: coralnpu.sll %7, %shamt : i32
  %8 = coralnpu.sll %7, %shamt : i32
  // CHECK: coralnpu.srl %8, %shamt : i32
  %9 = coralnpu.srl %8, %shamt : i32
  // CHECK: coralnpu.sra %9, %shamt : i32
  %10 = coralnpu.sra %9, %shamt : i32
  // CHECK: coralnpu.slt %10, %0 : i32
  %11 = coralnpu.slt %10, %0 : i32
  // CHECK: coralnpu.sltu %10, %0 : i32
  %12 = coralnpu.sltu %10, %0 : i32
  // CHECK: coralnpu.return %12 : i32
  coralnpu.return %12 : i32
}

// CHECK-LABEL: func.func @test_vector_ops
func.func @test_vector_ops(%addr: i32, %n: i32) {
  // CHECK: coralnpu.vsetvl e32, m1
  coralnpu.vsetvl e32, m1
  // CHECK: coralnpu.vle32 %arg0, %arg1 : i32
  %v32 = coralnpu.vle32 %addr, %n : i32
  // CHECK: coralnpu.vadd %v32, %v32 : i32
  %r = coralnpu.vadd %v32, %v32 : i32
  // CHECK: coralnpu.vse32 %r, %arg0, %arg1 : i32
  coralnpu.vse32 %r, %addr, %n : i32
  // CHECK: coralnpu.vsetvl e16, m2
  coralnpu.vsetvl e16, m2
  // CHECK: coralnpu.vle16 %arg0, %arg1 : i32
  %v16 = coralnpu.vle16 %addr, %n : i32
  // CHECK: coralnpu.vse16 %v16, %arg0, %arg1 : i32
  coralnpu.vse16 %v16, %addr, %n : i32
  // CHECK: coralnpu.vsub %v32, %v32 : i32
  %vs = coralnpu.vsub %v32, %v32 : i32
  // CHECK: coralnpu.vmul %vs, %vs : i32
  %vm = coralnpu.vmul %vs, %vs : i32
  coralnpu.return
}

// CHECK-LABEL: func.func @test_stripmine
func.func @test_stripmine(%addr: i32, %n: i32) {
  coralnpu.vsetvl e8, m2
  // CHECK: coralnpu.vle8 %arg0, %arg1 : i32
  %v = coralnpu.vle8 %addr, %n : i32
  // CHECK: coralnpu.vadd stripmine = 4 %v, %v : i32
  %r = coralnpu.vadd stripmine = 4 %v, %v : i32
  // CHECK: coralnpu.vdot stripmine = 2 %r, %r : i32
  %d = coralnpu.vdot stripmine = 2 %r, %r : i32
  // CHECK: coralnpu.vwadd stripmine = 2 %d, %d : i32
  %w = coralnpu.vwadd stripmine = 2 %d, %d : i32
  coralnpu.vse8 %w, %addr, %n : i32
  coralnpu.return
}

// CHECK-LABEL: func.func @test_reduction
func.func @test_reduction(%v: i32) {
  // CHECK: coralnpu.vredsum %arg0 : i32
  %r = coralnpu.vredsum %v : i32
  coralnpu.return %r : i32
}

// CHECK-LABEL: func.func @test_matrix_ops
func.func @test_matrix_ops(%inp: i32, %wgt: i32, %acc: i32) {
  coralnpu.vsetvl e8, m1
  // CHECK: coralnpu.outer_product %arg0, %arg1, %arg2 : i32
  %new_acc = coralnpu.outer_product %inp, %wgt, %acc : i32
  // CHECK: coralnpu.outer_product %arg0, %arg1, %new_acc stripmine = 4 : i32
  %new_acc2 = coralnpu.outer_product %inp, %wgt, %new_acc stripmine = 4 : i32
  // CHECK: coralnpu.aconv %new_acc2, %arg1, %arg0 : i32
  %aconv = coralnpu.aconv %new_acc2, %wgt, %inp : i32
  // CHECK: coralnpu.accread %aconv, %inp : i32
  %col = coralnpu.accread %aconv, %inp : i32
  coralnpu.return %col : i32
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

// CHECK-LABEL: func.func @test_dma_ops
func.func @test_dma_ops(%src: i32, %dst: i32, %size: i32) {
  // CHECK: coralnpu.dma_load %arg0, %arg1, %arg2
  coralnpu.dma_load %src, %dst, %size
  // CHECK: coralnpu.sw %arg0, %arg1 : i32
  coralnpu.sw %src, %dst : i32
  // CHECK: coralnpu.dma_store %arg0, %arg1, %arg2
  coralnpu.dma_store %src, %dst, %size
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

// CHECK-LABEL: func.func @test_branches
func.func @test_branches(%a: i32, %b: i32) {
  // CHECK: coralnpu.beq %arg0, %arg1, ^[[EQ:.*]], ^[[NEQ:.*]]
  coralnpu.beq %a, %b, ^eq, ^neq

^eq:
  // CHECK: coralnpu.jal @func : i32
  %ra = coralnpu.jal @func : i32
  coralnpu.br ^end

^neq:
  // CHECK: coralnpu.bne %arg0, %arg1, ^[[NE2:.*]], ^[[EQ2:.*]]
  coralnpu.bne %a, %b, ^neq2, ^eq2

^neq2:
  // CHECK: coralnpu.jal @func : i32
  %ra2 = coralnpu.jal @func : i32
  coralnpu.br ^end

^eq2:
  coralnpu.br ^end

^func:
  coralnpu.return

^end:
  coralnpu.return
}
