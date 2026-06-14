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
  %1 = coralnpu.li 7 : i32
  // CHECK: coralnpu.add %0, %0 : i32
  %2 = coralnpu.add %0, %0 : i32
  // CHECK: coralnpu.sub %2, %0 : i32
  %3 = coralnpu.sub %2, %0 : i32
  // CHECK: coralnpu.mul %3, %3 : i32
  %4 = coralnpu.mul %3, %3 : i32
  // CHECK: coralnpu.div %4, %0 : i32
  %5 = coralnpu.div %4, %0 : i32
  // CHECK: coralnpu.and %5, %0 : i32
  %6 = coralnpu.and %5, %0 : i32
  // CHECK: coralnpu.or %6, %0 : i32
  %7 = coralnpu.or %6, %0 : i32
  // CHECK: coralnpu.xor %7, %0 : i32
  %8 = coralnpu.xor %7, %0 : i32
  // CHECK: coralnpu.sll %8, %1 : i32
  %9 = coralnpu.sll %8, %1 : i32
  // CHECK: coralnpu.srl %9, %1 : i32
  %10 = coralnpu.srl %9, %1 : i32
  // CHECK: coralnpu.sra %10, %1 : i32
  %11 = coralnpu.sra %10, %1 : i32
  // CHECK: coralnpu.slt %11, %0 : i32
  %12 = coralnpu.slt %11, %0 : i32
  // CHECK: coralnpu.sltu %11, %0 : i32
  %13 = coralnpu.sltu %11, %0 : i32
  // CHECK: coralnpu.return %13 : i32
  coralnpu.return %13 : i32
}

// CHECK-LABEL: func.func @test_vector_ops
func.func @test_vector_ops(%arg0: i32, %arg1: i32) {
  // CHECK: coralnpu.vsetvl e32, m1
  coralnpu.vsetvl e32, m1
  // CHECK: coralnpu.vle32 %arg0, %arg1 : i32
  %0 = coralnpu.vle32 %arg0, %arg1 : i32
  // CHECK: coralnpu.vadd %0, %0 : i32
  %1 = coralnpu.vadd %0, %0 : i32
  // CHECK: coralnpu.vse32 %1, %arg0, %arg1 : i32
  coralnpu.vse32 %1, %arg0, %arg1 : i32
  // CHECK: coralnpu.vsetvl e16, m2
  coralnpu.vsetvl e16, m2
  // CHECK: coralnpu.vle16 %arg0, %arg1 : i32
  %2 = coralnpu.vle16 %arg0, %arg1 : i32
  // CHECK: coralnpu.vse16 %2, %arg0, %arg1 : i32
  coralnpu.vse16 %2, %arg0, %arg1 : i32
  // CHECK: coralnpu.vsub %0, %0 : i32
  %3 = coralnpu.vsub %0, %0 : i32
  // CHECK: coralnpu.vmul %3, %3 : i32
  %4 = coralnpu.vmul %3, %3 : i32
  coralnpu.return
}

// CHECK-LABEL: func.func @test_stripmine
func.func @test_stripmine(%arg0: i32, %arg1: i32) {
  coralnpu.vsetvl e8, m2
  // CHECK: coralnpu.vle8 %arg0, %arg1 : i32
  %0 = coralnpu.vle8 %arg0, %arg1 : i32
  // CHECK: coralnpu.vadd stripmine = 4 %0, %0 : i32
  %1 = coralnpu.vadd stripmine = 4 %0, %0 : i32
  // CHECK: coralnpu.vdot stripmine = 2 %1, %1 : i32
  %2 = coralnpu.vdot stripmine = 2 %1, %1 : i32
  // CHECK: coralnpu.vwadd stripmine = 2 %2, %2 : i32
  %3 = coralnpu.vwadd stripmine = 2 %2, %2 : i32
  coralnpu.vse8 %3, %arg0, %arg1 : i32
  coralnpu.return
}

// CHECK-LABEL: func.func @test_reduction
func.func @test_reduction(%arg0: i32) {
  // CHECK: coralnpu.vredsum %arg0 : i32
  %0 = coralnpu.vredsum %arg0 : i32
  coralnpu.return %0 : i32
}

// CHECK-LABEL: func.func @test_matrix_ops
func.func @test_matrix_ops(%arg0: i32, %arg1: i32, %arg2: i32) {
  coralnpu.vsetvl e8, m1
  // CHECK: coralnpu.outer_product %arg0, %arg1, %arg2 : i32
  %0 = coralnpu.outer_product %arg0, %arg1, %arg2 : i32
  // CHECK: coralnpu.outer_product %arg0, %arg1, %0 stripmine = 4 : i32
  %1 = coralnpu.outer_product %arg0, %arg1, %0 stripmine = 4 : i32
  // CHECK: coralnpu.aconv %1, %arg1, %arg0 : i32
  %2 = coralnpu.aconv %1, %arg1, %arg0 : i32
  // CHECK: coralnpu.accread %2, %arg0 : i32
  %3 = coralnpu.accread %2, %arg0 : i32
  coralnpu.return %3 : i32
}

// CHECK-LABEL: func.func @test_memory_ops
func.func @test_memory_ops(%arg0: i32) {
  // CHECK: coralnpu.lw %arg0 : i32
  %0 = coralnpu.lw %arg0 : i32
  // CHECK: coralnpu.add %0, %0 : i32
  %1 = coralnpu.add %0, %0 : i32
  // CHECK: coralnpu.sw %1, %arg0 : i32
  coralnpu.sw %1, %arg0 : i32
  coralnpu.return
}

// CHECK-LABEL: func.func @test_dma_ops
func.func @test_dma_ops(%arg0: i32, %arg1: i32, %arg2: i32) {
  // CHECK: coralnpu.dma_load %arg0, %arg1, %arg2
  coralnpu.dma_load %arg0, %arg1, %arg2
  // CHECK: coralnpu.sw %arg0, %arg1 : i32
  coralnpu.sw %arg0, %arg1 : i32
  // CHECK: coralnpu.dma_store %arg0, %arg1, %arg2
  coralnpu.dma_store %arg0, %arg1, %arg2
  coralnpu.return
}

// CHECK-LABEL: func.func @test_control_flow
func.func @test_control_flow(%arg0: i32, %arg1: i32) {
  // CHECK: coralnpu.cond_br eq = true %arg0, %arg1, ^[[BB1:.*]], ^[[BB2:.*]]
  coralnpu.cond_br eq = true %arg0, %arg1, ^bb1, ^bb2

^bb1:  // CHECK: ^[[BB1]]:
  %0 = coralnpu.add %arg0, %arg1 : i32
  coralnpu.return %0 : i32

^bb2:  // CHECK: ^[[BB2]]:
  %1 = coralnpu.sub %arg0, %arg1 : i32
  coralnpu.return %1 : i32
}

// CHECK-LABEL: func.func @test_branches
func.func @test_branches(%arg0: i32, %arg1: i32) {
  // CHECK: coralnpu.beq %arg0, %arg1, ^[[EQ:.*]], ^[[NEQ:.*]]
  coralnpu.beq %arg0, %arg1, ^eq, ^neq

^eq:
  // CHECK: coralnpu.jal @func : i32
  %0 = coralnpu.jal @func : i32
  coralnpu.br ^end

^neq:
  // CHECK: coralnpu.bne %arg0, %arg1, ^[[NE2:.*]], ^[[EQ2:.*]]
  coralnpu.bne %arg0, %arg1, ^neq2, ^eq2

^neq2:
  // CHECK: coralnpu.jal @func : i32
  %1 = coralnpu.jal @func : i32
  coralnpu.br ^end

^eq2:
  coralnpu.br ^end

^func:
  coralnpu.return

^end:
  coralnpu.return
}
