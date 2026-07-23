// RUN: circt-opt --linalg-to-coralnpu %s 2>/dev/null | FileCheck %s

// Tests linalg.batch_matmul → CoralNPU lowering.
// Input comes from the TOSA→linalg→bufferize pipeline.
// A: memref<1x1x4xi32> (arg0 → TCM slot 0 = 0x10000)
// B: memref<1x4x4xi32> (arg1 → TCM slot 1 = 0x11000, interpreted as [N,K]=[4,4])
// C: memref.alloc (slot 32 = 0x30000)

// CHECK-LABEL: func.func @gemv_linalg
// CHECK: coralnpu.vsetvl
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vle32
// CHECK: coralnpu.vmul
// CHECK: coralnpu.vredsum
// CHECK: coralnpu.sw
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: linalg.fill

func.func @gemv_linalg(%arg0: memref<1x1x4xi32, strided<[?, ?, ?], offset: ?>>,
                        %arg1: memref<1x4x4xi32, strided<[?, ?, ?], offset: ?>>) -> memref<1x1x4xi32> {
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x4xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<1x1x4xi32>)
  linalg.batch_matmul ins(%arg0, %arg1 : memref<1x1x4xi32, strided<[?, ?, ?], offset: ?>>,
                                         memref<1x4x4xi32, strided<[?, ?, ?], offset: ?>>) 
                      outs(%alloc : memref<1x1x4xi32>)
  return %alloc : memref<1x1x4xi32>
}
