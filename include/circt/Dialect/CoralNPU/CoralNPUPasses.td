//===- CoralNPUPasses.td - CoralNPU pass definitions ----------*- tablegen -*-===//

#ifndef CORALNPU_PASSES_TD
#define CORALNPU_PASSES_TD

include "mlir/Pass/PassBase.td"

def TosaToCoralNPU : Pass<"tosa-to-coralnpu"> {
  let summary = "Lower TOSA operations to CoralNPU dialect";
  let description = [{
    Converts TOSA ML operations (conv2d, matmul, add, etc.) into CoralNPU
    scalar, vector, and matrix operations suitable for the NPU hardware.
  }];
  let dependentDialects = ["circt::coralnpu::CoralNPUDialect"];
}

def CoralNPUStripmine : Pass<"coralnpu-stripmine"> {
  let summary = "Apply stripmine scheduling to vector operations";
  let description = [{
    Analyzes vector operations and determines optimal stripmine factors
    based on hardware parallelism (128-bit SIMD × 4-way stripmine).
  }];
  let dependentDialects = ["circt::coralnpu::CoralNPUDialect"];
}

def CoralNPULegalize : Pass<"coralnpu-legalize"> {
  let summary = "Legalize CoralNPU operations for instruction selection";
  let description = [{
    Lowers unsupported operations, validates constraints, and prepares
    the IR for register allocation and code emission.
  }];
  let dependentDialects = ["circt::coralnpu::CoralNPUDialect"];
}

def CoralNPURegAlloc : Pass<"coralnpu-regalloc"> {
  let summary = "Register allocation for Coral NPU hardware";
  let description = [{
    Maps virtual SSA values to physical Coral NPU registers:
    x0..x31 scalar, v0..v63 vector, acc[8][8] accumulator matrix.
    Annotates operations with register assignments for assembly emission.
  }];
  let dependentDialects = ["circt::coralnpu::CoralNPUDialect"];
}

def CheckCoralNPUHW : Pass<"check-coralnpu-hw"> {
  let summary = "Coral NPU hardware verification";
  let description = [{
    Verifies Coral NPU microarchitecture constraints:
    - AXI4 protocol compliance (handshake, burst alignment)
    - Pipeline depth analysis
    - Clock/reset domain crossing detection
  }];
  let dependentDialects = ["circt::coralnpu::CoralNPUDialect",
                           "circt::hw::HWDialect"];
}

def CoralNPUFormalVerify : Pass<"coralnpu-formal-verify"> {
  let summary = "Generate formal verification properties for Coral NPU";
  let description = [{
    Generates formal verification properties using CIRCT Verif dialect:
    - MAC outer-product correctness (equiv to scalar reference)
    - AXI4 handshake protocol assertions
    - Register file integrity (x0 zero-check)
    - Stripmine data stability

    Run with --verif-bmc for bounded model checking.
  }];
  let dependentDialects = ["circt::coralnpu::CoralNPUDialect",
                           "circt::verif::VerifDialect"];
}

def LoopToCoralNPU : Pass<"loop-to-coralnpu"> {
  let summary = "HLS: lower affine/scf loops to CoralNPU vector ops";
  let description = [{
    High-Level Synthesis pass that converts sequential loop nests
    into Coral NPU SIMD vector operations. Demonstrates the C-to-NPU
    compilation path:
      affine.for / scf.for → coralnpu.vsetvl + vadd/vsub/vmul
  }];
  let dependentDialects = ["circt::coralnpu::CoralNPUDialect",
                           "mlir::affine::AffineDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def CoralNPUEmitAssembly : Pass<"emit-coralnpu-assembly"> {
  let summary = "Emit Coral NPU assembly from legalized IR";
  let description = [{
    Translates legalized CoralNPU dialect operations into textual
    Coral NPU assembly instructions ready for assembling into ELF.
  }];
  let dependentDialects = ["circt::coralnpu::CoralNPUDialect"];
}

#endif // CORALNPU_PASSES_TD
