//===- CoralNPU.td - CoralNPU dialect definition -------------*- tablegen -*-===//
//
// Coral NPU MLIR Dialect - scalar, vector, and matrix operations for Google's
// Coral NPU hardware accelerator.
//
//===----------------------------------------------------------------------===//

#ifndef CORALNPU_TD
#define CORALNPU_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// CoralNPU Dialect
//===----------------------------------------------------------------------===//

def CoralNPU_Dialect : Dialect {
  let name = "coralnpu";
  let summary = "Coral NPU scalar, vector, and matrix operations";
  let description = [{
    The `coralnpu` dialect models the computation capabilities of Google's Coral
    NPU hardware accelerator. It provides:

    - **Scalar Ops**: RISC-V rv32im integer arithmetic and control flow
    - **Vector Ops**: 128-bit SIMD operations with configurable SEW/LMUL,
      including stripmine support for folding array parallelism
    - **Matrix Ops**: Outer-product multiply-accumulate engine operations,
      targeting the 8×8 accumulator grid with 256 MAC/cycle throughput

    The dialect serves as a compilation target for ML frameworks (TOSA, Linalg)
    and emits Coral NPU assembly/ELF for execution on real hardware or the
    Verilator simulation environment.
  }];
  let cppNamespace = "::circt::coralnpu";
  let dependentDialects = ["hw::HWDialect"];
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base operation class
//===----------------------------------------------------------------------===//

class CoralNPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<CoralNPU_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

// SEW: Selected Element Width
def CoralNPU_SEW_E8  : I32EnumAttrCase<"E8",  0, "e8">;
def CoralNPU_SEW_E16 : I32EnumAttrCase<"E16", 1, "e16">;
def CoralNPU_SEW_E32 : I32EnumAttrCase<"E32", 2, "e32">;
let cppNamespace = "::circt::coralnpu" in
def CoralNPU_SEW : I32EnumAttr<"SEW", "Selected Element Width", [
  CoralNPU_SEW_E8, CoralNPU_SEW_E16, CoralNPU_SEW_E32
]>;

// LMUL: Vector Register Group Multiplier
def CoralNPU_LMUL_MF4 : I32EnumAttrCase<"MF4", 0, "mf4">;
def CoralNPU_LMUL_MF2 : I32EnumAttrCase<"MF2", 1, "mf2">;
def CoralNPU_LMUL_M1  : I32EnumAttrCase<"M1",  2, "m1">;
def CoralNPU_LMUL_M2  : I32EnumAttrCase<"M2",  3, "m2">;
def CoralNPU_LMUL_M4  : I32EnumAttrCase<"M4",  4, "m4">;
def CoralNPU_LMUL_M8  : I32EnumAttrCase<"M8",  5, "m8">;
let cppNamespace = "::circt::coralnpu" in
def CoralNPU_LMUL : I32EnumAttr<"LMUL", "Vector Length Multiplier", [
  CoralNPU_LMUL_MF4, CoralNPU_LMUL_MF2, CoralNPU_LMUL_M1,
  CoralNPU_LMUL_M2,  CoralNPU_LMUL_M4,  CoralNPU_LMUL_M8
]>;

include "circt/Dialect/CoralNPU/CoralNPUTypes.td"
include "circt/Dialect/CoralNPU/CoralNPUAttributes.td"
include "circt/Dialect/CoralNPU/CoralNPUOps.td"

#endif // CORALNPU_TD
