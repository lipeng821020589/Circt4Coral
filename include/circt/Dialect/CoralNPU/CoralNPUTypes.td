//===- CoralNPUTypes.td - CoralNPU type definitions ----------*- tablegen -*-===//
//
// Type definitions for the CoralNPU dialect.
// Included from CoralNPU.td — do not include separately.
//
//===----------------------------------------------------------------------===//

#ifndef CORALNPU_TYPES_TD
#define CORALNPU_TYPES_TD

//===----------------------------------------------------------------------===//
// Base type class for CoralNPU types
//===----------------------------------------------------------------------===//

class CoralNPU_Type<string name, list<Trait> traits = []>
    : TypeDef<CoralNPU_Dialect, name, traits> {
}

//===----------------------------------------------------------------------===//
// Scalar register type: !coralnpu.xreg
//===----------------------------------------------------------------------===//

def XRegType : CoralNPU_Type<"XReg"> {
  let mnemonic = "xreg";
  let summary = "Coral NPU 32-bit scalar register (x0..x31)";
  let description = [{
    Represents one of the 32 scalar registers in the Coral NPU.
    x0 is hardwired to zero. This type has no parameters; it always
    represents a 32-bit scalar register value.
  }];
}

//===----------------------------------------------------------------------===//
// Vector register type: !coralnpu.vreg<sew, lmul>
//===----------------------------------------------------------------------===//

def VRegType : CoralNPU_Type<"VReg"> {
  let mnemonic = "vreg";
  let summary = "Coral NPU vector register group (v0..v63)";
  let description = [{
    Represents a group of vector registers (v0..v63) configured with the
    given SEW (Selected Element Width) and LMUL (Vector Length Multiplier).

    SEW determines the element size: e8 (8-bit), e16 (16-bit), e32 (32-bit).
    LMUL determines register grouping: mf4..m8.

    Example: `!coralnpu.vreg<e8, m1>` — 1× vector register group, 8-bit elements.

    A single physical vector register is 128-bit wide.
    With SEW=32 and VLEN=128, each register holds 4×32-bit elements.
  }];
  let parameters = (ins
    "::circt::coralnpu::SEW":$sew,
    "::circt::coralnpu::LMUL":$lmul
  );
  let assemblyFormat = "`<` $sew `,` $lmul `>`";
}

//===----------------------------------------------------------------------===//
// Accumulator matrix type: !coralnpu.acc<rows, cols>
//===----------------------------------------------------------------------===//

def AccType : CoralNPU_Type<"Acc"> {
  let mnemonic = "acc";
  let summary = "Coral NPU 8×8 accumulator matrix";
  let description = [{
    Represents the 8×8 accumulator register file. Each cell holds a 32-bit
    accumulated value. The outer-product MAC engine writes into this matrix:
      acc[row][col] += input[row] * weight[col]

    Default: 8 rows × 8 columns × 32-bit = 256 bytes total.

    Example: `!coralnpu.acc<8, 8>` — standard 8×8 accumulator.
  }];
  let parameters = (ins
    "unsigned":$rows,
    "unsigned":$cols
  );
  let assemblyFormat = "`<` $rows `,` $cols `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Vector type configuration: !coralnpu.vtype
//===----------------------------------------------------------------------===//

def VTypeConfig : CoralNPU_Type<"VTypeConfig"> {
  let mnemonic = "vtype";
  let summary = "Coral NPU vector type configuration token";
  let description = [{
    Represents the vtype configuration state. This is a token type
    used by `vsetvl` to propagate vector configuration through the IR.
    It carries no data; it serves as a compile-time configuration marker.

    Example: `!coralnpu.vtype`
  }];
}

#endif // CORALNPU_TYPES_TD
