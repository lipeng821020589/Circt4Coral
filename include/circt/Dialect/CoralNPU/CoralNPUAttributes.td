//===- CoralNPUAttributes.td - CoralNPU attributes -----------*- tablegen -*-===//
//
// Attribute definitions for the CoralNPU dialect.
// Included from CoralNPU.td — do not include separately.
//
//===----------------------------------------------------------------------===//

#ifndef CORALNPU_ATTRIBUTES_TD
#define CORALNPU_ATTRIBUTES_TD

//===----------------------------------------------------------------------===//
// Stripmine attribute
//===----------------------------------------------------------------------===//

def StripmineAttr : AttrDef<CoralNPU_Dialect, "Stripmine"> {
  let mnemonic = "stripmine";
  let summary = "Stripmine factor for SIMD hardware replication";
  let description = [{
    Controls how many times the hardware issues a single vector instruction.
    Valid values: 1 (default, no replication), 2, or 4.

    Example:
      %r = coralnpu.vadd stripmine=4 %a, %b : !coralnpu.vreg<e32, m1>

    This issues the vadd 4 times in hardware, processing 4× the data
    per dispatch.
  }];
  let parameters = (ins "unsigned":$factor);
  let assemblyFormat = "`<` $factor `>`";
}

#endif // CORALNPU_ATTRIBUTES_TD
