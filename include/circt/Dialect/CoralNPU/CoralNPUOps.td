//===- CoralNPUOps.td - CoralNPU dialect ops ------------------*- tablegen -*-===//
//
// Operation definitions for the Coral NPU dialect.
// Included from CoralNPU.td — do not include separately.
//
//===----------------------------------------------------------------------===//

#ifndef CORALNPU_OPS_TD
#define CORALNPU_OPS_TD

// Note: base TableGen includes (OpBase, SideEffectInterfaces, EnumAttr, etc.)
// are provided by the parent CoralNPU.td which includes this file.

//===----------------------------------------------------------------------===//
// Scalar operations
//===----------------------------------------------------------------------===//

// --- Scalar arithmetic ---

def ScalarAddOp : CoralNPU_Op<"add", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer addition";
  let arguments = (ins AnyTypeOf<[I32, UI32]>:$lhs, AnyTypeOf<[I32, UI32]>:$rhs);
  let results = (outs AnyTypeOf<[I32, UI32]>:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ScalarSubOp : CoralNPU_Op<"sub", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer subtraction";
  let arguments = (ins AnyTypeOf<[I32, UI32]>:$lhs, AnyTypeOf<[I32, UI32]>:$rhs);
  let results = (outs AnyTypeOf<[I32, UI32]>:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ScalarMulOp : CoralNPU_Op<"mul", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer multiplication (MLU)";
  let arguments = (ins AnyTypeOf<[I32, UI32]>:$lhs, AnyTypeOf<[I32, UI32]>:$rhs);
  let results = (outs AnyTypeOf<[I32, UI32]>:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// --- Scalar constant ---

def ScalarLiOp : CoralNPU_Op<"li", [Pure]> {
  let summary = "Load immediate 32-bit value into scalar register";
  let arguments = (ins I32Attr:$value);
  let results = (outs I32:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Vector configuration
//===----------------------------------------------------------------------===//

def VSetVLOp : CoralNPU_Op<"vsetvl", [Pure]> {
  let summary = "Configure vector length and type (SEW, LMUL)";
  let description = [{
    Sets the vector configuration for subsequent SIMD operations.
    Corresponds to the RISC-V vsetvl instruction.
  }];
  let arguments = (ins CoralNPU_SEW:$sew, CoralNPU_LMUL:$lmul);
  let results = (outs);
  let assemblyFormat = "$sew `,` $lmul attr-dict";
}

//===----------------------------------------------------------------------===//
// Vector memory operations
//===----------------------------------------------------------------------===//

def VLE8Op : CoralNPU_Op<"vle8", [Pure]> {
  let summary = "Vector load of 8-bit elements from memory";
  let arguments = (ins I32:$addr, I32:$n);
  let results = (outs AnyTypeOf<[I32, I64]>:$result); // simplified: result carries packed i8 data in i32/i64
  let assemblyFormat = "$addr `,` $n attr-dict `:` type($result)";
}

def VLE32Op : CoralNPU_Op<"vle32", [Pure]> {
  let summary = "Vector load of 32-bit elements from memory";
  let arguments = (ins I32:$addr, I32:$n);
  let results = (outs I32:$result);
  let assemblyFormat = "$addr `,` $n attr-dict `:` type($result)";
}

def VSE8Op : CoralNPU_Op<"vse8", [Pure]> {
  let summary = "Vector store of 8-bit elements to memory";
  let arguments = (ins AnyTypeOf<[I32, I64]>:$value, I32:$addr, I32:$n);
  let results = (outs);
  let assemblyFormat = "$value `,` $addr `,` $n attr-dict `:` type($value)";
}

def VSE32Op : CoralNPU_Op<"vse32", [Pure]> {
  let summary = "Vector store of 32-bit elements to memory";
  let arguments = (ins I32:$value, I32:$addr, I32:$n);
  let results = (outs);
  let assemblyFormat = "$value `,` $addr `,` $n attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// Vector arithmetic operations (with stripmine support)
//===----------------------------------------------------------------------===//

def VAddOp : CoralNPU_Op<"vadd", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Vector element-wise addition";
  let description = [{
    SIMD vector addition. The `stripmine` attribute controls how many times
    the hardware issues this instruction (1/2/4), allowing a single dispatch
    to produce multiple serialized SIMD operations.
  }];
  let arguments = (ins I32:$lhs, I32:$rhs, DefaultValuedAttr<ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<4>]>, "1">:$stripmine);
  let results = (outs I32:$result);
  let assemblyFormat = "(`stripmine` `=` $stripmine^)? $lhs `,` $rhs attr-dict `:` type($result)";
}

// TODO: VWAddOp has ODS constructor generation issue; will fix in next iteration.
// def VWAddOp : CoralNPU_Op<"vwadd", ...> { ... }

def VDotOp : CoralNPU_Op<"vdot", [Pure]> {
  let summary = "Vector dot product (4×int8→int32)";
  let description = [{
    Performs 4-way 8-bit dot product reduction to 32-bit.
    Used as the building block for outer-product MAC:
      result[i] += lhs[i*4+0]*rhs[i*4+0] + ... + lhs[i*4+3]*rhs[i*4+3]
  }];
  let arguments = (ins I32:$lhs, I32:$rhs, DefaultValuedAttr<ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<4>]>, "1">:$stripmine);
  let results = (outs I32:$result);
  let assemblyFormat = "(`stripmine` `=` $stripmine^)? $lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Outer-product MAC operation
//===----------------------------------------------------------------------===//

def OuterProductOp : CoralNPU_Op<"outer_product", [Pure]> {
  let summary = "Outer-product multiply-accumulate for matrix engine";
  let description = [{
    Performs the outer-product MAC:
      acc[row][col] += input[row] * weight[col]

    This is the core operation of the 8×8 accumulator engine.
    The `stripmine` attribute controls hardware replication (each lane
    processes different weight columns in parallel).
  }];
  let arguments = (ins I32:$input,        // 8-bit elements packed in i32
                       I32:$weight,       // 8-bit elements packed in i32
                       I32:$acc_old,      // previous accumulator value
                       DefaultValuedAttr<ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<4>]>, "1">:$stripmine);
  let results = (outs I32:$acc_new);      // updated accumulator value
  let assemblyFormat = "$input `,` $weight `,` $acc_old (`stripmine` `=` $stripmine^)? attr-dict `:` type($acc_new)";
}

//===----------------------------------------------------------------------===//
// Scalar memory operations
//===----------------------------------------------------------------------===//

def ScalarLwOp : CoralNPU_Op<"lw", [Pure]> {
  let summary = "Load 32-bit word from TCM";
  let arguments = (ins I32:$addr);
  let results = (outs I32:$result);
  let assemblyFormat = "$addr attr-dict `:` type($result)";
}

def ScalarSwOp : CoralNPU_Op<"sw", [Pure]> {
  let summary = "Store 32-bit word to TCM";
  let arguments = (ins I32:$value, I32:$addr);
  let results = (outs);
  let assemblyFormat = "$value `,` $addr attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

def BranchOp : CoralNPU_Op<"br", [Terminator]> {
  let summary = "Unconditional branch";
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = "$dest attr-dict";
}

def CondBranchOp : CoralNPU_Op<"cond_br", [Terminator]> {
  let summary = "Conditional branch (beq/bne)";
  let arguments = (ins I32:$lhs, I32:$rhs, BoolAttr:$eq);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = "`eq` `=` $eq $lhs `,` $rhs `,` $trueDest `,` $falseDest attr-dict";
}

def ReturnOp : CoralNPU_Op<"return", [Terminator]> {
  let summary = "Function return";
  let arguments = (ins Optional<I32>:$value);
  let assemblyFormat = "($value^)? attr-dict `:` type($value)";
}

#endif // CORALNPU_OPS_TD
