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

// --- Additional scalar arithmetic ---

def ScalarDivOp : CoralNPU_Op<"div", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer division (DVU)";
  let arguments = (ins AnyTypeOf<[I32, UI32]>:$lhs, AnyTypeOf<[I32, UI32]>:$rhs);
  let results = (outs AnyTypeOf<[I32, UI32]>:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// --- Scalar bitwise ---

def ScalarAndOp : CoralNPU_Op<"and", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer bitwise AND";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ScalarOrOp : CoralNPU_Op<"or", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer bitwise OR";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ScalarXorOp : CoralNPU_Op<"xor", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer bitwise XOR";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// --- Scalar shifts ---

def ScalarSllOp : CoralNPU_Op<"sll", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer shift left logical";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ScalarSrlOp : CoralNPU_Op<"srl", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer shift right logical";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ScalarSraOp : CoralNPU_Op<"sra", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer shift right arithmetic";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// --- Scalar comparison ---

def ScalarSltOp : CoralNPU_Op<"slt", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer set less than (signed)";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ScalarSltuOp : CoralNPU_Op<"sltu", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Scalar 32-bit integer set less than (unsigned)";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
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

def VLE16Op : CoralNPU_Op<"vle16", [Pure]> {
  let summary = "Vector load of 16-bit elements from memory";
  let arguments = (ins I32:$addr, I32:$n);
  let results = (outs I32:$result);
  let assemblyFormat = "$addr `,` $n attr-dict `:` type($result)";
}

def VSE16Op : CoralNPU_Op<"vse16", [Pure]> {
  let summary = "Vector store of 16-bit elements to memory";
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

def VSubOp : CoralNPU_Op<"vsub", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Vector element-wise subtraction";
  let arguments = (ins I32:$lhs, I32:$rhs, DefaultValuedAttr<ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<4>]>, "1">:$stripmine);
  let results = (outs I32:$result);
  let assemblyFormat = "(`stripmine` `=` $stripmine^)? $lhs `,` $rhs attr-dict `:` type($result)";
}

def VMulOp : CoralNPU_Op<"vmul", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Vector element-wise multiplication";
  let arguments = (ins I32:$lhs, I32:$rhs, DefaultValuedAttr<ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<4>]>, "1">:$stripmine);
  let results = (outs I32:$result);
  let assemblyFormat = "(`stripmine` `=` $stripmine^)? $lhs `,` $rhs attr-dict `:` type($result)";
}

def VWAddOp : CoralNPU_Op<"vwadd", [Pure]> {
  let summary = "Vector widening addition (e8+e8→e16, e16+e16→e32)";
  let description = [{
    Widening vector addition. Takes narrower elements and produces wider results.
    e.g., two e8 vectors produce an e16 result; two e16 vectors produce e32.
  }];
  let arguments = (ins I32:$lhs, I32:$rhs, DefaultValuedAttr<ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<4>]>, "1">:$stripmine);
  let results = (outs I32:$result);
  let assemblyFormat = "(`stripmine` `=` $stripmine^)? $lhs `,` $rhs attr-dict `:` type($result)";
}

def VRedSumOp : CoralNPU_Op<"vredsum", [Pure]> {
  let summary = "Vector reduction sum (vector→scalar)";
  let description = [{
    Reduces all elements of a vector register to a single scalar sum.
    Used as the building block for pooling operations.
  }];
  let arguments = (ins I32:$vector);
  let results = (outs I32:$result);
  let assemblyFormat = "$vector attr-dict `:` type($result)";
}

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
// Additional matrix operations
//===----------------------------------------------------------------------===//

def AConvOp : CoralNPU_Op<"aconv", [Pure]> {
  let summary = "Accumulator convolution";
  let description = [{
    Performs a spatial convolution over the 8×8 accumulator matrix using
    a smaller filter kernel. Used as a building block for depthwise conv.
  }];
  let arguments = (ins I32:$acc, I32:$kernel, I32:$bias);
  let results = (outs I32:$result);
  let assemblyFormat = "$acc `,` $kernel `,` $bias attr-dict `:` type($result)";
}

def AccReadOp : CoralNPU_Op<"accread", [Pure]> {
  let summary = "Read accumulator column back to scalar register";
  let description = [{
    Reads a specific column from the 8×8 accumulator matrix back into
    a scalar register. Used after outer_product to extract results.
  }];
  let arguments = (ins I32:$acc, I32:$col);
  let results = (outs I32:$result);
  let assemblyFormat = "$acc `,` $col attr-dict `:` type($result)";
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

// --- DMA operations ---

def DmaLoadOp : CoralNPU_Op<"dma_load", [Pure]> {
  let summary = "DMA load from AXI to TCM";
  let description = [{
    Initiates a DMA transfer from external AXI memory to the tightly-coupled
    memory (TCM). The source address, destination address, and size are
    specified as scalar registers.
  }];
  let arguments = (ins I32:$src, I32:$dst, I32:$size);
  let results = (outs);
  let assemblyFormat = "$src `,` $dst `,` $size attr-dict";
}

def DmaStoreOp : CoralNPU_Op<"dma_store", [Pure]> {
  let summary = "DMA store from TCM to AXI";
  let description = [{
    Initiates a DMA transfer from the tightly-coupled memory (TCM) to
    external AXI memory.
  }];
  let arguments = (ins I32:$src, I32:$dst, I32:$size);
  let results = (outs);
  let assemblyFormat = "$src `,` $dst `,` $size attr-dict";
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

def BeqOp : CoralNPU_Op<"beq", [Terminator]> {
  let summary = "Branch if equal";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = "$lhs `,` $rhs `,` $trueDest `,` $falseDest attr-dict";
}

def BneOp : CoralNPU_Op<"bne", [Terminator]> {
  let summary = "Branch if not equal";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = "$lhs `,` $rhs `,` $trueDest `,` $falseDest attr-dict";
}

def JalOp : CoralNPU_Op<"jal", [Pure]> {
  let summary = "Jump and link (function call)";
  let arguments = (ins FlatSymbolRefAttr:$callee);
  let results = (outs I32:$ra);
  let assemblyFormat = "$callee attr-dict `:` type($ra)";
}

def JalrOp : CoralNPU_Op<"jalr", [Pure]> {
  let summary = "Jump and link register (indirect call via register)";
  let arguments = (ins I32:$rs);
  let results = (outs I32:$ra);
  let assemblyFormat = "$rs attr-dict `:` type($ra)";
}

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
  let assemblyFormat = "($value^ `:` type($value))? attr-dict";
}

#endif // CORALNPU_OPS_TD
