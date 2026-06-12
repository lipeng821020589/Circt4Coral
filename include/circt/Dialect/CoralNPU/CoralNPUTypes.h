//===- CoralNPUTypes.h - CoralNPU type declarations -------------*- C++ -*-===//
//
// Part of the CoralNPU+CIRCT Integration Project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CORALNPU_CORALNPUTYPES_H
#define CIRCT_DIALECT_CORALNPU_CORALNPUTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

// Include the auto-generated type declarations.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/CoralNPU/CoralNPUTypes.h.inc"

#endif // CIRCT_DIALECT_CORALNPU_CORALNPUTYPES_H
