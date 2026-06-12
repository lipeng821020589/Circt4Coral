//===- CoralNPUDialect.h - CoralNPU dialect declaration ---------*- C++ -*-===//
//
// Part of the CoralNPU+CIRCT Integration Project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CORALNPU_CORALNPUDIALECT_H
#define CIRCT_DIALECT_CORALNPU_CORALNPUDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/TypeSwitch.h"

// Include the auto-generated enum declarations.
#include "circt/Dialect/CoralNPU/CoralNPUEnums.h.inc"

// Include the type declarations.
#include "circt/Dialect/CoralNPU/CoralNPUTypes.h"

// Include the auto-generated attribute declarations.
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/CoralNPU/CoralNPUAttributes.h.inc"

// Include the auto-generated dialect declaration.
#include "circt/Dialect/CoralNPU/CoralNPUDialect.h.inc"

#endif // CIRCT_DIALECT_CORALNPU_CORALNPUDIALECT_H
