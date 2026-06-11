//===- CoralNPUOps.h - CoralNPU operation declarations ----------*- C++ -*-===//
//
// Part of the CoralNPU+CIRCT Integration Project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CORALNPU_CORALNPUOPS_H
#define CIRCT_DIALECT_CORALNPU_CORALNPUOPS_H

#include "circt/Dialect/CoralNPU/CoralNPUDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the auto-generated operation declarations.
#define GET_OP_CLASSES
#include "circt/Dialect/CoralNPU/CoralNPU.h.inc"

#endif // CIRCT_DIALECT_CORALNPU_CORALNPUOPS_H
