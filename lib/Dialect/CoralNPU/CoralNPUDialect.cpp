//===- CoralNPUDialect.cpp - CoralNPU dialect implementation ---------------===//
//
// Part of the CoralNPU+CIRCT Integration Project.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUDialect.h"
#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "circt/Dialect/HW/HWDialect.h"

using namespace circt;
using namespace circt::coralnpu;

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void CoralNPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/CoralNPU/CoralNPU.cpp.inc"
      >();
}

// Provide implementations for the enums we use.
#include "circt/Dialect/CoralNPU/CoralNPUEnums.cpp.inc"

// Include the auto-generated dialect implementation.
#include "circt/Dialect/CoralNPU/CoralNPUDialect.cpp.inc"
