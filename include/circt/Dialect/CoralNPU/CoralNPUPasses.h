//===- CoralNPUPasses.h - CoralNPU pass declarations ------------*- C++ -*-===//
//
// Part of the CoralNPU+CIRCT Integration Project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CORALNPU_CORALNPUPASSES_H
#define CIRCT_DIALECT_CORALNPU_CORALNPUPASSES_H

#include "mlir/Pass/Pass.h"

namespace circt {
namespace coralnpu {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "circt/Dialect/CoralNPU/Passes.h.inc"

// Generate the pass registration.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/CoralNPU/Passes.h.inc"

} // namespace coralnpu
} // namespace circt

#endif // CIRCT_DIALECT_CORALNPU_CORALNPUPASSES_H
