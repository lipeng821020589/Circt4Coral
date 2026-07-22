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

/// Register the export-coralnpu translation.
void registerExportCoralNPUTranslation();

/// Register BufferizableOpInterface external models for TOSA dialect ops.
/// Call this before running --one-shot-bufferize on TOSA IR.
void registerTosaBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);

} // namespace coralnpu
} // namespace circt

#endif // CIRCT_DIALECT_CORALNPU_CORALNPUPASSES_H
