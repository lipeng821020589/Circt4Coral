//===- TosaBufferizableOpInterfaceImpl.cpp - TOSA bufferization ---===//
//
// Placeholder for TOSA BufferizableOpInterface implementations.
//
// STATUS: Analysis-only implementations exist but full bufferize() support
// requires either --tosa-to-linalg lowering or per-op memref conversions
// (future P3 work). See bufferize-pipeline.mlir for the currently supported
// --one-shot-bufferize='allow-unknown-ops bufferize-function-boundaries'
// pipeline.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace coralnpu {

/// Register TOSA ops as bufferizable (analysis + bufferize implementation).
/// Currently a no-op: TOSA ops require allow-unknown-ops in the pipeline.
/// Full implementation pending P3 (tosa-to-linalg or per-op memref lowering).
void registerTosaBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  // No registration: TOSA ops are treated as unknown ops by one-shot-bufferize.
  // Use: --one-shot-bufferize='allow-unknown-ops bufferize-function-boundaries'
  (void)registry;
}

} // namespace coralnpu
} // namespace circt
