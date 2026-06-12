//===- CoralNPUTypes.cpp - CoralNPU type implementations -----------------===//
//
// Part of the CoralNPU+CIRCT Integration Project.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUDialect.h"
#include "circt/Dialect/CoralNPU/CoralNPUTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::coralnpu;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Auto-generated type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/CoralNPU/CoralNPUTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// AccType verification
//===----------------------------------------------------------------------===//

LogicalResult AccType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    unsigned rows, unsigned cols) {
  if (rows != 8)
    return emitError() << "Coral NPU accumulator must have 8 rows, got " << rows;
  if (cols != 8)
    return emitError() << "Coral NPU accumulator must have 8 columns, got " << cols;
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Dialect type registration
//===----------------------------------------------------------------------===//

void CoralNPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/CoralNPU/CoralNPUTypes.cpp.inc"
      >();
}
