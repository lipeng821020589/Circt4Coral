//===- CoralNPUOps.cpp - CoralNPU operation implementations ----------------===//
//
// Part of the CoralNPU+CIRCT Integration Project.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace circt::coralnpu;

//===----------------------------------------------------------------------===//
// ScalarLiOp folder
//===----------------------------------------------------------------------===//

mlir::OpFoldResult ScalarLiOp::fold(FoldAdaptor adaptor) {
  // Constant folding: li(42) always folds to 42.
  return adaptor.getValueAttr();
}

// Include the auto-generated operation implementations.
#define GET_OP_CLASSES
#include "circt/Dialect/CoralNPU/CoralNPU.cpp.inc"
