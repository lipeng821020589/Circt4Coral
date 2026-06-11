//===- CoralNPUStripmine.cpp - Stripmine scheduling -----------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUPasses.h"
#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace circt;
using namespace circt::coralnpu;

namespace circt {
namespace coralnpu {
#define GEN_PASS_DEF_CORALNPUSTRIPMINE
#include "circt/Dialect/CoralNPU/Passes.h.inc"
} // namespace coralnpu
} // namespace circt

namespace {

static int computeStripmineFactor(mlir::Operation *op) {
  (void)op;
  return 1;
}

struct CoralNPUStripminePass
    : public impl::CoralNPUStripmineBase<CoralNPUStripminePass> {
  void runOnOperation() override {
    getOperation()->walk([&](mlir::Operation *op) {
      if (auto vaddOp = mlir::dyn_cast<VAddOp>(op)) {
        int factor = computeStripmineFactor(op);
        vaddOp.setStripmineAttr(
            mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32), factor));
      }
      if (auto vdotOp = mlir::dyn_cast<VDotOp>(op)) {
        int factor = computeStripmineFactor(op);
        vdotOp.setStripmineAttr(
            mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32), factor));
      }
    });
  }
};

} // namespace

namespace circt {
namespace coralnpu {
std::unique_ptr<mlir::Pass> createCoralNPUStripminePass() {
  return std::make_unique<CoralNPUStripminePass>();
}
} // namespace coralnpu
} // namespace circt
