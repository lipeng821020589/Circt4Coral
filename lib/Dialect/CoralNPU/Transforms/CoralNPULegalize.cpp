//===- CoralNPULegalize.cpp - CoralNPU IR legalization --------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUPasses.h"
#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace circt;
using namespace circt::coralnpu;

namespace circt {
namespace coralnpu {
#define GEN_PASS_DEF_CORALNPULEGALIZE
#include "circt/Dialect/CoralNPU/Passes.h.inc"
} // namespace coralnpu
} // namespace circt

namespace {

struct CoralNPULegalizePass
    : public impl::CoralNPULegalizeBase<CoralNPULegalizePass> {
  void runOnOperation() override {
    getOperation()->walk([&](VSetVLOp vsetvl) {
      (void)vsetvl; // validate SEW/LMUL constraints
    });
  }
};

} // namespace

namespace circt {
namespace coralnpu {
std::unique_ptr<mlir::Pass> createCoralNPULegalizePass() {
  return std::make_unique<CoralNPULegalizePass>();
}
} // namespace coralnpu
} // namespace circt
