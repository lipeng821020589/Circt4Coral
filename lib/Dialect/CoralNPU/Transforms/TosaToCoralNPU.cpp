//===- TosaToCoralNPU.cpp - TOSA to CoralNPU lowering ---------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUPasses.h"
#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace circt::coralnpu;

namespace circt {
namespace coralnpu {
#define GEN_PASS_DEF_TOSATOCORALNPU
#include "circt/Dialect/CoralNPU/Passes.h.inc"
} // namespace coralnpu
} // namespace circt

namespace {

struct TosaAddToCoralNPU : public mlir::OpRewritePattern<ScalarAddOp> {
  using mlir::OpRewritePattern<ScalarAddOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(ScalarAddOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    return mlir::success();
  }
};

struct TosaToCoralNPUPass
    : public impl::TosaToCoralNPUBase<TosaToCoralNPUPass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<TosaAddToCoralNPU>(ctx);
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace circt {
namespace coralnpu {
std::unique_ptr<mlir::Pass> createTosaToCoralNPUPass() {
  return std::make_unique<TosaToCoralNPUPass>();
}
} // namespace coralnpu
} // namespace circt
