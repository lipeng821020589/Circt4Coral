//===- CoralNPUStripmine.cpp - Stripmine scheduling -----------------------===//
//
// Implements the stripmine scheduling algorithm for Coral NPU vector ops.
// Stripmine controls how many times the hardware issues a single SIMD
// instruction (1/2/4), enabling up to 4× data processing per dispatch.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUPasses.h"
#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::coralnpu;

namespace circt {
namespace coralnpu {
#define GEN_PASS_DEF_CORALNPUSTRIPMINE
#include "circt/Dialect/CoralNPU/Passes.h.inc"
} // namespace coralnpu
} // namespace circt

namespace {

//===----------------------------------------------------------------------===//
// Stripmine factor computation
//===----------------------------------------------------------------------===//

/// Compute the number of elements a single vector register can hold
/// for a given SEW (Selected Element Width).
static unsigned getVRegCapacity(SEW sew) {
  constexpr unsigned kVLEN = 128; // Coral NPU SIMD width in bits
  switch (sew) {
  case SEW::E8:  return kVLEN / 8;   // 16 elements
  case SEW::E16: return kVLEN / 16;  // 8 elements
  case SEW::E32: return kVLEN / 32;  // 4 elements
  }
  return 4; // fallback
}

/// Compute optimal stripmine factor for a given tile size and SEW.
///
/// Algorithm per ARCHITECTURE.md §4.3:
///   needed = ceil(tile_elements / vreg_capacity)
///   stripmine_factor = min(4, needed)
///
/// Returns a value in {1, 2, 4}.
static unsigned computeStripmineFactor(unsigned tileElements, SEW sew) {
  unsigned vregCap = getVRegCapacity(sew);
  if (vregCap == 0)
    return 1;

  unsigned needed = (tileElements + vregCap - 1) / vregCap; // ceil division
  if (needed <= 1)
    return 1;
  if (needed == 2)
    return 2;
  // needed >= 3 → use maximum stripmine (4)
  return 4;
}

/// Attempt to determine the current SEW from the IR context.
/// Walks backwards from the op to find the nearest vsetvl.
static SEW getCurrentSEW(mlir::Operation *op) {
  auto *block = op->getBlock();
  auto opIt = mlir::Block::iterator(op);
  while (opIt != block->begin()) {
    --opIt;
    if (auto vsetvl = mlir::dyn_cast<VSetVLOp>(&*opIt))
      return vsetvl.getSew();
  }
  return SEW::E32; // default to 32-bit
}

/// Set the stripmine attribute on an operation.
template <typename OpTy>
static void setStripmine(OpTy op, unsigned factor) {
  auto intTy = mlir::IntegerType::get(op->getContext(), 32);
  op.setStripmineAttr(mlir::IntegerAttr::get(intTy, factor));
}

//===----------------------------------------------------------------------===//
// Stripmine rewrite patterns
//===----------------------------------------------------------------------===//

/// Generic pattern to optimize stripmine for any vector op that supports it.
template <typename OpTy>
struct StripminePattern : public mlir::OpRewritePattern<OpTy> {
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(OpTy op,
                                      mlir::PatternRewriter &rewriter) const override {
    // Only optimize if stripmine is not already explicitly set
    // (stripmine=1 means default/unset)
    if (op.getStripmine() != 1)
      return mlir::failure();

    SEW sew = getCurrentSEW(op);
    unsigned vregCap = getVRegCapacity(sew);

    // Compute the effective number of elements this op processes.
    // For now, use a heuristic: if the op produces a result, estimate
    // from the data flow. Otherwise default to vregCap.
    unsigned tileElements = vregCap * 4; // assume 4× vreg capacity by default
    unsigned factor = computeStripmineFactor(tileElements, sew);

    if (factor <= 1)
      return mlir::failure();

    setStripmine(op, factor);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct CoralNPUStripminePass
    : public impl::CoralNPUStripmineBase<CoralNPUStripminePass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    // Apply stripmine optimization to all supported vector ops
    patterns.add<
      StripminePattern<VAddOp>,
      StripminePattern<VSubOp>,
      StripminePattern<VMulOp>,
      StripminePattern<VWAddOp>,
      StripminePattern<VDotOp>,
      StripminePattern<OuterProductOp>
    >(ctx);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
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
