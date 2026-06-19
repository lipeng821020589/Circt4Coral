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
///   needed = ceil(tile_elements / vreg_capacity)
///   stripmine_factor = clamp(needed, 1, 4)
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

/// Read the constant element count from a vector-load op's `$n` operand,
/// i.e. the second operand produced by a `coralnpu.li`. Returns 0 if the
/// operand is not a known constant.
static int64_t loadElementCount(mlir::Operation *load) {
  if (load->getNumOperands() < 2)
    return 0;
  if (auto li = load->getOperand(1).getDefiningOp<ScalarLiOp>())
    return li.getValue();
  return 0;
}

/// Find the real element count this vector op processes by walking its
/// operands back to the producing vector load (vle8/vle16/vle32) and reading
/// the load's element count. Producers may themselves be earlier vector ops
/// (a register-chained element-wise sequence), so the walk follows i32
/// operands transitively. Returns 0 when no load anchors the chain (e.g.
/// hand-written IR whose operands are block arguments).
static int64_t traceElementCount(mlir::Operation *op) {
  llvm::SmallVector<mlir::Operation *, 8> worklist;
  llvm::SmallPtrSet<mlir::Operation *, 8> seen;
  for (auto v : op->getOperands())
    if (auto *def = v.getDefiningOp())
      worklist.push_back(def);

  while (!worklist.empty()) {
    auto *def = worklist.pop_back_val();
    if (!seen.insert(def).second)
      continue;
    if (mlir::isa<VLE8Op, VLE16Op, VLE32Op>(def)) {
      if (int64_t n = loadElementCount(def))
        return n;
    }
    for (auto v : def->getOperands())
      if (auto *d = v.getDefiningOp())
        worklist.push_back(d);
  }
  return 0;
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

    // Derive the element count from the real data flow: trace back to the
    // vector load that anchors this op's operands. Fall back to the
    // conservative 4×-capacity heuristic only when no load is found (e.g.
    // hand-written IR whose operands are block arguments).
    int64_t traced = traceElementCount(op);
    unsigned tileElements =
        traced > 0 ? (unsigned)traced : vregCap * 4;
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
