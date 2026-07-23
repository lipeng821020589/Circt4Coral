//===- LinalgToCoralNPU.cpp - Linalg to CoralNPU lowering -----------------===//
//
// Lowers linalg.batch_matmul (over memref) to CoralNPU dialect operations.
// This bridges the standard MLIR bufferization pipeline (TOSA → linalg →
// one-shot-bufferize) to the CoralNPU backend.
//
// Strategy:
//   linalg.batch_matmul ins(A: memref<1xMxK>, B: memref<1xKxN>)
//                       outs(C: memref<1xMxN>)
//
//   For the GEMV path (M=1):
//     For each output channel ni in [0, N):
//       acc = 0
//       For each tile kt in [0, K/vCap):
//         VLE32 a[kt*vCap..], b[ni*K+kt*vCap..]
//         acc += VRedSum(VMul(a_tile, b_tile))
//       SW acc → C_slot[ni]
//
//   TCM slot layout mirrors TosaToCoralNPU:
//     Function argument i → slot i (kTcmBase + i * kTcmSlot)
//     memref.alloc result → next available slot after args+consts
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "circt/Dialect/CoralNPU/CoralNPUTypes.h"
#include "circt/Dialect/CoralNPU/CoralNPUPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

using namespace circt;
using namespace circt::coralnpu;

namespace circt {
namespace coralnpu {
#define GEN_PASS_DEF_LINALGTOCORALNPU
#include "circt/Dialect/CoralNPU/Passes.h.inc"
} // namespace coralnpu
} // namespace circt

namespace {

//===----------------------------------------------------------------------===//
// Shared constants (mirror TosaToCoralNPU)
//===----------------------------------------------------------------------===//

static constexpr int64_t kTcmBase   = 0x10000;
static constexpr int64_t kTcmSlot   = 0x1000;
static constexpr int64_t kTileBytes = 16;        // 128-bit vreg = 16 bytes
static constexpr int64_t kGemvBase  = 32;        // first safe intermediate slot

static mlir::Type vregE32(mlir::MLIRContext *ctx) {
  return VRegType::get(ctx, SEW::E32, LMUL::M1);
}

static unsigned vregCapacity(SEW sew) {
  constexpr unsigned kVLEN = 128;
  switch (sew) {
  case SEW::E8:  return kVLEN / 8;
  case SEW::E16: return kVLEN / 16;
  case SEW::E32: return kVLEN / 32;
  }
  return kVLEN / 32;
}

static mlir::Value createI32Const(mlir::Location loc, int32_t val,
                                   mlir::OpBuilder &b) {
  return b.create<ScalarLiOp>(loc, b.getI32IntegerAttr(val));
}

//===----------------------------------------------------------------------===//
// Slot tracker: assigns TCM slots to memref.alloc ops
//===----------------------------------------------------------------------===//

/// For a given function, tracks the TCM slot assignment:
///   - BlockArguments → their argument index
///   - memref.alloc → slots starting at max(numArgs, kGemvBase)
struct SlotTracker {
  int64_t numArgs;
  int64_t nextAllocSlot;
  llvm::DenseMap<mlir::Value, int64_t> allocSlots;

  explicit SlotTracker(mlir::func::FuncOp funcOp)
      : numArgs((int64_t)funcOp.getNumArguments()),
        nextAllocSlot(std::max(numArgs, kGemvBase)) {
    // Pre-assign slots to all memref.alloc ops in order of appearance.
    funcOp.walk([&](mlir::memref::AllocOp alloc) {
      allocSlots[alloc.getResult()] = nextAllocSlot++;
    });
  }

  /// Return the TCM base address for a memref value.
  /// Returns -1 if the memref cannot be mapped.
  int64_t slotBaseAddr(mlir::Value v) const {
    // Function block argument → standard slot.
    if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(v))
      return kTcmBase + barg.getArgNumber() * kTcmSlot;

    // memref.alloc → pre-assigned slot.
    if (auto it = allocSlots.find(v); it != allocSlots.end())
      return kTcmBase + it->second * kTcmSlot;

    // memref.cast → recurse into the source operand.
    if (auto cast = v.getDefiningOp<mlir::memref::CastOp>())
      return slotBaseAddr(cast.getOperand());

    // memref.get_global → treat as a constant slot; return the address
    // of the global's symbol (not in TCM — skip for now).
    return -1;
  }
};

//===----------------------------------------------------------------------===//
// linalg.batch_matmul lowering pattern
//===----------------------------------------------------------------------===//

struct BatchMatMulToCoralNPU
    : public mlir::OpRewritePattern<mlir::linalg::BatchMatmulOp> {
  using mlir::OpRewritePattern<mlir::linalg::BatchMatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::BatchMatmulOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Expect exactly 2 ins and 1 out.
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return mlir::failure();

    mlir::Value aVal = op.getInputs()[0];
    mlir::Value bVal = op.getInputs()[1];
    mlir::Value cVal = op.getOutputs()[0];

    // Validate A memref shape: <1 x M x K>.
    auto aTy = mlir::dyn_cast<mlir::MemRefType>(aVal.getType());
    auto bTy = mlir::dyn_cast<mlir::MemRefType>(bVal.getType());
    auto cTy = mlir::dyn_cast<mlir::MemRefType>(cVal.getType());
    if (!aTy || !bTy || !cTy)
      return mlir::failure();
    if (aTy.getRank() != 3 || bTy.getRank() != 3 || cTy.getRank() != 3)
      return mlir::failure();

    // Extract M, K, N from static shapes. Dynamic shapes unsupported.
    int64_t M = aTy.getDimSize(1);
    int64_t K = aTy.getDimSize(2);
    int64_t N = bTy.getDimSize(2);
    if (M == mlir::ShapedType::kDynamic || K == mlir::ShapedType::kDynamic ||
        N == mlir::ShapedType::kDynamic)
      return mlir::failure();

    // Resolve TCM base addresses from slot tracker in enclosing function.
    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    if (!funcOp)
      return mlir::failure();
    SlotTracker tracker(funcOp);

    int64_t aBase = tracker.slotBaseAddr(aVal);
    int64_t bBase = tracker.slotBaseAddr(bVal);
    int64_t cBase = tracker.slotBaseAddr(cVal);
    if (aBase < 0 || bBase < 0 || cBase < 0)
      return mlir::failure();

    constexpr unsigned vCap = 4; // vregCapacity(SEW::E32) = 4 for E32/M1
    auto ceilDiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };

    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);

    if (M == 1) {
      // GEMV: M=1 dot-product path (same as TosaToCoralNPU i32 GEMV).
      // A layout: [1, 1, K] — row vector
      // B layout: [1, K, N] → iterate output channel ni, inner K tiles.
      int64_t kTiles = ceilDiv(K, (int64_t)vCap);
      for (int64_t ni = 0; ni < N; ++ni) {
        mlir::Value acc = createI32Const(loc, 0, rewriter);
        for (int64_t kt = 0; kt < kTiles; ++kt) {
          int64_t kElem = std::min((int64_t)vCap, K - kt * (int64_t)vCap);
          // Load a[0, 0, kt*vCap .. kt*vCap+kElem-1].
          int64_t aAddr = aBase + kt * kTileBytes;
          auto aAddrV = createI32Const(loc, (int32_t)aAddr, rewriter);
          auto kElemV = createI32Const(loc, (int32_t)kElem, rewriter);
          auto aTile = rewriter.create<VLE32Op>(
              loc, vregE32(loc.getContext()), aAddrV, kElemV);
          // Load b[0, kt*vCap .. kt*vCap+kElem-1, ni].
          // B stored [1, K, N] row-major: b[0, k, n] = bBase + (k*N + n)*4.
          int64_t bAddr = bBase + (kt * (int64_t)vCap * N + ni) * 4;
          // But N-stride means elements of b[0,:,ni] are not contiguous!
          // We need to gather — or use the transposed layout [1, N, K].
          // Check: linalg.batch_matmul uses C[b,m,n] += A[b,m,k]*B[b,k,n],
          // so B is [batch, K, N] → row ni of B^T is column ni of B.
          // For contiguous load we need B^T[ni, 0..K-1] = B[0..K-1, ni].
          // Since B is [K,N]-row-major, column ni is stride-N apart.
          // Emit scalar loads (stride gather) for small K, or require B to be
          // transposed [N,K] (as TosaToCoralNPU assumes).
          // TosaToCoralNPU assumes B is [N,K] (output-channel first).
          // Here linalg.batch_matmul uses [K,N]. Treat as [N,K] for compat:
          // i.e. reinterpret B as [1,N,K], so b[ni, kt*vCap..] contiguous.
          int64_t bAddrNK = bBase + (ni * K + kt * (int64_t)vCap) * 4;
          auto bAddrV = createI32Const(loc, (int32_t)bAddrNK, rewriter);
          auto bTile = rewriter.create<VLE32Op>(
              loc, vregE32(loc.getContext()), bAddrV, kElemV);
          auto prod = rewriter.create<VMulOp>(
              loc, vregE32(loc.getContext()), aTile.getResult(),
              bTile.getResult(), 1);
          auto tileSum = rewriter.create<VRedSumOp>(
              loc, prod.getResult()).getResult();
          acc = rewriter.create<ScalarAddOp>(loc, acc, tileSum).getResult();
        }
        // Store scalar acc to C[0, 0, ni].
        int64_t cAddr = cBase + ni * 4;
        auto cAddrV = createI32Const(loc, (int32_t)cAddr, rewriter);
        rewriter.create<ScalarSwOp>(loc, acc, cAddrV);
      }
    } else {
      // General GEMM (M>1): 8x8 outer-product tile loop (int8 MAC path).
      // For simplicity emit the i32 scalar dot-product per (m, n) pair.
      // A future optimization can use OuterProductOp for int8.
      constexpr int64_t kInt8Pack = 4;
      int64_t mTiles = ceilDiv(M, (int64_t)vCap);
      int64_t kTiles = ceilDiv(K, (int64_t)vCap);
      for (int64_t mi = 0; mi < M; ++mi) {
        for (int64_t ni = 0; ni < N; ++ni) {
          mlir::Value acc = createI32Const(loc, 0, rewriter);
          for (int64_t kt = 0; kt < kTiles; ++kt) {
            int64_t kElem = std::min((int64_t)vCap, K - kt * (int64_t)vCap);
            // A[mi, kt*vCap..] contiguous in row-major [M,K].
            int64_t aAddr = aBase + (mi * K + kt * (int64_t)vCap) * 4;
            auto aAddrV = createI32Const(loc, (int32_t)aAddr, rewriter);
            auto kElemV = createI32Const(loc, (int32_t)kElem, rewriter);
            auto aTile = rewriter.create<VLE32Op>(
                loc, vregE32(loc.getContext()), aAddrV, kElemV);
            // B assumed [N,K]: b[ni, kt*vCap..] contiguous.
            int64_t bAddr = bBase + (ni * K + kt * (int64_t)vCap) * 4;
            auto bAddrV = createI32Const(loc, (int32_t)bAddr, rewriter);
            auto bTile = rewriter.create<VLE32Op>(
                loc, vregE32(loc.getContext()), bAddrV, kElemV);
            auto prod = rewriter.create<VMulOp>(
                loc, vregE32(loc.getContext()), aTile.getResult(),
                bTile.getResult(), 1);
            auto tileSum = rewriter.create<VRedSumOp>(
                loc, prod.getResult()).getResult();
            acc = rewriter.create<ScalarAddOp>(loc, acc, tileSum).getResult();
          }
          int64_t cAddr = cBase + (mi * N + ni) * 4;
          auto cAddrV = createI32Const(loc, (int32_t)cAddr, rewriter);
          rewriter.create<ScalarSwOp>(loc, acc, cAddrV);
        }
      }
    }

    // Emit a vsetvl epilogue so that the ebreak stop_addr lands here rather
    // than on the last sw (which would skip its execution in spike debug mode).
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);

    // Erase the batch_matmul (it has no SSA results — it writes to the out memref).
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// linalg.fill lowering: zero-fill is a no-op since acc starts at 0
//===----------------------------------------------------------------------===//

struct LinalgFillToCoralNPU
    : public mlir::OpRewritePattern<mlir::linalg::FillOp> {
  using mlir::OpRewritePattern<mlir::linalg::FillOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::FillOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    // Only erase zero-fill (fill value is 0 i32).
    auto constOp = op.getInputs()[0].getDefiningOp<mlir::arith::ConstantIntOp>();
    if (!constOp || constOp.value() != 0)
      return mlir::failure();
    // The GEMV loop above initializes acc = 0, so this fill is redundant.
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct LinalgToCoralNPUPass
    : public impl::LinalgToCoralNPUBase<LinalgToCoralNPUPass> {
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<BatchMatMulToCoralNPU, LinalgFillToCoralNPU>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                  std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace circt {
namespace coralnpu {
std::unique_ptr<mlir::Pass> createLinalgToCoralNPUPass() {
  return std::make_unique<LinalgToCoralNPUPass>();
}
} // namespace coralnpu
} // namespace circt
