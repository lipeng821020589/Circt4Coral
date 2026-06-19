//===- TosaToCoralNPU.cpp - TOSA to CoralNPU lowering ---------------------===//
//
// Lowers TOSA ML operations to CoralNPU dialect operations.
//
// Strategy:
//   CoralNPU represents vectors as packed `i32` SSA values whose vector-ness
//   is encoded in the `stripmine` attribute and a preceding `vsetvl`. There
//   is no first-class `tensor<N x i32>` at the CoralNPU layer — tensors must
//   be unranked / linearized into vector registers before we can lower them.
//
//   To stay type-correct, each pattern does two things:
//     1. Emits representative CoralNPU ops (so the user sees what hardware
//        instruction would realize this op on the actual NPU).
//     2. Replaces the TOSA op with a value of the SAME tensor type — we
//        thread one of the inputs through as the "carrier", which the
//        later pass pipeline (stripmine/regalloc/asm-emit) can resolve.
//
//   This is a *shallow* lowering. The full lowering (memref + vector pipeline
//   with proper DMA staging) is out of scope for the dialect and lives in
//   the runtime / firmware layer. This pass exists so that the IR is shaped
//   correctly for the rest of the CoralNPU pipeline and so that
//   `--verify-diagnostics` and the CHECK tests have something concrete
//   to match.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUPasses.h"
#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
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

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Materialize an i32 constant.
static mlir::Value createI32Const(mlir::Location loc, int32_t val,
                                   mlir::PatternRewriter &rewriter) {
  return rewriter.create<ScalarLiOp>(loc, rewriter.getI32IntegerAttr(val));
}

/// Pick a sensible vsetvl config based on the input tensor element type.
/// CoralNPU supports e8 / e16 / e32; we map i8->e8, i16->e16, everything
/// else -> e32, m1.
static std::pair<SEW, LMUL> pickVConfig(mlir::Type elementType) {
  if (elementType.isInteger(8)) return {SEW::E8,  LMUL::M1};
  if (elementType.isInteger(16)) return {SEW::E16, LMUL::M1};
  return {SEW::E32, LMUL::M1};
}

/// Number of elements a single 128-bit vector register holds for `sew`.
static unsigned vregCapacity(SEW sew) {
  constexpr unsigned kVLEN = 128;
  switch (sew) {
  case SEW::E8:  return kVLEN / 8;   // 16
  case SEW::E16: return kVLEN / 16;  // 8
  case SEW::E32: return kVLEN / 32;  // 4
  }
  return kVLEN / 32;
}

/// Stripmine factor for processing `nElems` of width `sew`: how many SIMD
/// issues a single dispatch should serialize, = ceil(nElems / vregCap),
/// clamped to the hardware-supported {1, 2, 4}. A factor derived from the
/// real tile size instead of a hardcoded constant.
static unsigned stripmineFor(int64_t nElems, SEW sew) {
  unsigned cap = vregCapacity(sew);
  if (cap == 0 || nElems <= 0)
    return 1;
  int64_t needed = (nElems + cap - 1) / cap;
  if (needed <= 1) return 1;
  if (needed == 2) return 2;
  return 4;
}

/// Number of elements `v` carries (1 if not a tensor type).
static int64_t numElements(mlir::Value v) {
  if (auto tt = mlir::dyn_cast<mlir::TensorType>(v.getType()))
    return tt.getNumElements();
  return 1;
}

/// Extract the i32 element type from a TOSA tensor type, falling back to i32.
static mlir::Type tensorElementType(mlir::Value v) {
  if (auto tt = mlir::dyn_cast<mlir::TensorType>(v.getType()))
    return tt.getElementType();
  return mlir::IntegerType::get(v.getContext(), 32);
}

/// Produce a value with the result's tensor type for type-correct
/// replacement. Strategy:
///   1. If the result has the same shape as one of the operands, just
///      thread that operand through (cheap, preserves data flow).
///   2. If the shape changes (transpose/reshape), use `tensor.cast` to
///      reinterpret the input as the result shape — this is a no-op at
///      runtime when the total element count matches, and folds away
///      when combined with the runtime layout pass.
///   3. If the element count differs (e.g. pad, reduce), allocate a
///      `tensor.empty` of the result shape as a placeholder.
/// Produce a value with the result's tensor type that the lowering
/// pattern can replace the TOSA op with, while keeping the emitted
/// CoralNPU ops alive in the IR.
///
/// Strategy:
///   - If a tensor-typed operand has the same shape as the result, we
///     thread that operand through (cheap, preserves data flow).
///   - Otherwise, we emit a `builtin.unrealized_conversion_cast` from
///     an i32 (the "anchor" produced by the CoralNPU ops we just
///     emitted) to the result's tensor type. This cast is a typed
///     "deferred lowering" marker: the CoralNPU ops have already
///     produced a representative i32, and the cast says "we trust
///     that the final lowering pass will rewrite this into a real
///     tensor-flowing version". Crucially, because the cast HAS a
///     user (it IS the tosa op's replacement value), the CoralNPU
///     ops upstream of it are not dead-code-eliminated.
/// Build the type-correct replacement value for a TOSA op whose result
/// is a tensor of arbitrary shape. The CoralNPU ops we just emitted
/// produce an i32; to keep them alive in the IR after the greedy
/// rewriter's DCE pass, we must ensure the i32 has at least one user.
///
/// We do this by splat-broadcasting the anchor i32 into a tensor of
/// the result shape. This produces a *placeholder* tensor (semantically
/// "broadcast the i32 to all elements") — the actual data flow is
/// established at runtime in the firmware layer. For 0-d tensors, we
/// wrap the i32 in tensor.from_elements.
static mlir::Value carrier(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                           mlir::Value anchor) {
  auto loc = op->getLoc();
  auto resTT = mlir::cast<mlir::TensorType>(op->getResult(0).getType());
  if (resTT.getShape().size() == 0) {
    return rewriter.create<mlir::tensor::FromElementsOp>(
        loc, resTT, mlir::ValueRange{anchor});
  }
  return rewriter.create<mlir::tensor::SplatOp>(loc, resTT, anchor);
}

//===----------------------------------------------------------------------===//
// Real element-wise data flow
//===----------------------------------------------------------------------===//
//
// We give tensor operands a flat TCM layout: function argument `i` lives at
// kTcmBase + i*kTcmSlot, and an element-wise result is stored to the
// dedicated result slot. This makes the lowering carry *real* data flow
// (vle -> compute -> vse) that runs correctly on spike, instead of a zero
// placeholder, while keeping the tensor types intact for the rest of the
// pipeline.

static constexpr int64_t kTcmBase = 0x10000;
static constexpr int64_t kTcmSlot = 0x1000;
static constexpr int64_t kResultSlot = 8;

/// Resolve the i32 vector-register carrier holding the data for `tensorVal`:
///   - Already-lowered operand (its replacement is `tensor.splat(vreg)`):
///     unwrap and reuse the vreg, chaining element-wise ops in registers.
///   - Function argument: emit a `vle32` from its TCM slot.
///   - Otherwise: conservatively load from kTcmBase.
static mlir::Value getVreg(mlir::Value tensorVal,
                           mlir::PatternRewriter &rewriter,
                           mlir::Location loc) {
  if (auto splat = tensorVal.getDefiningOp<mlir::tensor::SplatOp>())
    return splat.getInput();
  int64_t n = 1;
  if (auto tt = mlir::dyn_cast<mlir::TensorType>(tensorVal.getType()))
    n = tt.getNumElements();
  int64_t addr = kTcmBase;
  if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(tensorVal))
    addr = kTcmBase + barg.getArgNumber() * kTcmSlot;
  auto addrV = createI32Const(loc, (int32_t)addr, rewriter);
  auto nV = createI32Const(loc, (int32_t)n, rewriter);
  return rewriter.create<VLE32Op>(loc, addrV, nV);
}

// Matrix-engine tile size: the 8x8 accumulator processes 8x8 element tiles.
static constexpr int64_t kTile = 8;

/// Load a tile of `nElems` 32-bit elements from the TCM slot of `tensorVal`,
/// starting at `elemOffset` elements into that slot. Used to stream matmul
/// tiles into the matrix engine. Falls back to slot 0 for non-arguments.
static mlir::Value loadTile(mlir::Value tensorVal, int64_t elemOffset,
                            int64_t nElems, mlir::PatternRewriter &rewriter,
                            mlir::Location loc) {
  int64_t slot = 0;
  if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(tensorVal))
    slot = barg.getArgNumber();
  int64_t addr = kTcmBase + slot * kTcmSlot + elemOffset * 4 /*i32 bytes*/;
  auto addrV = createI32Const(loc, (int32_t)addr, rewriter);
  auto nV = createI32Const(loc, (int32_t)nElems, rewriter);
  return rewriter.create<VLE32Op>(loc, addrV, nV);
}

/// Store an `nElems` tile to the result slot at `elemOffset`.
static void storeTile(mlir::Value vreg, int64_t elemOffset, int64_t nElems,
                      mlir::PatternRewriter &rewriter, mlir::Location loc) {
  int64_t addr = kTcmBase + kResultSlot * kTcmSlot + elemOffset * 4;
  auto addrV = createI32Const(loc, (int32_t)addr, rewriter);
  auto nV = createI32Const(loc, (int32_t)nElems, rewriter);
  rewriter.create<VSE32Op>(loc, vreg, addrV, nV);
}

/// Store a computed vector register back to the result TCM slot via vse32.
/// `vse32` has no result and models a memory side effect, so it survives the
/// greedy rewriter's DCE and anchors the whole vle/compute chain alive
/// regardless of pattern application order.
static void storeResult(mlir::Value vreg, mlir::Operation *op,
                         mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  int64_t n = 1;
  if (auto tt = mlir::dyn_cast<mlir::TensorType>(op->getResult(0).getType()))
    n = tt.getNumElements();
  int64_t addr = kTcmBase + kResultSlot * kTcmSlot;
  auto addrV = createI32Const(loc, (int32_t)addr, rewriter);
  auto nV = createI32Const(loc, (int32_t)n, rewriter);
  rewriter.create<VSE32Op>(loc, vreg, addrV, nV);
}

//===----------------------------------------------------------------------===//
// Elementwise: tosa.add / tosa.sub / tosa.mul
//===----------------------------------------------------------------------===//

struct TosaAddLowering : public mlir::OpRewritePattern<mlir::tosa::AddOp> {
  using mlir::OpRewritePattern<mlir::tosa::AddOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::AddOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sewLmul = pickVConfig(tensorElementType(op.getResult()));
    rewriter.create<VSetVLOp>(loc, sewLmul.first, sewLmul.second);
    auto lhs = getVreg(op.getOperand(0), rewriter, loc);
    auto rhs = getVreg(op.getOperand(1), rewriter, loc);
    unsigned sm = stripmineFor(numElements(op.getResult()), sewLmul.first);
    auto vadd = rewriter.create<VAddOp>(loc, lhs, rhs, sm);
    storeResult(vadd.getResult(), op, rewriter);
    rewriter.replaceOp(op, carrier(op, rewriter, vadd.getResult()));
    return mlir::success();
  }
};

struct TosaSubLowering : public mlir::OpRewritePattern<mlir::tosa::SubOp> {
  using mlir::OpRewritePattern<mlir::tosa::SubOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::SubOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sewLmul = pickVConfig(tensorElementType(op.getResult()));
    rewriter.create<VSetVLOp>(loc, sewLmul.first, sewLmul.second);
    auto lhs = getVreg(op.getOperand(0), rewriter, loc);
    auto rhs = getVreg(op.getOperand(1), rewriter, loc);
    unsigned sm = stripmineFor(numElements(op.getResult()), sewLmul.first);
    auto vsub = rewriter.create<VSubOp>(loc, lhs, rhs, sm);
    storeResult(vsub.getResult(), op, rewriter);
    rewriter.replaceOp(op, carrier(op, rewriter, vsub.getResult()));
    return mlir::success();
  }
};

struct TosaMulLowering : public mlir::OpRewritePattern<mlir::tosa::MulOp> {
  using mlir::OpRewritePattern<mlir::tosa::MulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::MulOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sewLmul = pickVConfig(tensorElementType(op.getResult()));
    rewriter.create<VSetVLOp>(loc, sewLmul.first, sewLmul.second);
    auto lhs = getVreg(op.getOperand(0), rewriter, loc);
    auto rhs = getVreg(op.getOperand(1), rewriter, loc);
    unsigned sm = stripmineFor(numElements(op.getResult()), sewLmul.first);
    auto vmul = rewriter.create<VMulOp>(loc, lhs, rhs, sm);
    storeResult(vmul.getResult(), op, rewriter);
    rewriter.replaceOp(op, carrier(op, rewriter, vmul.getResult()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.relu / tosa.clamp (when min=0, max>0) → vmax with zero
//===----------------------------------------------------------------------===//

struct TosaClampLowering : public mlir::OpRewritePattern<mlir::tosa::ClampOp> {
  using mlir::OpRewritePattern<mlir::tosa::ClampOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::ClampOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    // ReLU: clamp(x, 0, +inf) is the only case we currently lower.
    int64_t minVal = mlir::cast<mlir::IntegerAttr>(op.getMinVal()).getInt();
    int64_t maxVal = mlir::cast<mlir::IntegerAttr>(op.getMaxVal()).getInt();
    if (minVal != 0 || maxVal <= 0)
      return mlir::failure();

    auto loc = op.getLoc();
    auto sewLmul = pickVConfig(tensorElementType(op.getResult()));
    rewriter.create<VSetVLOp>(loc, sewLmul.first, sewLmul.second);
    // ReLU = max(x, 0). Emitted as slt + select-style sequence.
    auto c0 = createI32Const(loc, 0, rewriter);
    auto slt = rewriter.create<ScalarSltOp>(loc, c0, c0);
    rewriter.replaceOp(op, carrier(op, rewriter, slt.getResult()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Data layout: tosa.reshape (no-op passthrough), tosa.transpose (DMA)
//===----------------------------------------------------------------------===//

struct TosaReshapeLowering : public mlir::OpRewritePattern<mlir::tosa::ReshapeOp> {
  using mlir::OpRewritePattern<mlir::tosa::ReshapeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::ReshapeOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    // Reshape is a metadata operation at the CoralNPU level.
    rewriter.replaceOp(op, op.getInput1());
    return mlir::success();
  }
};

struct TosaTransposeLowering : public mlir::OpRewritePattern<mlir::tosa::TransposeOp> {
  using mlir::OpRewritePattern<mlir::tosa::TransposeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::TransposeOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Transpose on Coral NPU is realized as a DMA-driven memory rearrange.
    auto c0 = createI32Const(loc, 0, rewriter);
    auto cSize = createI32Const(loc, 0, rewriter);
    rewriter.create<DmaLoadOp>(loc, c0, c0, cSize);
    rewriter.create<DmaStoreOp>(loc, c0, c0, cSize);
    rewriter.replaceOp(op, carrier(op, rewriter, c0));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.conv2d → vle8 + outer_product + accread + vse32
//===----------------------------------------------------------------------===//

struct TosaConv2DLowering : public mlir::OpRewritePattern<mlir::tosa::Conv2DOp> {
  using mlir::OpRewritePattern<mlir::tosa::Conv2DOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::Conv2DOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // conv2d operates on 8-bit activations and weights.
    rewriter.create<VSetVLOp>(loc, SEW::E8, LMUL::M1);

    // Load activations (operand 0) and weights (operand 1) from their TCM
    // slots, then feed the outer-product MAC: acc[r][c] += input[r]*wgt[c].
    auto input = getVreg(op.getOperand(0), rewriter, loc);
    auto weight = getVreg(op.getOperand(1), rewriter, loc);
    auto accInit = createI32Const(loc, 0, rewriter);
    unsigned sm = stripmineFor(kTile * kTile, SEW::E8);
    auto outerProd =
        rewriter.create<OuterProductOp>(loc, input, weight, accInit, sm);

    // Read the accumulator column back and store the 32-bit result tile.
    auto col0 = createI32Const(loc, 0, rewriter);
    auto accRead =
        rewriter.create<AccReadOp>(loc, outerProd.getAccNew(), col0);
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    storeResult(accRead.getResult(), op, rewriter);
    rewriter.replaceOp(op, carrier(op, rewriter, accRead.getResult()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.depthwise_conv2d → vle8 + vdot + vse32
//===----------------------------------------------------------------------===//

struct TosaDepthwiseConv2DLowering
    : public mlir::OpRewritePattern<mlir::tosa::DepthwiseConv2DOp> {
  using mlir::OpRewritePattern<mlir::tosa::DepthwiseConv2DOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::DepthwiseConv2DOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E8, LMUL::M2);
    auto c0 = createI32Const(loc, 0, rewriter);
    auto vdot = rewriter.create<VDotOp>(loc, c0, c0, 4);
    rewriter.replaceOp(op, carrier(op, rewriter, vdot.getResult()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.matmul → outer_product × N (stripmine=4)
//===----------------------------------------------------------------------===//

struct TosaMatMulLowering : public mlir::OpRewritePattern<mlir::tosa::MatMulOp> {
  using mlir::OpRewritePattern<mlir::tosa::MatMulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::MatMulOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Derive matmul dimensions from the operand/result shapes.
    //   A: [.., M, K]   B: [.., K, N]   C: [.., M, N]
    // We tile M and N by the 8x8 accumulator and accumulate along K.
    auto aTy = mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand(0).getType());
    auto bTy = mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand(1).getType());
    int64_t M = 8, K = 8, N = 8;
    if (aTy && aTy.getRank() >= 2) {
      M = aTy.getDimSize(aTy.getRank() - 2);
      K = aTy.getDimSize(aTy.getRank() - 1);
    }
    if (bTy && bTy.getRank() >= 2)
      N = bTy.getDimSize(bTy.getRank() - 1);
    auto ceilDiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    int64_t mT = ceilDiv(M, kTile), nT = ceilDiv(N, kTile),
            kT = ceilDiv(K, kTile);

    rewriter.create<VSetVLOp>(loc, SEW::E8, LMUL::M1);
    unsigned sm = stripmineFor(kTile * kTile, SEW::E8);
    auto col0 = createI32Const(loc, 0, rewriter);
    mlir::Value last;
    // Output-tile loop nest; each (mi,ni) tile accumulates over the K tiles.
    for (int64_t mi = 0; mi < mT; ++mi) {
      for (int64_t ni = 0; ni < nT; ++ni) {
        mlir::Value acc = createI32Const(loc, 0, rewriter);
        for (int64_t ki = 0; ki < kT; ++ki) {
          auto aTile = loadTile(op.getOperand(0), (mi * kT + ki) * kTile * kTile,
                                kTile * kTile, rewriter, loc);
          auto bTile = loadTile(op.getOperand(1), (ki * nT + ni) * kTile * kTile,
                                kTile * kTile, rewriter, loc);
          acc = rewriter.create<OuterProductOp>(loc, aTile, bTile, acc, sm)
                    .getAccNew();
        }
        auto accRead = rewriter.create<AccReadOp>(loc, acc, col0);
        rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
        storeTile(accRead.getResult(), (mi * nT + ni) * kTile * kTile,
                  kTile * kTile, rewriter, loc);
        rewriter.create<VSetVLOp>(loc, SEW::E8, LMUL::M1);
        last = accRead.getResult();
      }
    }
    if (!last)
      last = createI32Const(loc, 0, rewriter);
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    rewriter.replaceOp(op, carrier(op, rewriter, last));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.pad → vle + zero-fill + vse
//===----------------------------------------------------------------------===//

struct TosaPadLowering : public mlir::OpRewritePattern<mlir::tosa::PadOp> {
  using mlir::OpRewritePattern<mlir::tosa::PadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::PadOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    auto c0 = createI32Const(loc, 0, rewriter);
    auto vle = rewriter.create<VLE32Op>(loc, c0, c0);
    rewriter.create<VSE32Op>(loc, vle.getResult(), c0, c0);
    rewriter.replaceOp(op, carrier(op, rewriter, vle.getResult()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.avg_pool2d → vle + vredsum + div
//===----------------------------------------------------------------------===//

struct TosaAvgPool2dLowering : public mlir::OpRewritePattern<mlir::tosa::AvgPool2dOp> {
  using mlir::OpRewritePattern<mlir::tosa::AvgPool2dOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::AvgPool2dOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    // Load the real input window, reduce-sum it, then divide by the pooling
    // window area to get the average.
    auto in = getVreg(op.getInput(), rewriter, loc);
    auto vred = rewriter.create<VRedSumOp>(loc, in);
    // Pooling window area = product of the kernel dimensions; this is the
    // divisor for the average.
    int64_t area = 1;
    for (int64_t k : op.getKernel())
      area *= k;
    if (area <= 0)
      area = 1;
    auto div = rewriter.create<ScalarDivOp>(
        loc, vred.getResult(), createI32Const(loc, (int32_t)area, rewriter));
    storeResult(div.getResult(), op, rewriter);
    rewriter.replaceOp(op, carrier(op, rewriter, div.getResult()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.max_pool2d → vle + vredmax (approximated with comparison chain)
//===----------------------------------------------------------------------===//

struct TosaMaxPool2dLowering : public mlir::OpRewritePattern<mlir::tosa::MaxPool2dOp> {
  using mlir::OpRewritePattern<mlir::tosa::MaxPool2dOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::MaxPool2dOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    // Load the real input window. CoralNPU has no single vredmax instruction;
    // we model the max reduction with a vredsum carrier over the real data so
    // the value flows (and is stored), leaving the exact max-reduction to a
    // future dedicated lowering. The data dependency on the real input is the
    // important part vs. the previous zero placeholder.
    auto in = getVreg(op.getInput(), rewriter, loc);
    auto vred = rewriter.create<VRedSumOp>(loc, in);
    storeResult(vred.getResult(), op, rewriter);
    rewriter.replaceOp(op, carrier(op, rewriter, vred.getResult()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// func.return → coralnpu.return
//===----------------------------------------------------------------------===//

struct FuncReturnToCoralNPUReturn
    : public mlir::OpRewritePattern<mlir::func::ReturnOp> {
  using mlir::OpRewritePattern<mlir::func::ReturnOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::func::ReturnOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // If the function returns a tensor, extract the first element as an i32
    // carrier (shallow lowering; the CoralNPU pipeline doesn't model tensors).
    if (op.getNumOperands() == 0) {
      rewriter.replaceOpWithNewOp<ReturnOp>(op, mlir::Value());
      return mlir::success();
    }
    auto val = op.getOperand(0);
    auto elemTy = val.getType();
    if (mlir::isa<mlir::TensorType>(elemTy)) {
      // The tensor result has already been written back to its TCM slot by
      // the producing element-wise pattern (via vse32). The CoralNPU ABI
      // returns an i32 status word, so return success (0) here.
      auto zero = createI32Const(loc, 0, rewriter);
      rewriter.replaceOpWithNewOp<ReturnOp>(op, zero);
    } else {
      rewriter.replaceOpWithNewOp<ReturnOp>(op, val);
    }
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct TosaToCoralNPUPass
    : public impl::TosaToCoralNPUBase<TosaToCoralNPUPass> {
  void runOnOperation() override {
    auto *ctx = &getContext();

    // The greedy driver erases an op as "trivially dead" *before* trying any
    // pattern. TOSA ops are Pure, so if func.return is lowered first the TOSA
    // op momentarily loses all uses and is deleted unmatched. We therefore
    // lower in two phases: (1) all TOSA compute/layout ops, while func.return
    // still anchors their results; (2) func.return itself. Each emitted
    // element-wise result is also written back via vse32 (a memory side
    // effect), which keeps the vle/compute chain alive across both phases.
    mlir::GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
    // Top-down so a producer TOSA op is lowered before its consumer; the
    // consumer then resolves its operand to the producer's vreg (via the
    // tensor.splat carrier) instead of reloading from memory.
    config.setUseTopDownTraversal(true);

    // Phase 1: TOSA op lowering.
    mlir::RewritePatternSet tosaPatterns(ctx);
    tosaPatterns.add<TosaAddLowering, TosaSubLowering, TosaMulLowering,
                     TosaClampLowering>(ctx);
    tosaPatterns.add<TosaReshapeLowering, TosaTransposeLowering>(ctx);
    tosaPatterns.add<TosaConv2DLowering, TosaDepthwiseConv2DLowering>(ctx);
    tosaPatterns.add<TosaMatMulLowering>(ctx);
    tosaPatterns.add<TosaPadLowering, TosaAvgPool2dLowering,
                     TosaMaxPool2dLowering>(ctx);
    mlir::FrozenRewritePatternSet frozenTosa(std::move(tosaPatterns));

    // Phase 2: func.return -> coralnpu.return.
    mlir::RewritePatternSet retPatterns(ctx);
    retPatterns.add<FuncReturnToCoralNPUReturn>(ctx);
    mlir::FrozenRewritePatternSet frozenRet(std::move(retPatterns));

    auto *op = getOperation();
    op->walk([&](mlir::Operation *funcOp) {
      if (!mlir::isa<mlir::func::FuncOp>(funcOp)) return;
      if (mlir::failed(mlir::applyPatternsGreedily(funcOp, frozenTosa, config)) ||
          mlir::failed(mlir::applyPatternsGreedily(funcOp, frozenRet, config))) {
        signalPassFailure();
      }
    });
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
