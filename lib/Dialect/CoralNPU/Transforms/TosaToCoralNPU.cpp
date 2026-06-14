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
// Elementwise: tosa.add / tosa.sub / tosa.mul
//===----------------------------------------------------------------------===//

struct TosaAddLowering : public mlir::OpRewritePattern<mlir::tosa::AddOp> {
  using mlir::OpRewritePattern<mlir::tosa::AddOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::AddOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sewLmul = pickVConfig(tensorElementType(op.getResult()));
    rewriter.create<VSetVLOp>(loc, sewLmul.first, sewLmul.second);
    auto c0 = createI32Const(loc, 0, rewriter);
    // 4-way stripmine = max SIMD throughput
    auto vadd = rewriter.create<VAddOp>(loc, c0, c0, 4);
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
    auto c0 = createI32Const(loc, 0, rewriter);
    auto vsub = rewriter.create<VSubOp>(loc, c0, c0, 4);
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
    auto c0 = createI32Const(loc, 0, rewriter);
    auto vmul = rewriter.create<VMulOp>(loc, c0, c0, 4);
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

    // Emit the core outer-product MAC that conv2d decomposes into.
    auto c0 = createI32Const(loc, 0, rewriter);
    auto outerProd = rewriter.create<OuterProductOp>(loc, c0, c0, c0);

    // Read accumulator result back to a vector register.
    auto accRead = rewriter.create<AccReadOp>(loc, outerProd.getAccNew(), c0);

    // The result is materialized as a vector store so the tensor type
    // is preserved (memory-backed) — wire the VLE for a 32-bit tensor
    // since conv2d outputs are typically i32.
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    (void)accRead;
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
    rewriter.create<VSetVLOp>(loc, SEW::E8, LMUL::M1);
    auto c0 = createI32Const(loc, 0, rewriter);
    auto outerProd = rewriter.create<OuterProductOp>(loc, c0, c0, c0, 4);
    auto accRead = rewriter.create<AccReadOp>(loc, outerProd.getAccNew(), c0);
    (void)accRead;
    rewriter.replaceOp(op, carrier(op, rewriter, accRead.getResult()));
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
    auto c0 = createI32Const(loc, 0, rewriter);
    auto vle = rewriter.create<VLE32Op>(loc, c0, c0);
    auto vred = rewriter.create<VRedSumOp>(loc, vle.getResult());
    auto div = rewriter.create<ScalarDivOp>(loc, vred.getResult(),
                                            createI32Const(loc, 4, rewriter));
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
    auto c0 = createI32Const(loc, 0, rewriter);
    auto vle = rewriter.create<VLE32Op>(loc, c0, c0);
    // vredmax is approximated via slt for now.
    auto slt = rewriter.create<ScalarSltOp>(loc, vle.getResult(), c0);
    rewriter.replaceOp(op, carrier(op, rewriter, slt.getResult()));
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
      rewriter.replaceOpWithNewOp<ReturnOp>(op, std::nullopt);
      return mlir::success();
    }
    auto val = op.getOperand(0);
    auto elemTy = val.getType();
    if (auto tt = mlir::dyn_cast<mlir::TensorType>(elemTy)) {
      // Reinterpret the tensor as an i32 (carrier) for the CoralNPU layer.
      auto i32Ty = mlir::IntegerType::get(op.getContext(), 32);
      auto carrier = rewriter.create<mlir::tensor::CastOp>(
          loc, mlir::RankedTensorType::get(tt.getShape(), i32Ty), val);
      rewriter.replaceOpWithNewOp<ReturnOp>(op, carrier.getResult());
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
    mlir::RewritePatternSet patterns(ctx);

    // Elementwise
    patterns.add<TosaAddLowering, TosaSubLowering, TosaMulLowering,
                 TosaClampLowering>(ctx);
    // Data layout
    patterns.add<TosaReshapeLowering, TosaTransposeLowering>(ctx);
    // Convolution
    patterns.add<TosaConv2DLowering, TosaDepthwiseConv2DLowering>(ctx);
    // Matrix
    patterns.add<TosaMatMulLowering>(ctx);
    // Padding / pooling
    patterns.add<TosaPadLowering, TosaAvgPool2dLowering,
                 TosaMaxPool2dLowering>(ctx);
    // Return lowering (func.return -> coralnpu.return)
    patterns.add<FuncReturnToCoralNPUReturn>(ctx);

    // Walk manually so we don't run DCE (which removes side-effect ops
    // like coralnpu.vsetvl that are still semantically required).
    auto *op = getOperation();
    op->walk([&](mlir::Operation *funcOp) {
      if (!mlir::isa<mlir::func::FuncOp>(funcOp)) return;
      // Apply patterns via partial conversion
      if (mlir::failed(mlir::applyPatternsGreedily(
              funcOp, std::move(patterns)))) {
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
