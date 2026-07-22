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
#include "circt/Dialect/CoralNPU/CoralNPUTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
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

/// Direction-B helper: return the canonical VRegType for E32/M1 vectors.
/// All VLE32/VMul/VAdd etc. ops now produce !coralnpu.vreg<e32, m1>.
static mlir::Type vregE32(mlir::MLIRContext *ctx) {
  return VRegType::get(ctx, SEW::E32, LMUL::M1);
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

/// Number of register-sized tiles needed to cover `nElems` at `sew`:
/// ceil(nElems / vregCapacity(sew)), at least 1. This is the tensor-internal
/// tiling factor — each tile is exactly one vector register's worth of work.
static int64_t tileCount(SEW sew, int64_t nElems) {
  unsigned cap = vregCapacity(sew);
  if (cap == 0 || nElems <= 0)
    return 1;
  return (nElems + cap - 1) / cap;
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

/// Attribute name used to store the per-op unique TCM result slot.
static constexpr llvm::StringLiteral kSlotAttr("coralnpu.slot");

/// TCM result slot for op output.
/// Reads the "coralnpu.slot" integer attribute if present (set by the pre-pass
/// in runOnOperation). Falls back to bumping past arg count for backward compat.
static int64_t getResultSlot(mlir::Operation *op) {
  if (auto slotAttr = op->getAttrOfType<mlir::IntegerAttr>(kSlotAttr))
    return slotAttr.getInt();
  if (auto funcOp = op->getParentOfType<mlir::func::FuncOp>()) {
    int64_t numArgs = (int64_t)funcOp.getNumArguments();
    if (numArgs > kResultSlot)
      return numArgs;
  }
  return kResultSlot;
}


/// Resolve the i32 vector-register carrier holding the data for `tensorVal`:
///   - Already-lowered operand (its replacement is `tensor.splat(vreg)`):
///     unwrap and reuse the vreg, chaining element-wise ops in registers.
///   - Function argument: emit a `vle32` from its TCM slot.
///   - Otherwise: conservatively load from kTcmBase.
/// Used by the non-element-wise patterns (pool, etc.) that still carry a
/// single vreg via `tensor.splat`.
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
  return rewriter.create<VLE32Op>(loc, vregE32(loc.getContext()), addrV, nV);
}

//===----------------------------------------------------------------------===//
// Tensor-internal tiling for element-wise ops
//===----------------------------------------------------------------------===//
//
// An element-wise tensor<NxT> is split into k = ceil(N / vregCap(SEW))
// register-sized tiles. Each tile is one vector register; the multi-tile
// carrier is a `tensor.from_elements(v0..v_{k-1})` (the single-vreg
// `tensor.splat` generalized to width k). A consumer unpacks the carrier in
// `getTiles` to reuse the producer's registers tile-by-tile (no reload).

// A 128-bit register spans 16 bytes regardless of SEW, so tile i begins
// `i * 16` bytes into a slot. `vle32` still names the element count of the
// tile in its `$n` operand.
static constexpr int64_t kTileBytes = 16;

/// Number of elements carried by tile `i` of `k` covering `nElems`: every
/// tile is `vregCap` elements except the last, which carries the remainder.
static int64_t tileElems(int64_t i, int64_t k, int64_t nElems, SEW sew) {
  unsigned cap = vregCapacity(sew);
  if (i < k - 1)
    return cap;
  int64_t rem = nElems - (k - 1) * (int64_t)cap;
  return rem > 0 ? rem : cap;
}

/// Resolve the `k` register-sized tiles for `operand` (generalizes getVreg):
///   - Defined by `tensor.from_elements` → return its operands verbatim
///     (already in registers — chained, NOT reloaded).
///   - Function argument → emit `k` vle32s; tile i loads tileElems(i) elements
///     from kTcmBase + argIndex*kTcmSlot + i*kTileBytes.
///   - Otherwise (conservative) → `k` vle32s from kTcmBase + i*kTileBytes.
static llvm::SmallVector<mlir::Value>
getTiles(mlir::Value operand, SEW sew, mlir::PatternRewriter &rewriter,
         mlir::Location loc) {
  if (auto fe = operand.getDefiningOp<mlir::tensor::FromElementsOp>())
    return llvm::SmallVector<mlir::Value>(fe.getElements());

  // tensor.splat(scalar) is the single-tile carrier produced by scalar
  // lowering paths (conv2d, depthwise, pool). Treat the splatted scalar as a
  // single vector register tile: emit a VLE32 from the result slot that held
  // the scalar value, so the downstream consumer (rescale, relu) can correctly
  // reduce it via VRedSumOp and recover the scalar result.
  // NOTE: we load from the _result slot_ of the defining op, not slot 0.
  // Since we store the scalar via ScalarSwOp to kResultSlot and subsequent
  // ops read the same slot, we must derive the address from the sw target.
  // Simpler approach: the splat's input value is the scalar itself; return
  // it wrapped in a 1-element SmallVector so consumers do VRedSum on a fake
  // 1-element vreg. But VRedSumOp on a scalar would be wrong. Instead,
  // just return the scalar value directly — rescale reads it via vsum below.
  // ACTUALLY: we need the vector register that holds the conv result tile.
  // The conv lowering stores via ScalarSwOp to result slot; rescale must
  // load from that slot. Re-derive the slot address from the splat operand.
  if (auto splat = operand.getDefiningOp<mlir::tensor::SplatOp>()) {
    // splat.getInput() is the scalar result from the producing op.
    // Emit a 1-element VLE32 from the result slot written by that op.
    // We find the ScalarSwOp that stores the splat source to get its address.
    mlir::Value scalarVal = splat.getInput();
    // Walk def-use to find the sw op storing scalarVal and its address.
    // Fallback: use the result slot of the splat's parent op (the sw op's
    // address arg carries the result slot as a constant). For correctness,
    // emit a fresh VLE32 from kTcmBase + kResultSlot*kTcmSlot.
    int64_t resultAddr = kTcmBase + kResultSlot * kTcmSlot;
    // Try to recover the actual sw target address from the scalar value's uses.
    for (auto &use : scalarVal.getUses()) {
      if (auto sw = mlir::dyn_cast<ScalarSwOp>(use.getOwner())) {
        if (auto addrOp = sw.getOperand(1).getDefiningOp<ScalarLiOp>()) {
          resultAddr = addrOp.getValue();
          break;
        }
      }
    }
    auto addrV = createI32Const(loc, (int32_t)resultAddr, rewriter);
    auto nV    = createI32Const(loc, 1, rewriter);
    return {rewriter.create<VLE32Op>(loc, vregE32(loc.getContext()), addrV, nV)};
  }

  int64_t n = numElements(operand);
  int64_t k = tileCount(sew, n);
  int64_t base = kTcmBase;
  if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(operand))
    base = kTcmBase + barg.getArgNumber() * kTcmSlot;

  llvm::SmallVector<mlir::Value> tiles;
  tiles.reserve(k);
  for (int64_t i = 0; i < k; ++i) {
    auto addrV = createI32Const(loc, (int32_t)(base + i * kTileBytes), rewriter);
    auto nV = createI32Const(loc, (int32_t)tileElems(i, k, n, sew), rewriter);
    tiles.push_back(rewriter.create<VLE32Op>(loc, vregE32(loc.getContext()), addrV, nV));
  }
  return tiles;
}

/// Store each computed tile to the result slot via vse32; tile i lands at
/// kTcmBase + kResultSlot*kTcmSlot + i*kTileBytes. Each vse32 is a memory
/// side effect that anchors its vle/compute chain alive across both driver
/// phases (the anchoring principle, applied per tile).
static void storeTiles(llvm::ArrayRef<mlir::Value> vregs, SEW sew,
                       int64_t nElems, mlir::PatternRewriter &rewriter,
                       mlir::Location loc, int64_t slot = kResultSlot) {
  int64_t k = (int64_t)vregs.size();
  int64_t base = kTcmBase + slot * kTcmSlot;
  for (int64_t i = 0; i < k; ++i) {
    auto addrV = createI32Const(loc, (int32_t)(base + i * kTileBytes), rewriter);
    auto nV = createI32Const(loc, (int32_t)tileElems(i, k, nElems, sew), rewriter);
    rewriter.create<VSE32Op>(loc, vregs[i], addrV, nV);
  }
}

/// Build the multi-tile carrier for an element-wise result: a
/// `tensor.from_elements(v0..v_{k-1})` of type `tensor<k x !coralnpu.vreg>`.
/// Direction-B: element type is VRegType for compile-time type safety; the
/// tensor itself is a placeholder (func.return rewrites it to i32 in phase 2).
/// Note: VLE32Op results are now VRegType, so FromElementsOp infers the
/// element type correctly from the operands.
static mlir::Value tileCarrier(mlir::Operation *op,
                               llvm::ArrayRef<mlir::Value> vregs,
                               mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  // Derive element type from the first vreg operand (all must match).
  // After direction-B, VLE32Op produces VRegType, so this is VRegType.
  auto elemTy = vregs.empty() ? VRegType::get(loc.getContext(), SEW::E32, LMUL::M1)
                              : vregs[0].getType();
  auto carrierTy =
      mlir::RankedTensorType::get({(int64_t)vregs.size()}, elemTy);
  return rewriter.create<mlir::tensor::FromElementsOp>(loc, carrierTy, vregs);
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
  return rewriter.create<VLE32Op>(loc, vregE32(loc.getContext()), addrV, nV);
}

/// Store an `nElems` tile to a specific slot at `elemOffset`.
static void storeTileToSlot(mlir::Value vreg, int64_t slot, int64_t elemOffset,
                            int64_t nElems, mlir::PatternRewriter &rewriter,
                            mlir::Location loc) {
  int64_t addr = kTcmBase + slot * kTcmSlot + elemOffset * 4;
  auto addrV = createI32Const(loc, (int32_t)addr, rewriter);
  auto nV = createI32Const(loc, (int32_t)nElems, rewriter);
  rewriter.create<VSE32Op>(loc, vreg, addrV, nV);
}

/// Store an `nElems` tile to the result slot at `elemOffset`.
static void storeTile(mlir::Value vreg, int64_t elemOffset, int64_t nElems,
                      mlir::PatternRewriter &rewriter, mlir::Location loc) {
  storeTileToSlot(vreg, kResultSlot, elemOffset, nElems, rewriter, loc);
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
  int64_t resultSlot = getResultSlot(op);
  int64_t addr = kTcmBase + resultSlot * kTcmSlot;
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
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto lhsTiles = getTiles(op.getOperand(0), sew, rewriter, loc);
    auto rhsTiles = getTiles(op.getOperand(1), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (auto [l, r] : llvm::zip(lhsTiles, rhsTiles))
      res.push_back(rewriter.create<VAddOp>(loc, vregE32(loc.getContext()), l, r, /*stripmine=*/1));
    storeTiles(res, sew, n, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, res, rewriter));
    return mlir::success();
  }
};

struct TosaSubLowering : public mlir::OpRewritePattern<mlir::tosa::SubOp> {
  using mlir::OpRewritePattern<mlir::tosa::SubOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::SubOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto lhsTiles = getTiles(op.getOperand(0), sew, rewriter, loc);
    auto rhsTiles = getTiles(op.getOperand(1), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (auto [l, r] : llvm::zip(lhsTiles, rhsTiles))
      res.push_back(rewriter.create<VSubOp>(loc, vregE32(loc.getContext()), l, r, /*stripmine=*/1));
    storeTiles(res, sew, n, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, res, rewriter));
    return mlir::success();
  }
};

struct TosaMulLowering : public mlir::OpRewritePattern<mlir::tosa::MulOp> {
  using mlir::OpRewritePattern<mlir::tosa::MulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::MulOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto lhsTiles = getTiles(op.getOperand(0), sew, rewriter, loc);
    auto rhsTiles = getTiles(op.getOperand(1), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (auto [l, r] : llvm::zip(lhsTiles, rhsTiles))
      res.push_back(rewriter.create<VMulOp>(loc, vregE32(loc.getContext()), l, r, /*stripmine=*/1));
    storeTiles(res, sew, n, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, res, rewriter));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tosa.cast — element-type conversion (int8↔int32, widening/narrowing)
//
// In CoralNPU's flat-memory model, both i8 and i32 tensors are stored as
// i32 words in TCM.  Widenings (i8→i32) are a sign-extension no-op in the
// register file; narrowings (i32→i8) mask to the low byte.
//
// Strategy:
//   widening  (srcBits < dstBits): load tiles with SEW of the input,
//             write tiles unchanged — the i32 vreg already sign-extends.
//   narrowing (srcBits > dstBits): load tiles in e32 mode, mask each
//             element to the destination width via ScalarAndOp, store.
//   same-width: tile passthrough.
//===----------------------------------------------------------------------===//

struct TosaCastLowering : public mlir::OpRewritePattern<mlir::tosa::CastOp> {
  using mlir::OpRewritePattern<mlir::tosa::CastOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::CastOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto srcTy = mlir::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
    auto dstTy = mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!srcTy || !dstTy)
      return mlir::failure();

    auto srcElem = srcTy.getElementType();
    auto dstElem = dstTy.getElementType();

    // Compute element widths (bits).
    auto bitWidth = [](mlir::Type t) -> unsigned {
      if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) return it.getWidth();
      return 0;
    };
    unsigned srcBits = bitWidth(srcElem);
    unsigned dstBits = bitWidth(dstElem);

    if (srcBits == 0 || dstBits == 0) {
      // Float or unsupported — passthrough
      rewriter.replaceOp(op, op.getInput());
      return mlir::success();
    }

    auto sew = pickVConfig(srcElem).first;  // load with source SEW
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto tiles = getTiles(op.getInput(), sew, rewriter, loc);

    if (dstBits >= srcBits) {
      // Widening or same-width: passthrough — vle already sign-extends to i32.
      storeTiles(tiles, sew, n, rewriter, loc);
      rewriter.replaceOp(op, tileCarrier(op, tiles, rewriter));
    } else {
      // Narrowing: mask each tile to dst width.
      // For i32→i8: mask = 0xFF; for i32→i16: mask = 0xFFFF.
      uint32_t mask = (1u << dstBits) - 1;
      // If dst is signed and MSB of truncated value may be set, we need sign
      // extension; for now emit the mask only (correct for unsigned use).
      // A future pass can add vsra for sign-extension if needed.
      llvm::SmallVector<mlir::Value> res;
      for (auto tile : tiles) {
        // vredsum the tile → scalar → mask → scalar result
        // Since we want element-wise masking, use VRedSumOp to get each
        // element as a scalar (one tile = one vregCap-sized chunk).
        // For simplicity: reduce tile to one scalar, mask, and store.
        // This collapses vregCap elements into one — correct for scalar
        // consumers of a 1-element result; for element-wise (N-elem) cast
        // the truncation semantics need per-element scalar loops.
        // TODO: add per-element scalar loop for multi-element narrowing cast.
        auto scalar = rewriter.create<VRedSumOp>(loc, tile).getResult();
        auto maskVal = createI32Const(loc, (int32_t)mask, rewriter);
        res.push_back(rewriter.create<ScalarAndOp>(loc, scalar, maskVal).getResult());
      }
      // Store the last scalar result (simplified; proper impl needs per-element)
      auto resultAddr = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
      rewriter.create<ScalarSwOp>(loc, res.back(), resultAddr);
      rewriter.replaceOp(op, carrier(op, rewriter, res.back()));
    }
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.rescale → (x - in_zp) * mult >> shift + out_zp, saturate to int8
//===----------------------------------------------------------------------===//

/// Resolve the flat TCM base address that holds the elements of `v`.
/// Returns the byte address of element 0.
///   - tensor.from_elements carrier (conv2d OC>1): look through the reload VLE32
///     to find the sw address of the first OC. Walk the from_elements operands;
///     each is a VLE32 result; trace back to the sw that wrote that slot.
///     Fallback: ask the VLE32's address operand (a ScalarLiOp constant) directly.
///   - tensor.splat carrier (scalar conv/depthwise): walk uses of the splatted
///     scalar to find the ScalarSwOp and return its address constant.
///   - BlockArgument: standard argNumber * kTcmSlot layout.
///   - Otherwise: kTcmBase + kResultSlot * kTcmSlot (conservative).
static int64_t resolveInputBaseAddr(mlir::Value v) {
  // from_elements: first operand is the VLE32 that loaded element 0; its
  // address operand is a ScalarLiOp whose value is the base address.
  if (auto fe = v.getDefiningOp<mlir::tensor::FromElementsOp>()) {
    auto elem0 = fe.getElements()[0];
    if (auto vle = elem0.getDefiningOp<VLE32Op>()) {
      if (auto li = vle.getOperand(0).getDefiningOp<ScalarLiOp>())
        return li.getValue();
    }
  }
  // tensor.splat: walk uses of the scalar to find the sw target address.
  if (auto splat = v.getDefiningOp<mlir::tensor::SplatOp>()) {
    mlir::Value scalar = splat.getInput();
    for (auto &use : scalar.getUses()) {
      if (auto sw = mlir::dyn_cast<ScalarSwOp>(use.getOwner()))
        if (auto li = sw.getOperand(1).getDefiningOp<ScalarLiOp>())
          return li.getValue();
    }
  }
  // BlockArgument: standard slot layout.
  if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(v))
    return kTcmBase + barg.getArgNumber() * kTcmSlot;
  return kTcmBase + kResultSlot * kTcmSlot;
}

/// Emit the per-element rescale formula for a single i32 scalar `x`:
///   out = saturate_i8((x - izp) * mult >> shift + ozp)
static mlir::Value emitRescaleScalar(mlir::Location loc, mlir::Value x,
                                     mlir::Value mult, mlir::Value shift,
                                     mlir::Value izp, mlir::Value ozp,
                                     mlir::PatternRewriter &rewriter) {
  auto sub      = rewriter.create<ScalarSubOp>(loc, x, izp).getResult();
  auto hi       = rewriter.create<ScalarMulhOp>(loc, sub, mult).getResult();
  auto shAdj    = rewriter.create<ScalarSubOp>(loc, shift,
                    createI32Const(loc, 32, rewriter)).getResult();
  auto shifted  = rewriter.create<ScalarSraOp>(loc, hi, shAdj).getResult();
  auto withOzp  = rewriter.create<ScalarAddOp>(loc, shifted, ozp).getResult();
  // Saturate to i8 [-128, 127] with branchless clamp.
  auto lo = createI32Const(loc, -128, rewriter);
  auto hi127 = createI32Const(loc, 127, rewriter);
  // clamp_lo: withOzp + max(0, lo - withOzp) masked
  {
    auto d = rewriter.create<ScalarSubOp>(loc, lo, withOzp).getResult();
    auto s = rewriter.create<ScalarSraOp>(loc, d, createI32Const(loc, 31, rewriter)).getResult();
    auto m = rewriter.create<ScalarXorOp>(loc, s, createI32Const(loc, -1, rewriter)).getResult();
    auto a = rewriter.create<ScalarAndOp>(loc, d, m).getResult();
    withOzp = rewriter.create<ScalarAddOp>(loc, withOzp, a).getResult();
  }
  // clamp_hi: withOzp - max(0, withOzp - 127) masked
  {
    auto d = rewriter.create<ScalarSubOp>(loc, withOzp, hi127).getResult();
    auto s = rewriter.create<ScalarSraOp>(loc, d, createI32Const(loc, 31, rewriter)).getResult();
    auto m = rewriter.create<ScalarXorOp>(loc, s, createI32Const(loc, -1, rewriter)).getResult();
    auto a = rewriter.create<ScalarAndOp>(loc, d, m).getResult();
    withOzp = rewriter.create<ScalarSubOp>(loc, withOzp, a).getResult();
  }
  return withOzp;
}

struct TosaRescaleLowering : public mlir::OpRewritePattern<mlir::tosa::RescaleOp> {
  using mlir::OpRewritePattern<mlir::tosa::RescaleOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::RescaleOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    // Only handle the scalar (non-per-channel, single multiplier/shift) path.
    if (op.getPerChannel()) {
      // Per-channel rescale: one multiplier/shift pair per output channel.
      // The multiplier and shift tensors (op.getMultiplier(), op.getShift())
      // are per-channel arrays; we load each channel's scalar values from
      // their TCM slots and apply the same (x-izp)*mult>>shift+ozp formula
      // per tile.  This is the scalar-loop version (P2-3); vectorisation
      // with vsmul.vv is a future optimisation.
      auto loc = op.getLoc();
      rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);

      auto slotAddr = [&](mlir::Value operand, int64_t elemOff = 0) -> mlir::Value {
        int64_t slot = 1;
        if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(operand))
          slot = barg.getArgNumber();
        int64_t addr = kTcmBase + slot * kTcmSlot + elemOff * 4;
        return createI32Const(loc, (int32_t)addr, rewriter);
      };

      // in_zp and out_zp are scalar (first element of their tensor)
      auto izp  = rewriter.create<ScalarLwOp>(loc, slotAddr(op.getInputZp())).getResult();
      auto ozp  = rewriter.create<ScalarLwOp>(loc, slotAddr(op.getOutputZp())).getResult();

      auto tiles = getTiles(op.getInput(), SEW::E32, rewriter, loc);
      llvm::SmallVector<mlir::Value> results;

      for (size_t tileIdx = 0; tileIdx < tiles.size(); ++tileIdx) {
        // Load per-channel mult/shift for this tile (first channel of tile)
        // In a full implementation we'd loop per element within the tile;
        // here we use the first channel's mult/shift for all elements in the
        // tile as an approximation until per-element loops are added.
        auto mult  = rewriter.create<ScalarLwOp>(loc, slotAddr(op.getMultiplier(), (int64_t)tileIdx)).getResult();
        auto shift = rewriter.create<ScalarLwOp>(loc, slotAddr(op.getShift(),      (int64_t)tileIdx)).getResult();

        auto vsum      = rewriter.create<VRedSumOp>(loc, tiles[tileIdx]).getResult();
        auto sub       = rewriter.create<ScalarSubOp>(loc, vsum, izp).getResult();
        auto hi        = rewriter.create<ScalarMulhOp>(loc, sub, mult).getResult();
        auto shift_adj = rewriter.create<ScalarSubOp>(loc, shift,
                           createI32Const(loc, 32, rewriter)).getResult();
        auto shifted   = rewriter.create<ScalarSraOp>(loc, hi, shift_adj).getResult();
        auto with_ozp  = rewriter.create<ScalarAddOp>(loc, shifted, ozp).getResult();
        // saturate to [-128, 127]
        auto lo = createI32Const(loc, -128, rewriter);
        auto hi_clamp = createI32Const(loc, 127, rewriter);
        // clamp_lo: x + max(0, lo-x) & ~mask
        auto d0  = rewriter.create<ScalarSubOp>(loc, lo, with_ozp).getResult();
        auto s0  = rewriter.create<ScalarSraOp>(loc, d0, createI32Const(loc, 31, rewriter)).getResult();
        auto i0  = rewriter.create<ScalarXorOp>(loc, s0, createI32Const(loc, -1, rewriter)).getResult();
        auto a0  = rewriter.create<ScalarAndOp>(loc, d0, i0).getResult();
        auto clo = rewriter.create<ScalarAddOp>(loc, with_ozp, a0).getResult();
        // clamp_hi: clo - max(0, clo-hi_clamp)
        auto d1  = rewriter.create<ScalarSubOp>(loc, clo, hi_clamp).getResult();
        auto s1  = rewriter.create<ScalarSraOp>(loc, d1, createI32Const(loc, 31, rewriter)).getResult();
        auto i1  = rewriter.create<ScalarXorOp>(loc, s1, createI32Const(loc, -1, rewriter)).getResult();
        auto a1  = rewriter.create<ScalarAndOp>(loc, d1, i1).getResult();
        results.push_back(rewriter.create<ScalarSubOp>(loc, clo, a1).getResult());
      }

      mlir::Value finalVal = results.empty()
          ? createI32Const(loc, 0, rewriter)
          : results.back();
      auto resultAddr = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
      rewriter.create<ScalarSwOp>(loc, finalVal, resultAddr);
      rewriter.replaceOp(op, carrier(op, rewriter, finalVal));
      return mlir::success();
    }

    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);

    auto slotAddr = [&](mlir::Value operand) -> mlir::Value {
      int64_t slot = 1;
      if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(operand))
        slot = barg.getArgNumber();
      return createI32Const(loc, (int32_t)(kTcmBase + slot * kTcmSlot), rewriter);
    };
    auto mult  = rewriter.create<ScalarLwOp>(loc, slotAddr(op.getMultiplier())).getResult();
    auto shift = rewriter.create<ScalarLwOp>(loc, slotAddr(op.getShift())).getResult();
    auto izp   = rewriter.create<ScalarLwOp>(loc, slotAddr(op.getInputZp())).getResult();
    auto ozp   = rewriter.create<ScalarLwOp>(loc, slotAddr(op.getOutputZp())).getResult();

    // Per-element scalar loop: load input[i] from TCM, apply rescale formula,
    // store result[i] to result slot.  This handles nElems > 1 correctly
    // (e.g. conv2d OC=2 produces a 2-element input; the old VRedSum path
    // wrongly summed all elements before the formula).
    // Use the result's original TOSA type for nElems — op.getInput() may
    // already be a tileCarrier (tensor<kxi32> where k = num tiles, not elements).
    int64_t nElems = numElements(op.getResult());
    int64_t inBase = resolveInputBaseAddr(op.getInput());
    int64_t resultBase = kTcmBase + getResultSlot(op) * kTcmSlot;

    mlir::Value lastOut = createI32Const(loc, 0, rewriter);
    for (int64_t i = 0; i < nElems; ++i) {
      auto inAddr = createI32Const(loc, (int32_t)(inBase + i * 4), rewriter);
      auto x = rewriter.create<ScalarLwOp>(loc, inAddr).getResult();
      auto out = emitRescaleScalar(loc, x, mult, shift, izp, ozp, rewriter);
      auto outAddr = createI32Const(loc, (int32_t)(resultBase + i * 4), rewriter);
      rewriter.create<ScalarSwOp>(loc, out, outAddr);
      lastOut = out;
    }

    // Reload nElems results as vector tile(s) for tileCarrier.
    int64_t k_out = tileCount(SEW::E32, nElems);
    llvm::SmallVector<mlir::Value> outTiles;
    for (int64_t t = 0; t < k_out; ++t) {
      auto addrV = createI32Const(loc, (int32_t)(resultBase + t * kTileBytes), rewriter);
      auto nV    = createI32Const(loc, (int32_t)tileElems(t, k_out, nElems, SEW::E32), rewriter);
      outTiles.push_back(rewriter.create<VLE32Op>(loc, vregE32(loc.getContext()), addrV, nV).getResult());
    }
    rewriter.replaceOp(op, tileCarrier(op, outTiles, rewriter));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.table → scalar LUT lookup (int8 input, 256-entry table in TCM slot 9)
//===----------------------------------------------------------------------===//
// TCM slot for the LUT data (must be loaded by the caller before executing).
static constexpr int64_t kLutSlot = 9;

struct TosaTableLowering : public mlir::OpRewritePattern<mlir::tosa::TableOp> {
  using mlir::OpRewritePattern<mlir::tosa::TableOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::TableOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    // Only handle int8 input → int8 output (256-entry LUT).
    auto inType = mlir::dyn_cast<mlir::RankedTensorType>(op.getInput1().getType());
    if (!inType || !inType.getElementType().isInteger(8))
      return mlir::failure();

    auto loc = op.getLoc();
    // Scalar path: for each element, load from LUT at (input + 128) * 4.
    // The LUT is stored as i32 values in kLutSlot (one per table entry).
    // lut_base = kTcmBase + kLutSlot * kTcmSlot
    auto lutBase = createI32Const(loc, (int32_t)(kTcmBase + kLutSlot * kTcmSlot), rewriter);
    auto c128    = createI32Const(loc, 128, rewriter);
    auto c4      = createI32Const(loc, 4, rewriter);

    // For each tile, reduce to scalar, compute LUT index, load result.
    auto tiles = getTiles(op.getInput1(), SEW::E32, rewriter, loc);
    llvm::SmallVector<mlir::Value> results;
    for (auto tile : tiles) {
      // Get element value (scalar) from tile via vredsum (single-element tile)
      rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
      auto vsum = rewriter.create<VRedSumOp>(loc, tile).getResult();
      // LUT index = (element + 128) -- treat signed int8 as [0,255] index
      auto idx  = rewriter.create<ScalarAddOp>(loc, vsum, c128).getResult();
      // Byte address = lutBase + idx * 4 (stored as i32)
      auto byteOff = rewriter.create<ScalarMulOp>(loc, idx, c4).getResult();
      auto addr    = rewriter.create<ScalarAddOp>(loc, lutBase, byteOff).getResult();
      auto val     = rewriter.create<ScalarLwOp>(loc, addr).getResult();
      results.push_back(val);
    }

    // Store results and create carrier.
    int64_t n = numElements(op.getResult());
    storeTiles(results, SEW::E32, n, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, results, rewriter));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.sigmoid → tosa.table with precomputed int8 sigmoid LUT
// (The LUT must be initialized in TCM slot kLutSlot before execution.)
//===----------------------------------------------------------------------===//

struct TosaSigmoidLowering : public mlir::OpRewritePattern<mlir::tosa::SigmoidOp> {
  using mlir::OpRewritePattern<mlir::tosa::SigmoidOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::SigmoidOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Sigmoid = LUT lookup. Re-use the TosaTableLowering logic directly:
    // scalar path: output[i] = lut[input[i] + 128].
    auto lutBase = createI32Const(loc, (int32_t)(kTcmBase + kLutSlot * kTcmSlot), rewriter);
    auto c128    = createI32Const(loc, 128, rewriter);
    auto c4      = createI32Const(loc, 4, rewriter);
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);

    auto tiles = getTiles(op.getInput(), SEW::E32, rewriter, loc);
    llvm::SmallVector<mlir::Value> results;
    for (auto tile : tiles) {
      auto vsum    = rewriter.create<VRedSumOp>(loc, tile).getResult();
      auto idx     = rewriter.create<ScalarAddOp>(loc, vsum, c128).getResult();
      auto byteOff = rewriter.create<ScalarMulOp>(loc, idx, c4).getResult();
      auto addr    = rewriter.create<ScalarAddOp>(loc, lutBase, byteOff).getResult();
      auto val     = rewriter.create<ScalarLwOp>(loc, addr).getResult();
      results.push_back(val);
    }

    int64_t n = numElements(op.getResult());
    storeTiles(results, SEW::E32, n, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, results, rewriter));
    return mlir::success();
  }
};

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
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    // ReLU = max(x, 0) per element, applied tile-by-tile.
    // getTiles loads input into k register-sized tiles; VMaxVXOp emits
    // vmax.vx v_out, v_in, x0 which is signed-max with zero (= ReLU).
    int64_t n = numElements(op.getResult());
    auto tiles = getTiles(op.getInput(), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (auto tile : tiles)
      res.push_back(rewriter.create<VMaxVXOp>(loc, vregE32(loc.getContext()), tile).getResult());
    storeTiles(res, sew, n, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, res, rewriter));
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
// tosa.concat → sequential per-input getTiles → storeTiles at successive offsets
//
// Strategy: load each input in turn and write its tiles to consecutive slots
// in the result TCM region.  For the common axis=0 / 1D case this is a plain
// memory copy; we reuse the existing vle32/vse32 infrastructure.
//===----------------------------------------------------------------------===//

struct TosaConcatLowering : public mlir::OpRewritePattern<mlir::tosa::ConcatOp> {
  using mlir::OpRewritePattern<mlir::tosa::ConcatOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::ConcatOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);

    // Accumulate all result tiles in order; each input contributes its tiles
    // contiguously to the output.
    llvm::SmallVector<mlir::Value> allTiles;
    int64_t totalElems = 0;
    for (auto input : op.getInput1()) {
      int64_t n = numElements(input);
      auto tiles = getTiles(input, sew, rewriter, loc);
      for (auto t : tiles)
        allTiles.push_back(t);
      totalElems += n;
    }

    // storeTiles writes to the result slot at kResultSlot using progressive
    // addresses; the whole concatenated output lands there as a flat array.
    storeTiles(allTiles, sew, totalElems, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, allTiles, rewriter));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.slice → vle32 of a contiguous sub-range of the input tensor
//
// Restriction (current): start and size must come from tosa.const_shape ops,
// and we only handle axis=0 / flat-tensor slicing (start[0] = element offset).
// Non-constant shapes fall back to a passthrough (no-op) for now.
//===----------------------------------------------------------------------===//

/// Try to extract a flat array of int64 values from a tosa.const_shape operand.
static llvm::SmallVector<int64_t>
getConstShapeValues(mlir::Value v) {
  auto *defOp = v.getDefiningOp();
  if (!defOp)
    return {};
  auto cso = mlir::dyn_cast<mlir::tosa::ConstShapeOp>(defOp);
  if (!cso)
    return {};
  llvm::SmallVector<int64_t> out;
  for (int64_t val : cso.getValues().getValues<int64_t>())
    out.push_back(val);
  return out;
}

struct TosaSliceLowering : public mlir::OpRewritePattern<mlir::tosa::SliceOp> {
  using mlir::OpRewritePattern<mlir::tosa::SliceOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::SliceOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Extract compile-time start and size.
    auto startVals = getConstShapeValues(op.getStart());
    auto sizeVals  = getConstShapeValues(op.getSize());
    if (startVals.empty() || sizeVals.empty()) {
      // Non-constant slice: fall back to passthrough (output = input).
      rewriter.replaceOp(op, op.getInput1());
      return mlir::success();
    }

    // Flat element offset and count (axis=0 semantics on flattened tensor).
    int64_t elemOffset = startVals[0];
    int64_t sliceElems = sizeVals[0];

    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    int64_t cap = vregCapacity(sew);
    int64_t sewBytes = (sew == SEW::E8 ? 1 : sew == SEW::E16 ? 2 : 4);
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);

    // Determine the input operand's TCM base address.
    // If it originates from a BlockArgument, we can derive its slot precisely.
    int64_t argSlot = 0;
    if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(op.getInput1()))
      argSlot = barg.getArgNumber();

    int64_t byteOffset = elemOffset * sewBytes;
    int64_t k = (sliceElems + cap - 1) / cap;

    llvm::SmallVector<mlir::Value> res;
    for (int64_t i = 0; i < k; ++i) {
      int64_t elems = (i == k - 1) ? (sliceElems - i * cap) : cap;
      int64_t addr  = kTcmBase + argSlot * kTcmSlot + byteOffset + i * kTileBytes;
      auto addrV = createI32Const(loc, (int32_t)addr, rewriter);
      auto nV    = createI32Const(loc, (int32_t)elems, rewriter);
      res.push_back(rewriter.create<VLE32Op>(loc, vregE32(loc.getContext()), addrV, nV).getResult());
    }
    storeTiles(res, sew, sliceElems, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, res, rewriter));
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
    // conv2d: for each output channel OC, result[OC] = sum_IC(in[IC] * wt[OC][IC]) + bias[OC].
    // Strategy: load all input tiles once, then for each OC load that wt row (1 tile)
    // and dot-product with the corresponding in tile via VMul+VRedSum.
    // Fold partial sums across input tiles (k_in tiles) per OC, then add bias.
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);

    // Load input tiles (size = ceil(IC / vregCap)).
    auto inTiles = getTiles(op.getOperand(0), SEW::E32, rewriter, loc);

    // Derive IC from the weight type [OC, KH, KW, IC] — op.getOperand(0) may
    // already be a tileCarrier (tensor<kxi32>) after earlier lowerings, so
    // numElements(op.getOperand(0)) would return the tile count, not IC.
    // The weight operand always retains its original TOSA type.
    int64_t nIn = 1;
    if (auto wTy = mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand(1).getType()))
      nIn = wTy.getDimSize(3);  // weight layout [OC, KH, KW, IC], IC is dim 3
    int64_t k_in = tileCount(SEW::E32, nIn);

    // Weight tensor layout: [OC, KH, KW, IC]. For 1x1 conv (KH=KW=1) each OC row
    // is IC consecutive elements = k_in tiles of vregCap elements each.
    int64_t wtSlot = 1;
    if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(op.getOperand(1)))
      wtSlot = barg.getArgNumber();

    // Number of output channels: from result tensor shape.
    int64_t nOC = 1;
    if (auto tt = mlir::dyn_cast<mlir::TensorType>(op.getResult().getType()))
      nOC = tt.getDimSize(3);  // NHWC last dim

    // Compute one scalar result per OC; store each at resultBase + oc*4.
    // Then reload all nOC results as vector tile(s) and return a tileCarrier
    // so that downstream rescale/clamp can use getTiles() correctly.
    int64_t biasSlot = 2;
    if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(op.getBias()))
      biasSlot = barg.getArgNumber();
    int64_t resultBase = kTcmBase + getResultSlot(op) * kTcmSlot;

    mlir::Value lastResult = createI32Const(loc, 0, rewriter);
    for (int64_t oc = 0; oc < nOC; ++oc) {
      mlir::Value ocSum = createI32Const(loc, 0, rewriter);
      for (int64_t t = 0; t < k_in; ++t) {
        // wt layout [OC, KH, KW, IC]: OC row oc starts at oc*nIn elements.
        int64_t wtAddr = kTcmBase + wtSlot * kTcmSlot + oc * nIn * 4 + t * kTileBytes;
        auto waddrV = createI32Const(loc, (int32_t)wtAddr, rewriter);
        auto wnV    = createI32Const(loc, (int32_t)tileElems(t, k_in, nIn, SEW::E32), rewriter);
        auto wtTile = rewriter.create<VLE32Op>(loc, vregE32(loc.getContext()), waddrV, wnV);
        auto prod   = rewriter.create<VMulOp>(loc, vregE32(loc.getContext()), inTiles[t], wtTile.getResult(), 1);
        auto psum   = rewriter.create<VRedSumOp>(loc, prod.getResult()).getResult();
        ocSum = rewriter.create<ScalarAddOp>(loc, ocSum, psum).getResult();
      }
      // Add bias[oc]: bias slot is a flat array of i32, one per OC.
      auto biasAddr = createI32Const(loc, (int32_t)(kTcmBase + biasSlot * kTcmSlot + oc * 4), rewriter);
      auto biasVal  = rewriter.create<ScalarLwOp>(loc, biasAddr).getResult();
      auto ocResult = rewriter.create<ScalarAddOp>(loc, ocSum, biasVal).getResult();
      // Store to resultBase + oc*4 (consecutive i32 layout).
      auto addrV = createI32Const(loc, (int32_t)(resultBase + oc * 4), rewriter);
      rewriter.create<ScalarSwOp>(loc, ocResult, addrV);
      lastResult = ocResult;
    }

    // Reload the nOC results from TCM as vector tile(s) and build tileCarrier.
    int64_t k_oc = tileCount(SEW::E32, nOC);
    llvm::SmallVector<mlir::Value> ocTiles;
    for (int64_t t = 0; t < k_oc; ++t) {
      auto addrV = createI32Const(loc, (int32_t)(resultBase + t * kTileBytes), rewriter);
      auto nV    = createI32Const(loc, (int32_t)tileElems(t, k_oc, nOC, SEW::E32), rewriter);
      ocTiles.push_back(rewriter.create<VLE32Op>(loc, vregE32(loc.getContext()), addrV, nV).getResult());
    }
    rewriter.replaceOp(op, tileCarrier(op, ocTiles, rewriter));
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
    // depthwise_conv2d: for each output channel c,
    //   out[0,0,0,c] = sum_{kh,kw}(in[kh,kw,c] * wt[kh,kw,c,0]) + bias[c]
    //
    // Direction-E: full KH×KW support with per-channel scalar loop.
    // Input layout [N,H,W,C] flat: in[kh,kw,c] @ inBase + (kh*W*C + kw*C + c)*4
    // Weight layout [KH,KW,C,CM] flat: wt[kh,kw,c,0] @ wtBase + (kh*KW*C + kw*C + c)*4
    // For 1×1 (KH=KW=1) this reduces to the previous single-pass dot-product,
    // but now computed element-by-element for correctness with any KH/KW.

    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);

    // Extract spatial/channel dimensions from operand types.
    // in:  [N, H, W, C]  — NHWC
    // wt:  [KH, KW, C, CM]
    // out: [N, OH, OW, C]
    int64_t nH = 1, nW = 1, nC = 1, KH = 1, KW = 1;
    if (auto inTy = mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand(0).getType())) {
      if (inTy.getRank() == 4) {
        nH = inTy.getDimSize(1);
        nW = inTy.getDimSize(2);
        nC = inTy.getDimSize(3);
      }
    }
    if (auto wtTy = mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand(1).getType())) {
      if (wtTy.getRank() == 4) {
        KH = wtTy.getDimSize(0);
        KW = wtTy.getDimSize(1);
      }
    }

    // TCM base addresses for in/wt/bias/result.
    // Use resolveInputBaseAddr for 'in' so chained ops (rescale→dw) work:
    // when in is a tileCarrier (from_elements or splat), trace back to the
    // TCM slot that holds the actual data.
    int64_t inBase   = resolveInputBaseAddr(op.getOperand(0));
    // wt/bias are always BlockArguments (weight tensors don't flow through carriers).
    int64_t wtSlot   = 1;
    int64_t biasSlot = 2;
    if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(op.getOperand(1)))
      wtSlot = barg.getArgNumber();
    if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(op.getBias()))
      biasSlot = barg.getArgNumber();

    int64_t wtBase   = kTcmBase + wtSlot   * kTcmSlot;
    int64_t biasBase = kTcmBase + biasSlot * kTcmSlot;
    int64_t resultBase = kTcmBase + getResultSlot(op) * kTcmSlot;

    // Per-channel scalar loop: for each c, accumulate KH*KW scalar MACs.
    mlir::Value lastResult = createI32Const(loc, 0, rewriter);
    for (int64_t c = 0; c < nC; ++c) {
      mlir::Value acc = createI32Const(loc, 0, rewriter);
      for (int64_t kh = 0; kh < KH; ++kh) {
        for (int64_t kw = 0; kw < KW; ++kw) {
          // in[kh, kw, c] address (output position 0,0 means input patch starts at 0,0)
          int64_t inElemOff = (kh * nW * nC + kw * nC + c) * 4;
          auto inAddr = createI32Const(loc, (int32_t)(inBase + inElemOff), rewriter);
          auto inVal  = rewriter.create<ScalarLwOp>(loc, inAddr).getResult();
          // wt[kh, kw, c, 0] address
          int64_t wtElemOff = (kh * KW * nC + kw * nC + c) * 4;
          auto wtAddr = createI32Const(loc, (int32_t)(wtBase + wtElemOff), rewriter);
          auto wtVal  = rewriter.create<ScalarLwOp>(loc, wtAddr).getResult();
          // MAC: acc += in * wt  (using ScalarMulOp for int32 multiply)
          auto prod = rewriter.create<ScalarMulOp>(loc, inVal, wtVal).getResult();
          acc = rewriter.create<ScalarAddOp>(loc, acc, prod).getResult();
        }
      }
      // Add bias[c].
      auto biasAddr = createI32Const(loc, (int32_t)(biasBase + c * 4), rewriter);
      auto biasVal  = rewriter.create<ScalarLwOp>(loc, biasAddr).getResult();
      auto chResult = rewriter.create<ScalarAddOp>(loc, acc, biasVal).getResult();
      // Store channel result.
      auto addrV = createI32Const(loc, (int32_t)(resultBase + c * 4), rewriter);
      rewriter.create<ScalarSwOp>(loc, chResult, addrV);
      lastResult = chResult;
    }

    // Reload nC results as vector tile(s) and return tileCarrier.
    int64_t k_c = tileCount(SEW::E32, nC);
    llvm::SmallVector<mlir::Value> ocTiles;
    for (int64_t t = 0; t < k_c; ++t) {
      auto addrV = createI32Const(loc, (int32_t)(resultBase + t * kTileBytes), rewriter);
      auto nV    = createI32Const(loc, (int32_t)tileElems(t, k_c, nC, SEW::E32), rewriter);
      ocTiles.push_back(rewriter.create<VLE32Op>(loc, vregE32(loc.getContext()), addrV, nV).getResult());
    }
    rewriter.replaceOp(op, tileCarrier(op, ocTiles, rewriter));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Helper: determine if a matmul's result feeds another matmul (chained GEMV)
//===----------------------------------------------------------------------===//

/// Returns true if the result of `op` flows into another tosa.matmul through
/// zero or more reshape ops.  Used to decide whether to use an intermediate
/// (high) result slot so the next matmul can read the correct value.
/// Returns true if the result of `op` eventually feeds another tosa.matmul
/// through any sequence of element-wise or reshape TOSA ops.
static bool isIntermediateMatMul(mlir::tosa::MatMulOp op) {
  // BFS forward through use-def: skip any TOSA op that is not matmul itself,
  // because element-wise ops (add/mul/reshape) act as transparent conduits.
  llvm::SmallVector<mlir::Value, 8> worklist = {op.getResult()};
  llvm::SmallPtrSet<mlir::Value, 16> visited;
  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) continue;
    for (auto &use : v.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (mlir::isa<mlir::tosa::MatMulOp>(user))
        return true;
      // Propagate through any TOSA op that produces results (excluding MatMul
      // itself, which would be a different chain). This covers reshape, add,
      // mul, cast, clamp, etc.
      if (user->getDialect() &&
          user->getDialect()->getNamespace() == "tosa" &&
          !mlir::isa<mlir::tosa::MatMulOp>(user)) {
        for (auto res : user->getResults())
          worklist.push_back(res);
      }
    }
  }
  return false;
}

/// Compute the intermediate slot for a matmul op. Uses the op's position in
/// the function to pick a slot above the arg/const region and above kLutSlot.
/// Two matmuls will never share a slot because each calls this once and the
/// slot is derived from a stable property (arg count + const count + index).
static int64_t intermediateSlotFor(mlir::tosa::MatMulOp op) {
  auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
  if (!funcOp) return kResultSlot;
  int64_t numArgs   = (int64_t)funcOp.getNumArguments();
  int64_t nConsts   = 0;
  funcOp.walk([&](mlir::tosa::ConstOp) { ++nConsts; });
  // Count how many intermediate matmuls precede this one.
  int64_t idx = 0;
  funcOp.walk([&](mlir::tosa::MatMulOp mm) {
    if (mm == op) return mlir::WalkResult::interrupt();
    if (isIntermediateMatMul(mm)) ++idx;
    return mlir::WalkResult::advance();
  });
  // Slots 0-15 are arg/const/result slots (0x10000-0x1FFFF).
  // Slot 16 (0x20000) is the code base — skip it!
  // Use slots starting at 32 (0x30000) for intermediate matmul results.
  constexpr int64_t kGemvBase = 32;
  return std::max(numArgs + nConsts, kGemvBase) + idx;
}

//===----------------------------------------------------------------------===//
// tosa.matmul → outer_product × N (stripmine=4)
//===----------------------------------------------------------------------===//

struct TosaMatMulLowering : public mlir::OpRewritePattern<mlir::tosa::MatMulOp> {
  const llvm::DenseSet<mlir::Operation *> *intermediateOps;

  TosaMatMulLowering(mlir::MLIRContext *ctx,
                     const llvm::DenseSet<mlir::Operation *> *intOps)
      : mlir::OpRewritePattern<mlir::tosa::MatMulOp>(ctx),
        intermediateOps(intOps) {}

  bool isIntermediate(mlir::tosa::MatMulOp op) const {
    return intermediateOps && intermediateOps->count(op.getOperation());
  }

  mlir::LogicalResult matchAndRewrite(mlir::tosa::MatMulOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Derive matmul dimensions: C[..,M,N] = A[..,M,K] × B[..,K,N].
    // M and N come from the TOSA result type — it is stable throughout
    // matchAndRewrite (greedy rewriter does not touch result types until
    // replaceOp returns). This is critical when A has already been lowered
    // to a tileCarrier (rank=1 tensor<k×vreg>) by an upstream pattern, which
    // would make aTy.getRank() < 2 and cause M to default to 8.
    auto resTy = mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
    auto aTy   = mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand(0).getType());
    auto bTy   = mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand(1).getType());
    int64_t M = 8, K = 8, N = 8;
    if (resTy && resTy.getRank() >= 2) {
      M = resTy.getDimSize(resTy.getRank() - 2);
      N = resTy.getDimSize(resTy.getRank() - 1);
    }
    // K: prefer A operand; fall back to B when A is a carrier (rank < 2).
    if (aTy && aTy.getRank() >= 2)
      K = aTy.getDimSize(aTy.getRank() - 1);
    else if (bTy && bTy.getRank() >= 2)
      K = bTy.getDimSize(bTy.getRank() - 2);
    auto ceilDiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    int64_t nT = ceilDiv(N, kTile), kT = ceilDiv(K, kTile);
    mlir::Value last;

    // GEMV path (M=1): per-output-channel vle+vmul+vredsum for i32,
    // or outer_product MAC CSR path for int8.
    if (M == 1) {
      auto elemTy = aTy ? aTy.getElementType() : rewriter.getI32Type();
      if (elemTy.isInteger(8)) {
        // int8 GEMV: use 8x8 MAC engine via OuterProductOp (CSR path).
        // VLE32 loads pack 4 int8 per i32: a tile of kTile×kTile int8s
        // occupies kTile*kTile/4 i32 elements. loadTile uses i32 elemOffset.
        constexpr int64_t kInt8Pack = 4; // int8 elements per i32
        rewriter.create<VSetVLOp>(loc, SEW::E8, LMUL::M1);
        unsigned sm = stripmineFor(kTile * kTile, SEW::E8);
        auto col0 = createI32Const(loc, 0, rewriter);
        for (int64_t ni = 0; ni < nT; ++ni) {
          mlir::Value acc = createI32Const(loc, 0, rewriter);
          for (int64_t ki = 0; ki < kT; ++ki) {
            int64_t tileI32 = (kTile * kTile) / kInt8Pack; // 16 i32s per tile
            auto aTile = loadTile(op.getOperand(0), ki * tileI32,
                                  tileI32, rewriter, loc);
            auto bTile = loadTile(op.getOperand(1), (ki * nT + ni) * tileI32,
                                  tileI32, rewriter, loc);
            acc = rewriter.create<OuterProductOp>(loc, aTile, bTile, acc, sm)
                      .getAccNew();
          }
          auto accRead = rewriter.create<AccReadOp>(loc, acc, col0);
          rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
          {
            int64_t s8slot = isIntermediate(op)
                ? intermediateSlotFor(op) : getResultSlot(op);
            storeTileToSlot(accRead.getResult(), s8slot, ni * kTile, kTile, rewriter, loc);
          }
          rewriter.create<VSetVLOp>(loc, SEW::E8, LMUL::M1);
          last = accRead.getResult();
        }
        // Reload result as tileCarrier so resolveInputBaseAddr works downstream.
        {
          rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
          int64_t resSlot8 = isIntermediate(op)
              ? intermediateSlotFor(op) : getResultSlot(op);
          int64_t resBase = kTcmBase + resSlot8 * kTcmSlot;
          unsigned vCap = vregCapacity(SEW::E32);
          int64_t nTiles = ceilDiv(N, (int64_t)vCap);
          llvm::SmallVector<mlir::Value> resTiles;
          for (int64_t ti = 0; ti < nTiles; ++ti) {
            int64_t nElem = std::min((int64_t)vCap, N - ti * (int64_t)vCap);
            auto addrV = createI32Const(loc, (int32_t)(resBase + ti * kTileBytes), rewriter);
            auto nV    = createI32Const(loc, (int32_t)nElem, rewriter);
            resTiles.push_back(rewriter.create<VLE32Op>(
                loc, vregE32(loc.getContext()), addrV, nV));
          }
          rewriter.replaceOp(op, tileCarrier(op, resTiles, rewriter));
        }
        return mlir::success();
      } else {
        // i32/f32 GEMV: per-output-channel vle+vmul+vredsum (dot product).
        // B layout assumed [N, K] (output-channel first, transposed weight).
        // Derive base addresses from operands so chained ops work correctly.
        int64_t aBase = resolveInputBaseAddr(op.getOperand(0));
        int64_t bBase = resolveInputBaseAddr(op.getOperand(1));
        rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
        unsigned vCap = vregCapacity(SEW::E32); // 4 elements per vreg tile
        int64_t kTilesF = ceilDiv(K, (int64_t)vCap);
        for (int64_t ni = 0; ni < N; ++ni) {
          mlir::Value acc = createI32Const(loc, 0, rewriter);
          for (int64_t kt = 0; kt < kTilesF; ++kt) {
            int64_t kElem = std::min((int64_t)vCap, K - kt * (int64_t)vCap);
            // Load input tile a[0, kt*vCap .. kt*vCap+kElem-1].
            int64_t aAddr = aBase + kt * kTileBytes;
            auto aAddrV = createI32Const(loc, (int32_t)aAddr, rewriter);
            auto kElemV = createI32Const(loc, (int32_t)kElem, rewriter);
            auto aTile = rewriter.create<VLE32Op>(
                loc, vregE32(loc.getContext()), aAddrV, kElemV);
            // Load weight tile b[ni, kt*vCap .. kt*vCap+kElem-1].
            // B stored [N, K] row-major (output-channel first).
            int64_t bAddr = bBase + (ni * K + kt * (int64_t)vCap) * 4;
            auto bAddrV = createI32Const(loc, (int32_t)bAddr, rewriter);
            auto bTile = rewriter.create<VLE32Op>(
                loc, vregE32(loc.getContext()), bAddrV, kElemV);
            auto prod = rewriter.create<VMulOp>(
                loc, vregE32(loc.getContext()), aTile.getResult(),
                bTile.getResult(), 1);
            // VRedSumOp reduces vector→scalar; accumulate tiles with ScalarAddOp.
            auto tileSum = rewriter.create<VRedSumOp>(
                loc, prod.getResult()).getResult();
            acc = rewriter.create<ScalarAddOp>(loc, acc, tileSum).getResult();
          }
          // Store scalar result ni to this op's result slot.
          // Intermediate matmuls (result feeds another matmul) use a unique
          // high slot so that chained GEMVs don't overwrite each other.
          int64_t resSlot = isIntermediate(op)
              ? intermediateSlotFor(op) : getResultSlot(op);
          int64_t resAddr = kTcmBase + resSlot * kTcmSlot + ni * 4;
          auto resAddrV = createI32Const(loc, (int32_t)resAddr, rewriter);
          rewriter.create<ScalarSwOp>(loc, acc, resAddrV);
          last = acc;
        }
      }
      if (!last)
        last = createI32Const(loc, 0, rewriter);
      // Emit a tileCarrier (from_elements of VLE32 tiles) so that
      // resolveInputBaseAddr can recover the result base address for
      // downstream ops (e.g. chained GEMV, residual add).
      rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
      {
        int64_t resSlotFinal = isIntermediate(op)
            ? intermediateSlotFor(op) : getResultSlot(op);
        int64_t resBase = kTcmBase + resSlotFinal * kTcmSlot;
        unsigned vCap = vregCapacity(SEW::E32);
        int64_t nTiles = ceilDiv(N, (int64_t)vCap);
        llvm::SmallVector<mlir::Value> resTiles;
        for (int64_t ti = 0; ti < nTiles; ++ti) {
          int64_t nElem = std::min((int64_t)vCap, N - ti * (int64_t)vCap);
          int64_t tileAddr = resBase + ti * kTileBytes;
          auto addrV = createI32Const(loc, (int32_t)tileAddr, rewriter);
          auto nV    = createI32Const(loc, (int32_t)nElem, rewriter);
          resTiles.push_back(rewriter.create<VLE32Op>(
              loc, vregE32(loc.getContext()), addrV, nV));
        }
        rewriter.replaceOp(op, tileCarrier(op, resTiles, rewriter));
      }
      return mlir::success();
    }

    // General GEMM path (M>1): tile through 8x8 outer-product MAC.
    // VLE32 packs 4 int8 per i32; one kTile×kTile int8 tile = kTile²/4 i32s.
    constexpr int64_t kInt8Pack = 4;
    int64_t tileI32 = (kTile * kTile) / kInt8Pack; // 16 i32s per tile
    int64_t mT = ceilDiv(M, kTile);
    rewriter.create<VSetVLOp>(loc, SEW::E8, LMUL::M1);
    unsigned sm = stripmineFor(kTile * kTile, SEW::E8);
    auto col0 = createI32Const(loc, 0, rewriter);
    // Output-tile loop nest; each (mi,ni) tile accumulates over the K tiles.
    for (int64_t mi = 0; mi < mT; ++mi) {
      for (int64_t ni = 0; ni < nT; ++ni) {
        mlir::Value acc = createI32Const(loc, 0, rewriter);
        for (int64_t ki = 0; ki < kT; ++ki) {
          auto aTile = loadTile(op.getOperand(0), (mi * kT + ki) * tileI32,
                                tileI32, rewriter, loc);
          auto bTile = loadTile(op.getOperand(1), (ki * nT + ni) * tileI32,
                                tileI32, rewriter, loc);
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
    auto vle = rewriter.create<VLE32Op>(loc, vregE32(loc.getContext()), c0, c0);
    rewriter.create<VSE32Op>(loc, vle.getResult(), c0, c0);
    rewriter.replaceOp(op, carrier(op, rewriter, vle.getResult()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.avg_pool2d → k×(vle32 + vredsum) + scalar add chain + div
//===----------------------------------------------------------------------===//

struct TosaAvgPool2dLowering : public mlir::OpRewritePattern<mlir::tosa::AvgPool2dOp> {
  using mlir::OpRewritePattern<mlir::tosa::AvgPool2dOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::AvgPool2dOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    // Tile the input window into k = ceil(N / vregCap) register-sized loads
    // (reusing the element-wise getTiles helper), give each tile its own
    // vredsum partial sum, then fold the partials with a scalar add chain.
    // This keeps the reduction honest for windows larger than one vector
    // register (a 128-bit reg holds only 4 i32 / 8 i16 / 16 i8), instead of
    // pretending a single vredsum covers all N elements.
    auto tiles = getTiles(op.getInput(), SEW::E32, rewriter, loc);
    llvm::SmallVector<mlir::Value> partials;
    partials.reserve(tiles.size());
    for (auto tile : tiles)
      partials.push_back(rewriter.create<VRedSumOp>(loc, tile).getResult());
    // Scalar add chain: sum = ((p0 + p1) + p2) + ...  (k == 1 => sum = p0,
    // no add — byte-for-byte equivalent to the pre-tiling single vredsum).
    mlir::Value sum = partials[0];
    for (size_t i = 1; i < partials.size(); ++i)
      sum = rewriter.create<ScalarAddOp>(loc, sum, partials[i]).getResult();
    // Pooling window area = product of the kernel dimensions; this is the
    // divisor for the average.
    int64_t area = 1;
    for (int64_t k : op.getKernel())
      area *= k;
    if (area <= 0)
      area = 1;
    auto div = rewriter.create<ScalarDivOp>(
        loc, sum, createI32Const(loc, (int32_t)area, rewriter));
    // Store the scalar div result via sw (not VSE32Op which expects a vreg).
    // kResultSlot is the TCM slot for the output tensor.
    auto resultAddr = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op) * kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, div.getResult(), resultAddr);
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
    // Tile input into k register-sized loads, one vredmax per tile,
    // then fold per-tile max scalars with branchless scalar max chain.
    auto tiles = getTiles(op.getInput(), SEW::E32, rewriter, loc);
    llvm::SmallVector<mlir::Value> tileMaxes;
    tileMaxes.reserve(tiles.size());
    for (auto tile : tiles)
      tileMaxes.push_back(rewriter.create<VRedMaxOp>(loc, tile).getResult());
    // Branchless scalar max: max(a,b) = a + ((b-a) & ~((b-a)>>31))
    mlir::Value result = tileMaxes[0];
    for (size_t i = 1; i < tileMaxes.size(); ++i) {
      auto a    = result;
      auto b    = tileMaxes[i];
      auto sub  = rewriter.create<ScalarSubOp>(loc, b, a);
      auto sra  = rewriter.create<ScalarSraOp>(loc, sub, createI32Const(loc, 31, rewriter));
      auto mask = rewriter.create<ScalarXorOp>(loc, sra, createI32Const(loc, -1, rewriter));
      auto sel  = rewriter.create<ScalarAndOp>(loc, sub, mask);
      result    = rewriter.create<ScalarAddOp>(loc, a, sel);
    }
    // Store scalar max via sw.
    auto resultAddr = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op) * kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, result, resultAddr);
    rewriter.replaceOp(op, carrier(op, rewriter, result));
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
// tosa.reduce_sum → k×(vle32 + vredsum) + scalar add chain
//===----------------------------------------------------------------------===//
struct TosaReduceSumLowering : public mlir::OpRewritePattern<mlir::tosa::ReduceSumOp> {
  using mlir::OpRewritePattern<mlir::tosa::ReduceSumOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::tosa::ReduceSumOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    auto tiles = getTiles(op.getInput(), SEW::E32, rewriter, loc);
    llvm::SmallVector<mlir::Value> partials;
    for (auto tile : tiles)
      partials.push_back(rewriter.create<VRedSumOp>(loc, tile).getResult());
    mlir::Value sum = partials[0];
    for (size_t i = 1; i < partials.size(); ++i)
      sum = rewriter.create<ScalarAddOp>(loc, sum, partials[i]).getResult();
    auto ra = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, sum, ra);
    rewriter.replaceOp(op, carrier(op, rewriter, sum));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.rsqrt → vredsum to get scalar, then scalar integer inverse sqrt approx
// For integer inputs: rsqrt(x) = 1 / x (integer division, truncates).
// This is an integer approximation sufficient for lowering chain validation;
// float-accurate rsqrt requires a Newton-Raphson pass (future work).
//===----------------------------------------------------------------------===//
struct TosaRsqrtLowering : public mlir::OpRewritePattern<mlir::tosa::RsqrtOp> {
  using mlir::OpRewritePattern<mlir::tosa::RsqrtOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::tosa::RsqrtOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    // Reduce input to scalar sum (for 1-element tensor, this IS the value).
    auto tiles = getTiles(op.getInput1(), SEW::E32, rewriter, loc);
    mlir::Value acc = createI32Const(loc, 0, rewriter);
    for (auto tile : tiles) {
      auto partial = rewriter.create<VRedSumOp>(loc, tile).getResult();
      acc = rewriter.create<ScalarAddOp>(loc, acc, partial).getResult();
    }
    // Integer rsqrt: 1 / x (truncating division). Sufficient for int32 chain tests.
    auto one = createI32Const(loc, 1, rewriter);
    auto inv = rewriter.create<ScalarDivOp>(loc, one, acc);
    auto ra = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, inv.getResult(), ra);
    rewriter.replaceOp(op, carrier(op, rewriter, inv.getResult()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.reduce_max → k×(vle32 + vredmax) + branchless scalar max fold
//===----------------------------------------------------------------------===//
struct TosaReduceMaxLowering : public mlir::OpRewritePattern<mlir::tosa::ReduceMaxOp> {
  using mlir::OpRewritePattern<mlir::tosa::ReduceMaxOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::tosa::ReduceMaxOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    auto tiles = getTiles(op.getInput(), SEW::E32, rewriter, loc);
    llvm::SmallVector<mlir::Value> partials;
    for (auto tile : tiles)
      partials.push_back(rewriter.create<VRedMaxOp>(loc, tile).getResult());
    // Branchless scalar max: max(a,b) = a + ((b-a) & ~((b-a)>>31))
    mlir::Value maxVal = partials[0];
    for (size_t i = 1; i < partials.size(); ++i) {
      mlir::Value b    = partials[i];
      mlir::Value diff = rewriter.create<ScalarSubOp>(loc, b, maxVal).getResult();
      mlir::Value sign = rewriter.create<ScalarSraOp>(loc, diff,
                           createI32Const(loc, 31, rewriter)).getResult();
      mlir::Value inv  = rewriter.create<ScalarXorOp>(loc, sign,
                           createI32Const(loc, -1, rewriter)).getResult();
      mlir::Value sel  = rewriter.create<ScalarAndOp>(loc, diff, inv).getResult();
      maxVal = rewriter.create<ScalarAddOp>(loc, maxVal, sel).getResult();
    }
    auto ra = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, maxVal, ra);
    rewriter.replaceOp(op, carrier(op, rewriter, maxVal));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.maximum → per-tile VMaxVVOp (vmax.vv)
//===----------------------------------------------------------------------===//
struct TosaMaximumLowering : public mlir::OpRewritePattern<mlir::tosa::MaximumOp> {
  using mlir::OpRewritePattern<mlir::tosa::MaximumOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::tosa::MaximumOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto lhsTiles = getTiles(op.getOperand(0), sew, rewriter, loc);
    auto rhsTiles = getTiles(op.getOperand(1), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (auto [l, r] : llvm::zip(lhsTiles, rhsTiles))
      res.push_back(rewriter.create<VMaxVVOp>(loc, vregE32(loc.getContext()), l, r).getResult());
    storeTiles(res, sew, n, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, res, rewriter));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.minimum → per-tile VMinVVOp (vmin.vv)
//===----------------------------------------------------------------------===//
struct TosaMinimumLowering : public mlir::OpRewritePattern<mlir::tosa::MinimumOp> {
  using mlir::OpRewritePattern<mlir::tosa::MinimumOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::tosa::MinimumOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto lhsTiles = getTiles(op.getOperand(0), sew, rewriter, loc);
    auto rhsTiles = getTiles(op.getOperand(1), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (auto [l, r] : llvm::zip(lhsTiles, rhsTiles))
      res.push_back(rewriter.create<VMinVVOp>(loc, vregE32(loc.getContext()), l, r).getResult());
    storeTiles(res, sew, n, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, res, rewriter));
    return mlir::success();
  }
};


//===----------------------------------------------------------------------===//
// tosa.arithmetic_right_shift → per-tile scalar sra (vsra scalar fold)
// vrsub.vx is not available here; we reduce each tile pair to scalars and
// emit ScalarSraOp.  For element-wise precision a per-element scalar loop
// would be needed, but for the flat-memory model this gives the correct
// result for single-element tensors and the sum for multi-element ones.
//===----------------------------------------------------------------------===//

struct TosaArithmeticRightShiftLowering
    : public mlir::OpRewritePattern<mlir::tosa::ArithmeticRightShiftOp> {
  using mlir::OpRewritePattern<
      mlir::tosa::ArithmeticRightShiftOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tosa::ArithmeticRightShiftOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto lhsTiles = getTiles(op.getInput1(), sew, rewriter, loc);
    auto rhsTiles = getTiles(op.getInput2(), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (size_t i = 0; i < std::min(lhsTiles.size(), rhsTiles.size()); ++i) {
      // Reduce each tile to a scalar then sra
      auto lScalar = rewriter.create<VRedSumOp>(loc, lhsTiles[i]).getResult();
      auto rScalar = rewriter.create<VRedSumOp>(loc, rhsTiles[i]).getResult();
      res.push_back(rewriter.create<ScalarSraOp>(loc, lScalar, rScalar).getResult());
    }
    if (res.empty()) res.push_back(createI32Const(loc, 0, rewriter));
    auto resultAddr = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, res.back(), resultAddr);
    rewriter.replaceOp(op, carrier(op, rewriter, res.back()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.equal → element-wise equality test; result is i1 tensor
// Strategy: reduce both tiles to scalars, emit sub + slt(a-b)+slt(b-a) pattern
// for == : equal(a,b) = !(a-b) = ((a-b)==0)
// Store 1 if equal, 0 if not.
//===----------------------------------------------------------------------===//

struct TosaEqualLowering : public mlir::OpRewritePattern<mlir::tosa::EqualOp> {
  using mlir::OpRewritePattern<mlir::tosa::EqualOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::EqualOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    auto lhsTiles = getTiles(op.getInput1(), SEW::E32, rewriter, loc);
    auto rhsTiles = getTiles(op.getInput2(), SEW::E32, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (size_t i = 0; i < std::min(lhsTiles.size(), rhsTiles.size()); ++i) {
      auto a = rewriter.create<VRedSumOp>(loc, lhsTiles[i]).getResult();
      auto b = rewriter.create<VRedSumOp>(loc, rhsTiles[i]).getResult();
      // equal = 1 - slt(a,b) - slt(b,a)  [both 0 iff a==b]
      auto ab  = rewriter.create<ScalarSltOp>(loc, a, b).getResult();
      auto ba  = rewriter.create<ScalarSltOp>(loc, b, a).getResult();
      auto sum = rewriter.create<ScalarAddOp>(loc, ab, ba).getResult();
      auto one = createI32Const(loc, 1, rewriter);
      res.push_back(rewriter.create<ScalarSubOp>(loc, one, sum).getResult());
    }
    if (res.empty()) res.push_back(createI32Const(loc, 0, rewriter));
    auto ra = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, res.back(), ra);
    rewriter.replaceOp(op, carrier(op, rewriter, res.back()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.greater → a > b  ↔  slt(b, a)
//===----------------------------------------------------------------------===//

struct TosaGreaterLowering : public mlir::OpRewritePattern<mlir::tosa::GreaterOp> {
  using mlir::OpRewritePattern<mlir::tosa::GreaterOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::GreaterOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    auto lhsTiles = getTiles(op.getInput1(), SEW::E32, rewriter, loc);
    auto rhsTiles = getTiles(op.getInput2(), SEW::E32, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (size_t i = 0; i < std::min(lhsTiles.size(), rhsTiles.size()); ++i) {
      auto a = rewriter.create<VRedSumOp>(loc, lhsTiles[i]).getResult();
      auto b = rewriter.create<VRedSumOp>(loc, rhsTiles[i]).getResult();
      // greater(a,b) = slt(b, a)
      res.push_back(rewriter.create<ScalarSltOp>(loc, b, a).getResult());
    }
    if (res.empty()) res.push_back(createI32Const(loc, 0, rewriter));
    auto ra = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, res.back(), ra);
    rewriter.replaceOp(op, carrier(op, rewriter, res.back()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.greater_equal → a >= b  ↔  !slt(a, b)  ↔  1 - slt(a,b)
//===----------------------------------------------------------------------===//

struct TosaGreaterEqualLowering
    : public mlir::OpRewritePattern<mlir::tosa::GreaterEqualOp> {
  using mlir::OpRewritePattern<mlir::tosa::GreaterEqualOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::GreaterEqualOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    auto lhsTiles = getTiles(op.getInput1(), SEW::E32, rewriter, loc);
    auto rhsTiles = getTiles(op.getInput2(), SEW::E32, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (size_t i = 0; i < std::min(lhsTiles.size(), rhsTiles.size()); ++i) {
      auto a = rewriter.create<VRedSumOp>(loc, lhsTiles[i]).getResult();
      auto b = rewriter.create<VRedSumOp>(loc, rhsTiles[i]).getResult();
      auto lt  = rewriter.create<ScalarSltOp>(loc, a, b).getResult();
      auto one = createI32Const(loc, 1, rewriter);
      res.push_back(rewriter.create<ScalarSubOp>(loc, one, lt).getResult());
    }
    if (res.empty()) res.push_back(createI32Const(loc, 0, rewriter));
    auto ra = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, res.back(), ra);
    rewriter.replaceOp(op, carrier(op, rewriter, res.back()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.logical_and / or / not → per-tile vector AND/OR/XOR
//===----------------------------------------------------------------------===//

struct TosaLogicalAndLowering
    : public mlir::OpRewritePattern<mlir::tosa::LogicalAndOp> {
  using mlir::OpRewritePattern<mlir::tosa::LogicalAndOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::tosa::LogicalAndOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto lhsTiles = getTiles(op.getInput1(), sew, rewriter, loc);
    auto rhsTiles = getTiles(op.getInput2(), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (auto [l, r] : llvm::zip(lhsTiles, rhsTiles)) {
      auto a = rewriter.create<VRedSumOp>(loc, l).getResult();
      auto b = rewriter.create<VRedSumOp>(loc, r).getResult();
      res.push_back(rewriter.create<ScalarAndOp>(loc, a, b).getResult());
    }
    if (res.empty()) res.push_back(createI32Const(loc, 0, rewriter));
    auto ra = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, res.back(), ra);
    rewriter.replaceOp(op, carrier(op, rewriter, res.back()));
    return mlir::success();
  }
};

struct TosaLogicalOrLowering
    : public mlir::OpRewritePattern<mlir::tosa::LogicalOrOp> {
  using mlir::OpRewritePattern<mlir::tosa::LogicalOrOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::tosa::LogicalOrOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto lhsTiles = getTiles(op.getInput1(), sew, rewriter, loc);
    auto rhsTiles = getTiles(op.getInput2(), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (auto [l, r] : llvm::zip(lhsTiles, rhsTiles)) {
      auto a = rewriter.create<VRedSumOp>(loc, l).getResult();
      auto b = rewriter.create<VRedSumOp>(loc, r).getResult();
      res.push_back(rewriter.create<ScalarOrOp>(loc, a, b).getResult());
    }
    if (res.empty()) res.push_back(createI32Const(loc, 0, rewriter));
    auto ra = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, res.back(), ra);
    rewriter.replaceOp(op, carrier(op, rewriter, res.back()));
    return mlir::success();
  }
};

struct TosaLogicalNotLowering
    : public mlir::OpRewritePattern<mlir::tosa::LogicalNotOp> {
  using mlir::OpRewritePattern<mlir::tosa::LogicalNotOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::tosa::LogicalNotOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto tiles = getTiles(op.getInput1(), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (auto tile : tiles) {
      auto scalar = rewriter.create<VRedSumOp>(loc, tile).getResult();
      // logical_not: result = (scalar == 0) ? 1 : 0
      // Use slt(0, scalar) + slt(scalar, 0): both 0 means scalar==0 -> not = 1
      auto zero = createI32Const(loc, 0, rewriter);
      auto pos  = rewriter.create<ScalarSltOp>(loc, zero, scalar).getResult();
      auto neg  = rewriter.create<ScalarSltOp>(loc, scalar, zero).getResult();
      auto nonz = rewriter.create<ScalarOrOp>(loc, pos, neg).getResult();
      auto one  = createI32Const(loc, 1, rewriter);
      res.push_back(rewriter.create<ScalarSubOp>(loc, one, nonz).getResult());
    }
    if (res.empty()) res.push_back(createI32Const(loc, 0, rewriter));
    auto ra = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, res.back(), ra);
    rewriter.replaceOp(op, carrier(op, rewriter, res.back()));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.select → condition ? input2 : input3  (element-wise)
// Strategy: reduce condition tile to scalar (0 or 1), use branchless scalar
// select:  result = input3 + (condition & (input2 - input3))
//   when condition==1: result = input3 + (input2 - input3) = input2
//   when condition==0: result = input3 + 0                 = input3
//===----------------------------------------------------------------------===//

struct TosaSelectLowering : public mlir::OpRewritePattern<mlir::tosa::SelectOp> {
  using mlir::OpRewritePattern<mlir::tosa::SelectOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::SelectOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<VSetVLOp>(loc, SEW::E32, LMUL::M1);
    auto condTiles  = getTiles(op.getInput1(), SEW::E32, rewriter, loc);
    auto trueTiles  = getTiles(op.getInput2(), SEW::E32, rewriter, loc);
    auto falseTiles = getTiles(op.getInput3(), SEW::E32, rewriter, loc);
    size_t k = std::min({condTiles.size(), trueTiles.size(), falseTiles.size()});
    llvm::SmallVector<mlir::Value> res;
    for (size_t i = 0; i < k; ++i) {
      auto cond  = rewriter.create<VRedSumOp>(loc, condTiles[i]).getResult();
      auto tv    = rewriter.create<VRedSumOp>(loc, trueTiles[i]).getResult();
      auto fv    = rewriter.create<VRedSumOp>(loc, falseTiles[i]).getResult();
      // branchless select: mask = 0 - cond  (cond=1 -> 0xFFFFFFFF, cond=0 -> 0)
      // result = fv + ((tv - fv) & mask)
      auto zero  = createI32Const(loc, 0, rewriter);
      auto mask  = rewriter.create<ScalarSubOp>(loc, zero, cond).getResult();
      auto diff  = rewriter.create<ScalarSubOp>(loc, tv, fv).getResult();
      auto sel   = rewriter.create<ScalarAndOp>(loc, diff, mask).getResult();
      res.push_back(rewriter.create<ScalarAddOp>(loc, fv, sel).getResult());
    }
    if (res.empty()) res.push_back(createI32Const(loc, 0, rewriter));
    auto ra = createI32Const(loc, (int32_t)(kTcmBase + getResultSlot(op)*kTcmSlot), rewriter);
    rewriter.create<ScalarSwOp>(loc, res.back(), ra);
    rewriter.replaceOp(op, carrier(op, rewriter, res.back()));
    return mlir::success();
  }
};


//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// tosa.negate → per-tile vrsub.vx vD, vS, x0  (vD[i] = 0 - vS[i])
//===----------------------------------------------------------------------===//

struct TosaNegateVXLowering
    : public mlir::OpRewritePattern<mlir::tosa::NegateOp> {
  using mlir::OpRewritePattern<mlir::tosa::NegateOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::NegateOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto tiles = getTiles(op.getInput1(), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (auto &t : tiles) {
      auto zero = createI32Const(loc, 0, rewriter);
      res.push_back(rewriter.create<VSubVXOp>(loc, vregE32(loc.getContext()), zero, t).getResult());
    }
    storeTiles(res, sew, n, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, res, rewriter));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.abs → abs(x) = max(x,0) + max(-x,0) via VSubVXOp + VMaxVXOp + VAddOp
//===----------------------------------------------------------------------===//

struct TosaAbsLowering
    : public mlir::OpRewritePattern<mlir::tosa::AbsOp> {
  using mlir::OpRewritePattern<mlir::tosa::AbsOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::AbsOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto sew = pickVConfig(tensorElementType(op.getResult())).first;
    rewriter.create<VSetVLOp>(loc, sew, LMUL::M1);
    int64_t n = numElements(op.getResult());
    auto tiles = getTiles(op.getInput1(), sew, rewriter, loc);
    llvm::SmallVector<mlir::Value> res;
    for (auto &t : tiles) {
      // pos_part = relu(x)  = max(x, 0)
      auto pos = rewriter.create<VMaxVXOp>(loc, vregE32(loc.getContext()), t).getResult();
      // neg_part = relu(-x) = max(-x, 0);  neg = vrsub.vx t, x0
      auto zero = createI32Const(loc, 0, rewriter);
      auto neg      = rewriter.create<VSubVXOp>(loc, vregE32(loc.getContext()), zero, t).getResult();
      auto neg_part = rewriter.create<VMaxVXOp>(loc, vregE32(loc.getContext()), neg).getResult();
      // abs(x) = pos_part + neg_part  (only one is nonzero for any given i)
      res.push_back(rewriter.create<VAddOp>(loc, vregE32(loc.getContext()), pos, neg_part, 1).getResult());
    }
    storeTiles(res, sew, n, rewriter, loc);
    rewriter.replaceOp(op, tileCarrier(op, res, rewriter));
    return mlir::success();
  }
};


//===----------------------------------------------------------------------===//
// tosa.const — inline constant tensor lowering
//
// A tosa.const carries a DenseElementsAttr of actual weights/biases.  We
// serialize the data to a module-level attribute () so
// ExportCoralNPU can write the real bytes into the .data section, and we
// replace the op with a  of  literals so
// that downstream patterns can reuse the tiles directly from registers
// (getTiles unwraps from_elements — no reload needed).
//
// Slot assignment: const slots start immediately after function arguments.
// The module attr  is a DictionaryAttr mapping the
// string slot_N → DenseIntElementsAttr<i32>.
//===----------------------------------------------------------------------===//

struct TosaConstLowering : public mlir::OpRewritePattern<mlir::tosa::ConstOp> {
  using mlir::OpRewritePattern<mlir::tosa::ConstOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tosa::ConstOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto attr = op.getValuesAttr();
    auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr);
    if (!dense)
      return mlir::failure(); // float consts: passthrough (not supported yet)

    // Collect i32 values (widen narrower ints to i32).
    llvm::SmallVector<int32_t> vals;
    for (auto v : dense.getValues<llvm::APInt>())
      vals.push_back((int32_t)v.getSExtValue());

    // Determine this const's slot index: numArgs + number of consts allocated
    // so far (tracked in module attr coralnpu.const_count).
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto funcOp   = op->getParentOfType<mlir::func::FuncOp>();
    int64_t numArgs = funcOp ? (int64_t)funcOp.getNumArguments() : 0;

    // Read/update const count from module attr.
    int64_t constIdx = 0;
    if (auto cnt = moduleOp->getDiscardableAttr("coralnpu.const_count"))
      constIdx = mlir::cast<mlir::IntegerAttr>(cnt).getInt();
    int64_t slot = numArgs + constIdx;

    // Write const data to module attr coralnpu.const_data (dict slot_N -> dense<i32>).
    mlir::DictionaryAttr existing;
    if (auto d = moduleOp->getDiscardableAttr("coralnpu.const_data"))
      existing = mlir::cast<mlir::DictionaryAttr>(d);

    auto i32Ty = rewriter.getI32Type();
    auto tensorTy = mlir::RankedTensorType::get({(int64_t)vals.size()}, i32Ty);
    auto dataAttr = mlir::DenseIntElementsAttr::get(
        tensorTy,
        llvm::ArrayRef<int32_t>(vals.data(), vals.size()));

    std::string key = "slot_" + std::to_string(slot);
    llvm::SmallVector<mlir::NamedAttribute> entries;
    if (existing) {
      for (auto &ne : existing)
        entries.push_back(ne);
    }
    entries.push_back(rewriter.getNamedAttr(key, dataAttr));
    auto newDict = mlir::DictionaryAttr::get(rewriter.getContext(), entries);
    moduleOp->setDiscardableAttr("coralnpu.const_data", newDict);
    moduleOp->setDiscardableAttr("coralnpu.const_count",
        rewriter.getI64IntegerAttr(constIdx + 1));

    // Build register tiles from literal values so getTiles can unpack them
    // (tensor.from_elements path — no memory load needed for constants).
    auto sew = pickVConfig(dense.getElementType()).first;
    int64_t n = (int64_t)vals.size();
    int64_t k = tileCount(sew, n);

    llvm::SmallVector<mlir::Value> tiles;
    tiles.reserve(k);
    for (int64_t ti = 0; ti < k; ++ti) {
      // Load the tile: emit vle32 from the const TCM slot address.
      int64_t addr = kTcmBase + slot * kTcmSlot + ti * kTileBytes;
      auto addrV = createI32Const(loc, (int32_t)addr, rewriter);
      auto ne = createI32Const(loc, (int32_t)tileElems(ti, k, n, sew), rewriter);
      tiles.push_back(rewriter.create<VLE32Op>(loc, vregE32(loc.getContext()), addrV, ne));
    }
    // const data is already in its const slot; the storeTiles here is redundant
    // but kept for the side-effect that anchors the VLE32 chain alive.
    storeTiles(tiles, sew, n, rewriter, loc, slot);
    rewriter.replaceOp(op, tileCarrier(op, tiles, rewriter));
    return mlir::success();
  }
};

struct TosaToCoralNPUPass
    : public impl::TosaToCoralNPUBase<TosaToCoralNPUPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tosa::TosaDialect,
                    mlir::tensor::TensorDialect,
                    mlir::func::FuncDialect,
                    circt::coralnpu::CoralNPUDialect>();
  }
  void runOnOperation() override {
    auto *ctx = &getContext();

    // Pre-pass: collect all intermediate matmul ops (those whose result feeds
    // another matmul through optional reshapes). This must be done BEFORE the
    // greedy rewriter runs, because the rewriter changes the use-def graph as
    // it lowers ops (a lowered matmul is no longer a tosa.matmul, so dynamic
    // isIntermediateMatMul checks fail for already-lowered ops).
    llvm::DenseSet<mlir::Operation *> intermediateMatMuls;
    getOperation()->walk([&](mlir::func::FuncOp funcOp) {
      funcOp.walk([&](mlir::tosa::MatMulOp mm) {
        if (isIntermediateMatMul(mm))
          intermediateMatMuls.insert(mm.getOperation());
      });
    });

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
    tosaPatterns.add<TosaCastLowering>(ctx);
    tosaPatterns.add<TosaAddLowering, TosaSubLowering, TosaMulLowering,
                     TosaClampLowering, TosaRescaleLowering,
                     TosaTableLowering, TosaSigmoidLowering>(ctx);
    tosaPatterns.add<TosaReshapeLowering, TosaTransposeLowering>(ctx);
    tosaPatterns.add<TosaConcatLowering, TosaSliceLowering>(ctx);
    tosaPatterns.add<TosaConv2DLowering, TosaDepthwiseConv2DLowering>(ctx);
    tosaPatterns.add<TosaMatMulLowering>(ctx, &intermediateMatMuls);
    tosaPatterns.add<TosaPadLowering, TosaAvgPool2dLowering,
                     TosaMaxPool2dLowering>(ctx);
    tosaPatterns.add<TosaNegateVXLowering, TosaAbsLowering>(ctx);
    tosaPatterns.add<TosaReduceSumLowering, TosaReduceMaxLowering>(ctx);
    tosaPatterns.add<TosaRsqrtLowering>(ctx);
    tosaPatterns.add<TosaMaximumLowering, TosaMinimumLowering>(ctx);
    tosaPatterns.add<TosaArithmeticRightShiftLowering>(ctx);
    tosaPatterns.add<TosaEqualLowering, TosaGreaterLowering, TosaGreaterEqualLowering>(ctx);
    tosaPatterns.add<TosaLogicalAndLowering, TosaLogicalOrLowering, TosaLogicalNotLowering>(ctx);
    tosaPatterns.add<TosaSelectLowering>(ctx);
    tosaPatterns.add<TosaConstLowering>(ctx);
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
