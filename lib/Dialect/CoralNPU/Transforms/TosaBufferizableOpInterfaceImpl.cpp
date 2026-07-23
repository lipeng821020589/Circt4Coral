//===- TosaBufferizableOpInterfaceImpl.cpp - TOSA bufferization -----------===//

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {

//===----------------------------------------------------------------------===//
// tosa.const → memref.global + memref.get_global
//===----------------------------------------------------------------------===//

struct TosaConstOpInterface
    : public BufferizableOpInterface::ExternalModel<TosaConstOpInterface,
                                                    mlir::tosa::ConstOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const { return false; }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const { return false; }
  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                     const AnalysisState &) const { return {}; }
  BufferRelation bufferRelation(Operation *, OpResult,
                                const AnalysisState &) const {
    return BufferRelation::Unknown;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto constOp = cast<mlir::tosa::ConstOp>(op);
    auto tensorType = dyn_cast<RankedTensorType>(constOp.getType());
    if (!tensorType) return failure();

    auto moduleOp = constOp->getParentOfType<ModuleOp>();
    if (!moduleOp) return failure();

    auto memrefType = MemRefType::get(tensorType.getShape(),
                                      tensorType.getElementType());

    static unsigned counter = 0;
    std::string name = "__tosa_const_" + std::to_string(counter++);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    rewriter.create<memref::GlobalOp>(
        constOp.getLoc(),
        rewriter.getStringAttr(name),
        rewriter.getStringAttr("private"),
        memrefType, constOp.getValues(), /*constant=*/true,
        IntegerAttr{});

    rewriter.setInsertionPoint(op);
    replaceOpWithNewBufferizedOp<memref::GetGlobalOp>(
        rewriter, op, memrefType, name);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// tosa.reshape → memref.reshape (using static output shape)
//===----------------------------------------------------------------------===//

struct TosaReshapeOpInterface
    : public BufferizableOpInterface::ExternalModel<TosaReshapeOpInterface,
                                                    mlir::tosa::ReshapeOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &) const {
    // Read from input tensor; shape operand is !tosa.shape, not a tensor.
    auto reshapeOp = cast<mlir::tosa::ReshapeOp>(op);
    return opOperand == reshapeOp.getInput1Mutable();
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const { return false; }
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                     const AnalysisState &) const {
    auto reshapeOp = cast<mlir::tosa::ReshapeOp>(op);
    if (opOperand == reshapeOp.getInput1Mutable())
      return {{op->getOpResult(0), BufferRelation::Equivalent}};
    return {};
  }
  BufferRelation bufferRelation(Operation *, OpResult,
                                const AnalysisState &) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto reshapeOp = cast<mlir::tosa::ReshapeOp>(op);
    auto resultTy = cast<RankedTensorType>(reshapeOp.getType());

    FailureOr<Value> inputBuf =
        getBuffer(rewriter, reshapeOp.getInput1(), options, state);
    if (failed(inputBuf)) return failure();

    auto resultMemrefType = MemRefType::get(resultTy.getShape(),
                                            resultTy.getElementType());

    // memref.reshape requires identity layout source.
    // If source has non-identity layout (strided), copy to a flat buffer first.
    Value srcBuf = *inputBuf;
    auto srcMemrefTy = dyn_cast<MemRefType>(srcBuf.getType());
    if (srcMemrefTy && !srcMemrefTy.getLayout().isIdentity()) {
      auto flatSrcTy = MemRefType::get(srcMemrefTy.getShape(),
                                       srcMemrefTy.getElementType());
      auto flatAlloc = rewriter.create<memref::AllocOp>(op->getLoc(), flatSrcTy);
      rewriter.create<memref::CopyOp>(op->getLoc(), srcBuf, flatAlloc);
      srcBuf = flatAlloc;
    }

    // Build shape memref for reshape.
    SmallVector<Value> shapeValues;
    for (int64_t dim : resultTy.getShape())
      shapeValues.push_back(rewriter.create<arith::ConstantIndexOp>(
          op->getLoc(), dim));

    int64_t rank = resultTy.getRank();
    auto shapeTy = MemRefType::get({rank}, rewriter.getIndexType());
    auto shapeMem = rewriter.create<memref::AllocOp>(op->getLoc(), shapeTy);
    for (int64_t i = 0; i < rank; ++i) {
      auto idx = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), i);
      rewriter.create<memref::StoreOp>(op->getLoc(), shapeValues[i],
                                       shapeMem, ValueRange{idx});
    }

    replaceOpWithNewBufferizedOp<memref::ReshapeOp>(
        rewriter, op, resultMemrefType, srcBuf, shapeMem);
    return success();
  }
};

} // namespace

namespace circt {
namespace coralnpu {

void registerTosaBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::tosa::TosaDialect *) {
        mlir::tosa::ConstOp::attachInterface<TosaConstOpInterface>(*ctx);
        mlir::tosa::ReshapeOp::attachInterface<TosaReshapeOpInterface>(*ctx);
      });
}

} // namespace coralnpu
} // namespace circt
