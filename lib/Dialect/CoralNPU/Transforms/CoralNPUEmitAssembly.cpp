//===- CoralNPUEmitAssembly.cpp - Assembly emission -----------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUPasses.h"
#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::coralnpu;

namespace circt {
namespace coralnpu {
#define GEN_PASS_DEF_CORALNPUEMITASSEMBLY
#include "circt/Dialect/CoralNPU/Passes.h.inc"
} // namespace coralnpu
} // namespace circt

namespace {

static void emitInstruction(mlir::Operation *op, llvm::raw_ostream &os) {
  // Helper: read xreg annotation from an op
  auto getXReg = [&](mlir::Operation *o, llvm::StringRef attrName) -> std::string {
    if (auto attr = o->getDiscardableAttr(attrName))
      return "x" + std::to_string(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    return "x?";
  };

  os << "  ";
  if (auto addOp = mlir::dyn_cast<ScalarAddOp>(op))
    os << "add    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto subOp = mlir::dyn_cast<ScalarSubOp>(op))
    os << "sub    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto mulOp = mlir::dyn_cast<ScalarMulOp>(op))
    os << "mul    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto divOp = mlir::dyn_cast<ScalarDivOp>(op))
    os << "div    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto andOp = mlir::dyn_cast<ScalarAndOp>(op))
    os << "and    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto orOp = mlir::dyn_cast<ScalarOrOp>(op))
    os << "or     " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto xorOp = mlir::dyn_cast<ScalarXorOp>(op))
    os << "xor    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto sllOp = mlir::dyn_cast<ScalarSllOp>(op))
    os << "sll    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto srlOp = mlir::dyn_cast<ScalarSrlOp>(op))
    os << "srl    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto sraOp = mlir::dyn_cast<ScalarSraOp>(op))
    os << "sra    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto sltOp = mlir::dyn_cast<ScalarSltOp>(op))
    os << "slt    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto sltuOp = mlir::dyn_cast<ScalarSltuOp>(op))
    os << "sltu   " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1");
  else if (auto liOp = mlir::dyn_cast<ScalarLiOp>(op))
    os << "li     " << getXReg(op, "xreg_out_0") << ", " << liOp.getValue();
  else if (auto vaddOp = mlir::dyn_cast<VAddOp>(op)) {
    os << "vadd.vv";
    if (vaddOp.getStripmine() > 1)
      os << "  # stripmine=" << vaddOp.getStripmine();
  } else if (auto vsubOp = mlir::dyn_cast<VSubOp>(op)) {
    os << "vsub.vv";
    if (vsubOp.getStripmine() > 1)
      os << "  # stripmine=" << vsubOp.getStripmine();
  } else if (auto vmulOp = mlir::dyn_cast<VMulOp>(op)) {
    os << "vmul.vv";
    if (vmulOp.getStripmine() > 1)
      os << "  # stripmine=" << vmulOp.getStripmine();
  } else if (auto vwaddOp = mlir::dyn_cast<VWAddOp>(op)) {
    os << "vwadd.vv";
    if (vwaddOp.getStripmine() > 1)
      os << "  # stripmine=" << vwaddOp.getStripmine();
  } else if (auto vdotOp = mlir::dyn_cast<VDotOp>(op)) {
    os << "vdot.vv";
    if (vdotOp.getStripmine() > 1)
      os << "  # stripmine=" << vdotOp.getStripmine();
  } else if (auto vredsumOp = mlir::dyn_cast<VRedSumOp>(op))
    os << "vredsum.vs";
  else if (mlir::isa<VLE8Op>(op))
    os << "vle8.v";
  else if (mlir::isa<VLE16Op>(op))
    os << "vle16.v";
  else if (mlir::isa<VLE32Op>(op))
    os << "vle32.v";
  else if (mlir::isa<VSE8Op>(op))
    os << "vse8.v";
  else if (mlir::isa<VSE16Op>(op))
    os << "vse16.v";
  else if (mlir::isa<VSE32Op>(op))
    os << "vse32.v";
  else if (mlir::isa<OuterProductOp>(op))
    os << "outer_product";
  else if (mlir::isa<AConvOp>(op))
    os << "aconv";
  else if (mlir::isa<AccReadOp>(op))
    os << "accread  " << getXReg(op, "xreg_out_0") << ", acc";
  else if (auto lwOp = mlir::dyn_cast<ScalarLwOp>(op))
    os << "lw     " << getXReg(op, "xreg_out_0") << ", " << getXReg(op, "xreg_0") << "(tcm)";
  else if (auto swOp = mlir::dyn_cast<ScalarSwOp>(op))
    os << "sw     " << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "(tcm)";
  else if (mlir::isa<DmaLoadOp>(op))
    os << "dma_load";
  else if (mlir::isa<DmaStoreOp>(op))
    os << "dma_store";
  else if (mlir::isa<BeqOp>(op))
    os << "beq    .L?.L?";
  else if (mlir::isa<BneOp>(op))
    os << "bne    .L?.L?";
  else if (mlir::isa<JalOp>(op))
    os << "jal    .L?";
  else if (mlir::isa<JalrOp>(op))
    os << "jalr";
  else if (mlir::isa<BranchOp>(op))
    os << "j       .L?";
  else if (mlir::isa<ReturnOp>(op))
    os << "ret";
  else
    os << "# unknown: " << op->getName().getStringRef();
  os << "\n";
}

struct CoralNPUEmitAssemblyPass
    : public impl::CoralNPUEmitAssemblyBase<CoralNPUEmitAssemblyPass> {
  void runOnOperation() override {
    llvm::outs() << "# Coral NPU Assembly (generated by CIRCT)\n\n";
    getOperation()->walk([&](mlir::Operation *op) {
      if (op->getDialect()->getNamespace() == "coralnpu")
        emitInstruction(op, llvm::outs());
    });
    llvm::outs() << "\n# End of assembly\n";
  }
};

} // namespace

namespace circt {
namespace coralnpu {
std::unique_ptr<mlir::Pass> createCoralNPUEmitAssemblyPass() {
  return std::make_unique<CoralNPUEmitAssemblyPass>();
}
} // namespace coralnpu
} // namespace circt
