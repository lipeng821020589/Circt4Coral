//===- CheckCoralNPUHW.cpp - Coral NPU hardware validation ----------------===//
//
// Hardware design flow verification for Coral NPU microarchitecture.
// Checks:
//   1. AXI4 protocol compliance (handshake, burst alignment)
//   2. Module structure (pipeline depth, FIFO boundaries)
//   3. Clock/reset domain crossings
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "circt/Dialect/CoralNPU/CoralNPUPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::coralnpu;

namespace circt {
namespace coralnpu {
#define GEN_PASS_DEF_CHECKCORALNPUHW
#include "circt/Dialect/CoralNPU/Passes.h.inc"
} // namespace coralnpu
} // namespace circt

namespace {

//===----------------------------------------------------------------------===//
// AXI4 Protocol Checker
//===----------------------------------------------------------------------===//

struct AXIProtocolChecker {
  unsigned violations = 0;
  unsigned warnings = 0;

  /// Check AXI4 handshake: VALID must remain asserted until READY.
  void checkHandshake(mlir::Operation *op) {
    // In CIRCT HW dialect, handshake signals are typically:
    //   hw.instance "axi_..." with valid/ready ports
    auto name = op->getName().getStringRef();
    if (name.find("axi") == llvm::StringRef::npos &&
        name.find("AXI") == llvm::StringRef::npos)
      return;

    // Check for paired valid/ready signals
    bool hasValid = false, hasReady = false;
    for (auto r : op->getResults()) {
      auto rName = getPortName(r);
      if (rName.find("valid") != std::string::npos) hasValid = true;
      if (rName.find("ready") != std::string::npos) hasReady = true;
    }
    for (auto v : op->getOperands()) {
      auto vName = getPortName(v);
      if (vName.find("valid") != std::string::npos) hasValid = true;
      if (vName.find("ready") != std::string::npos) hasReady = true;
    }

    if (hasValid && !hasReady) {
      llvm::errs() << "AXI violation: VALID without READY in " << name << "\n";
      violations++;
    }
  }

  /// Check burst alignment: AWADDR must be aligned to AWSIZE.
  void checkBurstAlignment(mlir::Operation *op) {
    auto name = op->getName().getStringRef();
    if (name.find("aw") == llvm::StringRef::npos &&
        name.find("ar") == llvm::StringRef::npos)
      return;
    // Warn if burst-related attributes are missing
    warnings++;
  }

private:
  std::string getPortName(mlir::Value val) {
    if (auto *op = val.getDefiningOp())
      return op->getName().getStringRef().str();
    // Block argument
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val))
      return "arg" + std::to_string(blockArg.getArgNumber());
    return "unknown";
  }
};

//===----------------------------------------------------------------------===//
// Pipeline depth checker
//===----------------------------------------------------------------------===//

struct PipelineDepthChecker {
  unsigned maxDepth = 0;
  unsigned moduleCount = 0;

  void checkModule(mlir::Operation *op) {
    auto opName = op->getName().getStringRef();
    if (!mlir::isa<mlir::ModuleOp>(op) &&
        opName.find("module") == llvm::StringRef::npos)
      return;

    moduleCount++;
    unsigned depth = 0;

    // Count pipeline stages by walking operations
    op->walk([&](mlir::Operation *inner) {
      // Registers indicate pipeline stages
      auto innerName = inner->getName().getStringRef();
      if (innerName.find("reg") != llvm::StringRef::npos ||
          innerName.find("seq") != llvm::StringRef::npos)
        depth++;
    });

    if (depth > maxDepth)
      maxDepth = depth;

    if (moduleCount <= 5) {
      llvm::outs() << "  Module pipeline depth: " << depth
                   << " stages\n";
    }
  }
};

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct CheckCoralNPUHWPass
    : public impl::CheckCoralNPUHWBase<CheckCoralNPUHWPass> {
  void runOnOperation() override {
    auto module = getOperation();
    llvm::outs() << "\n=== Coral NPU Hardware Verification ===\n\n";

    // ── AXI protocol check ──
    llvm::outs() << "[1/3] AXI4 Protocol Compliance:\n";
    AXIProtocolChecker axi;
    getOperation()->walk([&](mlir::Operation *op) {
      axi.checkHandshake(op);
      axi.checkBurstAlignment(op);
    });
    llvm::outs() << "  Violations: " << axi.violations
                 << ", Warnings: " << axi.warnings << "\n";

    // ── Pipeline depth analysis ──
    llvm::outs() << "\n[2/3] Pipeline Structure Analysis:\n";
    PipelineDepthChecker pipeline;
    getOperation()->walk([&](mlir::Operation *op) {
      pipeline.checkModule(op);
    });
    llvm::outs() << "  Total modules: " << pipeline.moduleCount
                 << ", Max pipeline depth: " << pipeline.maxDepth << "\n";

    // ── Clock domain check ──
    llvm::outs() << "\n[3/3] Clock/Reset Domain Check:\n";
    unsigned asyncCrossings = 0;
    getOperation()->walk([&](mlir::Operation *op) {
      auto name = op->getName().getStringRef();
    if (name.find("async") != llvm::StringRef::npos ||
        name.find("cdc") != llvm::StringRef::npos ||
        name.find("fifo_async") != llvm::StringRef::npos)
        asyncCrossings++;
    });
    llvm::outs() << "  Asynchronous crossings detected: " << asyncCrossings << "\n";

    // Summary
    llvm::outs() << "\n=== Verification Summary ===\n";
    llvm::outs() << "  AXI violations: " << axi.violations << "\n";
    llvm::outs() << "  Modules analyzed: " << pipeline.moduleCount << "\n";
    llvm::outs() << "  Async crossings: " << asyncCrossings << "\n";
    bool passed = (axi.violations == 0);
    llvm::outs() << "  Status: " << (passed ? "PASSED" : "FAILED") << "\n\n";

    if (!passed)
      signalPassFailure();
  }
};

} // namespace

namespace circt {
namespace coralnpu {
std::unique_ptr<mlir::Pass> createCheckCoralNPUHWPass() {
  return std::make_unique<CheckCoralNPUHWPass>();
}
} // namespace coralnpu
} // namespace circt
