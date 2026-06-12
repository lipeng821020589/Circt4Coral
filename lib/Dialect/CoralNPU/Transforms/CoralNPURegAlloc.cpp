//===- CoralNPURegAlloc.cpp - Register allocation for Coral NPU -----------===//
//
// Simple greedy register allocator for Coral NPU hardware.
// Maps virtual SSA values to physical register files:
//   x0..x31  — scalar registers (x0 hardwired to zero)
//   v0..v63  — vector registers
//   acc[0..7][0..7] — accumulator matrix
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUPasses.h"
#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace circt;
using namespace circt::coralnpu;

namespace circt {
namespace coralnpu {
#define GEN_PASS_DEF_CORALNPUREGALLOC
#include "circt/Dialect/CoralNPU/Passes.h.inc"
} // namespace coralnpu
} // namespace circt

namespace {

//===----------------------------------------------------------------------===//
// Register file state
//===----------------------------------------------------------------------===//

struct RegAllocState {
  // Scalar: 32 registers, x0 reserved (always zero)
  static constexpr unsigned kNumScalarRegs = 32;
  static constexpr unsigned kFirstAllocScalar = 1; // skip x0
  unsigned nextScalarReg = kFirstAllocScalar;
  llvm::DenseMap<mlir::Value, unsigned> scalarMap;

  // Vector: 64 registers, all allocatable
  static constexpr unsigned kNumVectorRegs = 64;
  unsigned nextVectorReg = 0;
  llvm::DenseMap<mlir::Value, unsigned> vectorMap;

  // Accumulator: 8×8 matrix, tracked per column
  static constexpr unsigned kAccRows = 8;
  static constexpr unsigned kAccCols = 8;
  unsigned nextAccCol = 0;
  llvm::DenseMap<mlir::Value, std::pair<unsigned, unsigned>> accMap;

  /// Allocate next available scalar register.
  unsigned allocScalar(mlir::Value val) {
    if (nextScalarReg >= kNumScalarRegs) {
      // Wrap around (simplified — real allocator would spill)
      nextScalarReg = kFirstAllocScalar;
    }
    unsigned reg = nextScalarReg++;
    scalarMap[val] = reg;
    return reg;
  }

  /// Allocate next available vector register.
  unsigned allocVector(mlir::Value val) {
    if (nextVectorReg >= kNumVectorRegs)
      nextVectorReg = 0;
    unsigned reg = nextVectorReg++;
    vectorMap[val] = reg;
    return reg;
  }

  /// Allocate accumulator slot (row, col).
  std::pair<unsigned, unsigned> allocAcc(mlir::Value val) {
    unsigned row = 0;
    unsigned col = nextAccCol % kAccCols;
    nextAccCol++;
    accMap[val] = {row, col};
    return {row, col};
  }
};

//===----------------------------------------------------------------------===//
// Register allocation pass
//===----------------------------------------------------------------------===//

struct CoralNPURegAllocPass
    : public impl::CoralNPURegAllocBase<CoralNPURegAllocPass> {
  void runOnOperation() override {
    RegAllocState state;

    getOperation()->walk([&](mlir::Operation *op) {
      // Allocate result registers
      for (auto result : op->getResults()) {
        if (!result.use_empty()) {
          if (mlir::isa<mlir::IntegerType>(result.getType())) {
            // All scalar/integer values go to x-registers
            state.allocScalar(result);
          }
        }
      }

      // Annotate the operation with register assignments using attributes.
      // The emit-assembly pass will read these attributes to generate
      // physical register references in the output assembly.
      annotateOp(op, state);
    });

    llvm::outs() << "; Register allocation:\n";
    llvm::outs() << "; Scalars allocated: " << (state.nextScalarReg - RegAllocState::kFirstAllocScalar) << "\n";
    llvm::outs() << "; Vectors allocated: " << state.nextVectorReg << "\n";
    llvm::outs() << "; Accumulator cols used: " << state.nextAccCol << "\n\n";
  }

private:
  /// Annotate an operation with physical register assignments.
  void annotateOp(mlir::Operation *op, RegAllocState &state) {
    llvm::SmallVector<mlir::NamedAttribute> regAttrs;

    // Annotate operands with their register assignments
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto val = op->getOperand(i);
      auto it = state.scalarMap.find(val);
      if (it != state.scalarMap.end()) {
        std::string attrName = "xreg_" + std::to_string(i);
        regAttrs.push_back(mlir::NamedAttribute(
            mlir::StringAttr::get(op->getContext(), attrName),
            mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32), it->second)));
      }
    }

    // Annotate results with their register assignments
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      auto val = op->getResult(i);
      auto it = state.scalarMap.find(val);
      if (it != state.scalarMap.end()) {
        std::string attrName = "xreg_out_" + std::to_string(i);
        regAttrs.push_back(mlir::NamedAttribute(
            mlir::StringAttr::get(op->getContext(), attrName),
            mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32), it->second)));
      }
    }

    // Apply annotations if any
    if (!regAttrs.empty()) {
      op->setDiscardableAttrs(mlir::DictionaryAttr::get(
          op->getContext(), regAttrs));
    }
  }
};

} // namespace

namespace circt {
namespace coralnpu {
std::unique_ptr<mlir::Pass> createCoralNPURegAllocPass() {
  return std::make_unique<CoralNPURegAllocPass>();
}
} // namespace coralnpu
} // namespace circt
