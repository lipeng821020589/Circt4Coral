//===- CoralNPUEmitAssembly.cpp - Assembly emission -----------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUEncodings.h"
#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "circt/Dialect/CoralNPU/CoralNPUPasses.h"
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
    llvm::outs() << "# Coral NPU Assembly (generated by CIRCT)\n";
    llvm::outs() << "# ISA: RISC-V rv32im + Coral NPU custom extensions\n\n";

    // Text assembly
    llvm::outs() << ".text\n";
    getOperation()->walk([&](mlir::Operation *op) {
      if (op->getDialect()->getNamespace() == "coralnpu")
        emitInstruction(op, llvm::outs());
    });
    llvm::outs() << "\n# End of assembly\n\n";

    // Binary encoding section
    llvm::outs() << "# Binary encoding (RISC-V little-endian):\n";
    llvm::outs() << "# addr : hex_word  mnemonic\n";
    unsigned pc = 0;
    getOperation()->walk([&](mlir::Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() == "coralnpu") {
        uint32_t encoded = encodeBinary(op);
        char buf[64];
        snprintf(buf, sizeof(buf), "%04x : %08x  ", pc, encoded);
        llvm::outs() << buf;
        emitInstruction(op, llvm::outs());
        pc += 4;
      }
    });
    llvm::outs() << "\n; Total: " << pc << " bytes (" << (pc / 4) << " instructions)\n";
  }

  /// Encode a CoralNPU operation to a 32-bit RISC-V instruction word.
  static uint32_t encodeBinary(mlir::Operation *op) {
    auto rd = [](mlir::Operation *o) -> uint8_t {
      if (auto a = o->getDiscardableAttr("xreg_out_0"))
        return mlir::cast<mlir::IntegerAttr>(a).getInt();
      return 0;
    };
    auto rs1 = [](mlir::Operation *o) -> uint8_t {
      if (auto a = o->getDiscardableAttr("xreg_0"))
        return mlir::cast<mlir::IntegerAttr>(a).getInt();
      return 0;
    };
    auto rs2 = [](mlir::Operation *o) -> uint8_t {
      if (auto a = o->getDiscardableAttr("xreg_1"))
        return mlir::cast<mlir::IntegerAttr>(a).getInt();
      return 0;
    };

    // Scalar RISC-V instructions
    if (mlir::isa<ScalarAddOp>(op))
      return encodeRType(Opcode::OP, rd(op), Funct3::ADD_SUB, rs1(op), rs2(op), Funct7::BASE);
    if (mlir::isa<ScalarSubOp>(op))
      return encodeRType(Opcode::OP, rd(op), Funct3::ADD_SUB, rs1(op), rs2(op), Funct7::SUB_SRA);
    if (mlir::isa<ScalarMulOp>(op))
      return encodeRType(Opcode::OP, rd(op), Funct3::MUL_DIV, rs1(op), rs2(op), Funct7::MULDIV);
    if (mlir::isa<ScalarAndOp>(op))
      return encodeRType(Opcode::OP, rd(op), Funct3::AND, rs1(op), rs2(op), Funct7::BASE);
    if (mlir::isa<ScalarOrOp>(op))
      return encodeRType(Opcode::OP, rd(op), Funct3::OR, rs1(op), rs2(op), Funct7::BASE);
    if (mlir::isa<ScalarXorOp>(op))
      return encodeRType(Opcode::OP, rd(op), Funct3::XOR, rs1(op), rs2(op), Funct7::BASE);
    if (mlir::isa<ScalarSllOp>(op))
      return encodeRType(Opcode::OP, rd(op), Funct3::SLL, rs1(op), rs2(op), Funct7::BASE);
    if (mlir::isa<ScalarSrlOp>(op))
      return encodeRType(Opcode::OP, rd(op), Funct3::SRL_SRA, rs1(op), rs2(op), Funct7::BASE);
    if (mlir::isa<ScalarSraOp>(op))
      return encodeRType(Opcode::OP, rd(op), Funct3::SRL_SRA, rs1(op), rs2(op), Funct7::SUB_SRA);
    if (mlir::isa<ScalarSltOp>(op))
      return encodeRType(Opcode::OP, rd(op), Funct3::SLT, rs1(op), rs2(op), Funct7::BASE);
    if (mlir::isa<ScalarSltuOp>(op))
      return encodeRType(Opcode::OP, rd(op), Funct3::SLTU, rs1(op), rs2(op), Funct7::BASE);
    if (mlir::isa<ScalarDivOp>(op))
      return encodeRType(Opcode::OP, rd(op), 0b100, rs1(op), rs2(op), Funct7::MULDIV);
    if (auto liOp = mlir::dyn_cast<ScalarLiOp>(op))
      return encodeIType(Opcode::OP_IMM, rd(op), 0b000, 0, liOp.getValue());
    if (mlir::isa<ScalarLwOp>(op))
      return encodeIType(Opcode::LOAD, rd(op), Funct3::LW, rs1(op), 0);
    if (mlir::isa<ScalarSwOp>(op)) {
      uint32_t sw = ((0 & 0x7F) << 25) | (rs2(op) << 20) | (rs1(op) << 15) |
                    (Funct3::SW << 12) | (0 & 0x1F << 7) | Opcode::STORE;
      return sw;
    }

    // Control flow
    if (mlir::isa<BeqOp>(op))
      return encodeBType(Opcode::BRANCH, Funct3::BEQ, rs1(op), rs2(op), 0);
    if (mlir::isa<BneOp>(op))
      return encodeBType(Opcode::BRANCH, Funct3::BNE, rs1(op), rs2(op), 0);
    if (mlir::isa<JalOp>(op))
      return encodeUType(Opcode::JAL, rd(op), 0);
    if (mlir::isa<JalrOp>(op))
      return encodeIType(Opcode::JALR, rd(op), 0, rs1(op), 0);

    // Vector ops (custom encoding)
    if (mlir::isa<VAddOp>(op))
      return encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VADD, 0, 0, 0);
    if (mlir::isa<VSubOp>(op))
      return encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VSUB, 0, 0, 0);
    if (mlir::isa<VMulOp>(op))
      return encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VMUL, 0, 0, 0);
    if (mlir::isa<VDotOp>(op))
      return encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VDOT, 0, 0, 0);
    if (mlir::isa<VWAddOp>(op))
      return encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VWADD, 0, 0, 0);
    if (mlir::isa<VRedSumOp>(op))
      return encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VREDSUM, 0, 0, 0);
    if (mlir::isa<VLE8Op>(op))
      return encodeIType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VLE, 0, 0);
    if (mlir::isa<VLE16Op>(op))
      return encodeIType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VLE, 0, 1);
    if (mlir::isa<VLE32Op>(op))
      return encodeIType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VLE, 0, 2);

    // Matrix ops
    if (mlir::isa<OuterProductOp>(op))
      return encodeRType(Opcode::CUSTOM_MAT, 0, NPUFunct3::OUTER_PRODUCT, 0, 0, 0);
    if (mlir::isa<AConvOp>(op))
      return encodeRType(Opcode::CUSTOM_MAT, 0, NPUFunct3::ACONV, 0, 0, 0);
    if (mlir::isa<AccReadOp>(op))
      return encodeRType(Opcode::CUSTOM_MAT, 0, NPUFunct3::ACCREAD, 0, 0, 0);

    // DMA
    if (mlir::isa<DmaLoadOp>(op))
      return encodeIType(Opcode::CUSTOM_DMA, 0, NPUFunct3::DMA_LOAD, 0, 0);
    if (mlir::isa<DmaStoreOp>(op))
      return encodeIType(Opcode::CUSTOM_DMA, 0, NPUFunct3::DMA_STORE, 0, 0);

    // Default: NOP
    return encodeIType(Opcode::OP_IMM, 0, 0, 0, 0);
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
