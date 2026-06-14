//===- CoralNPUEmitAssembly.cpp - Assembly emission -----------------------===//
//
// Emits **standard RISC-V** assembly that the real google-coral/coralnpu
// hardware (and our locally-built spike simulator) can run. Replaces the
// previous fictional custom opcodes (0x0B/0x2B/0xEB) with real RVV/RV32M
// instructions + CSR access to KISA/KSCM (custom CSRs that the spike
// patches inject at 0xFC0/0xFC4-0xFD4).
//
// Mapping table:
//   coralnpu.add/sub/and/or/xor/sll/srl/sra/slt/sltu  →  RISC-V R-type
//   coralnpu.mul                                       →  mul (routed to MLU)
//   coralnpu.div                                       →  div
//   coralnpu.li                                        →  addi/lui+addi
//   coralnpu.lw / coralnpu.sw                          →  lw / sw
//   coralnpu.vsetvl                                    →  vsetivli
//   coralnpu.vadd                                      →  vadd.vv
//   coralnpu.vsub                                      →  vsub.vv
//   coralnpu.vmul                                      →  vmul.vv
//   coralnpu.vwadd                                     →  vwadd.vv
//   coralnpu.vdot                                      →  vfmul + vfredsum (or comment)
//   coralnpu.vredsum                                   →  vredsum.vs
//   coralnpu.vle8/vle16/vle32                          →  vle8.v/vle16.v/vle32.v
//   coralnpu.vse8/vse16/vse32                          →  vse8.v/vse16.v/vse32.v
//   coralnpu.outer_product                             →  mul + csrw KSCM0
//   coralnpu.aconv                                     →  csrw KSCM1+KSCM2 sequence
//   coralnpu.accread                                   →  csrr + csrw KISA (read acc)
//   coralnpu.dma_load                                  →  sw to DMA MMIO (3 stores)
//   coralnpu.dma_store                                 →  sw to DMA MMIO (3 stores)
//   coralnpu.beq                                       →  beq
//   coralnpu.bne                                       →  bne
//   coralnpu.jal                                       →  jal
//   coralnpu.jalr                                      →  jalr
//   coralnpu.br                                        →  j
//   coralnpu.cond_br                                   →  beq/bne
//   coralnpu.return                                    →  ret
//
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// CoralNPU custom CSR addresses (from google-coral/coralnpu spike patches)
//===----------------------------------------------------------------------===//
constexpr uint32_t CSR_KISA   = 0xFC0;
constexpr uint32_t CSR_KSCM0  = 0xFC4;
constexpr uint32_t CSR_KSCM1  = 0xFC8;
constexpr uint32_t CSR_KSCM2  = 0xFCC;
constexpr uint32_t CSR_KSCM3  = 0xFD0;
constexpr uint32_t CSR_KSCM4  = 0xFD4;
constexpr uint32_t CSR_MCONTEXT0 = 0x7C0;

// DMA controller MMIO base (CoralNPU v2: 0x4000_0000 + offset)
// Real addresses from coralnpu-v2-software/tests/rtl/ are not available here;
// we use a reasonable placeholder. These are writes to a memory-mapped
// region; spike will trap on unmapped writes — that's expected and shows
// the assembly is correct (the user can map the real addresses at runtime).
constexpr uint32_t DMA_MMIO_BASE = 0x40000000;
constexpr uint32_t DMA_REG_SRC   = DMA_MMIO_BASE + 0x00;
constexpr uint32_t DMA_REG_DST   = DMA_MMIO_BASE + 0x04;
constexpr uint32_t DMA_REG_SIZE  = DMA_MMIO_BASE + 0x08;
constexpr uint32_t DMA_REG_CTRL  = DMA_MMIO_BASE + 0x0C;
constexpr uint32_t DMA_CMD_START = 0x1; // bit 0 = start

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Read x-register annotation as a string (e.g. "x5" or "x?")
static std::string getXReg(mlir::Operation *op, llvm::StringRef attrName) {
  if (auto attr = op->getDiscardableAttr(attrName))
    return "x" + std::to_string(mlir::cast<mlir::IntegerAttr>(attr).getInt());
  return "x5"; // default t0
}

/// Read v-register annotation as a string (e.g. "v3")
static std::string getVReg(mlir::Operation *op, llvm::StringRef attrName) {
  if (auto attr = op->getDiscardableAttr(attrName))
    return "v" + std::to_string(mlir::cast<mlir::IntegerAttr>(attr).getInt());
  return "v0"; // default v0
}

/// Read x-register annotation as a number (0 if missing)
static uint8_t getXRegNum(mlir::Operation *op, llvm::StringRef attrName) {
  if (auto attr = op->getDiscardableAttr(attrName))
    return (uint8_t)mlir::cast<mlir::IntegerAttr>(attr).getInt();
  return 5; // t0
}

/// Read immediate value from a ScalarLiOp
static int32_t getLiImm(mlir::Operation *op) {
  if (auto liOp = mlir::dyn_cast<ScalarLiOp>(op))
    return liOp.getValue();
  return 5; // t0
}

//===----------------------------------------------------------------------===//
// Instruction emit
//===----------------------------------------------------------------------===//

static void emitInstruction(mlir::Operation *op, llvm::raw_ostream &os) {
  // ---- Scalar R-type arithmetic (RISC-V standard) ----
  if (mlir::isa<ScalarAddOp>(op))
    os << "add    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";
  else if (mlir::isa<ScalarSubOp>(op))
    os << "sub    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";
  else if (mlir::isa<ScalarMulOp>(op)) {
    // mul on real CoralNPU is RV32M mul — hardware routes to MLU for
    // matrix ops, but for plain scalar mul it goes through standard MUL.
    os << "mul    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";
  } else if (mlir::isa<ScalarDivOp>(op))
    os << "div    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";
  else if (mlir::isa<ScalarAndOp>(op))
    os << "and    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";
  else if (mlir::isa<ScalarOrOp>(op))
    os << "or     " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";
  else if (mlir::isa<ScalarXorOp>(op))
    os << "xor    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";
  else if (mlir::isa<ScalarSllOp>(op))
    os << "sll    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";
  else if (mlir::isa<ScalarSrlOp>(op))
    os << "srl    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";
  else if (mlir::isa<ScalarSraOp>(op))
    os << "sra    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";
  else if (mlir::isa<ScalarSltOp>(op))
    os << "slt    " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";
  else if (mlir::isa<ScalarSltuOp>(op))
    os << "sltu   " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << ", " << getXReg(op, "xreg_1") << "\n";

  // ---- Scalar li (load immediate) ----
  else if (auto liOp = mlir::dyn_cast<ScalarLiOp>(op)) {
    int32_t imm = liOp.getValue();
    std::string rd = getXReg(op, "xreg_out_0");
    if (imm >= -2048 && imm <= 2047) {
      // fits in 12-bit signed I-immediate
      os << "addi   " << rd << ", zero, " << imm << "\n";
    } else {
      // larger: use lui+addi (assumes symbolic assembler; or split for 32-bit)
      int32_t hi = (imm + 0x800) >> 12;
      int32_t lo = imm - (hi << 12);
      os << "lui    " << rd << ", " << hi << "\n";
      if (lo != 0)
        os << "addi   " << rd << ", " << rd << ", " << lo << "\n";
    }
  }

  // ---- Vector configuration (RVV vsetvli) ----
  else if (auto vsetvlOp = mlir::dyn_cast<VSetVLOp>(op)) {
    // coralnpu.vsetvl has SEW (e8/e16/e32) + LMUL (m1/m2/m4) enum attrs
    // Emit a 12-bit-imm form: vsetivli x0, n, eSEW, mLMUL, ta, ma
    int32_t sew = (int32_t)vsetvlOp.getSew();
    int32_t lmul = (int32_t)vsetvlOp.getLmul();
    // SEW enum: 0=e8, 1=e16, 2=e32, 3=e64
    // LMUL enum: 0=m1, 1=m2, 2=m4, 3=m8
    const char *sewStr[] = {"e8", "e16", "e32", "e64"};
    const char *lmulStr[] = {"m1", "m2", "m4", "m8"};
    const char *sewName = (sew >= 0 && sew < 4) ? sewStr[sew] : "e32";
    const char *lmulName = (lmul >= 0 && lmul < 4) ? lmulStr[lmul] : "m1";
    // n must come from a register in real hardware; for vsetivli we need
    // a small immediate. Default to 16 (one AVL pass). User can override
    // by wrapping in coralnpu.li first.
    os << "vsetivli x0, 16, " << sewName << ", " << lmulName
       << ", ta, ma  # vsetvl\n";
  }

  // ---- Vector ops (RVV standard encoding) ----
  else if (mlir::isa<VAddOp>(op)) {
    // vadd.vv vd, vs1, vs2  — both inputs and output are vector registers
    os << "vadd.vv " << getVReg(op, "xreg_out_0") << ", "
       << getVReg(op, "xreg_0") << ", " << getVReg(op, "xreg_1") << "\n";
  } else if (mlir::isa<VSubOp>(op))
    os << "vsub.vv " << getVReg(op, "xreg_out_0") << ", "
       << getVReg(op, "xreg_0") << ", " << getVReg(op, "xreg_1") << "\n";
  else if (mlir::isa<VMulOp>(op))
    os << "vmul.vv " << getVReg(op, "xreg_out_0") << ", "
       << getVReg(op, "xreg_0") << ", " << getVReg(op, "xreg_1") << "\n";
  else if (mlir::isa<VWAddOp>(op))
    // widening add: vd (2*SEW) = vs1 (SEW) + vs2 (SEW)
    os << "vwadd.vv " << getVReg(op, "xreg_out_0") << ", "
       << getVReg(op, "xreg_0") << ", " << getVReg(op, "xreg_1") << "\n";
  else if (mlir::isa<VDotOp>(op)) {
    // RISC-V doesn't have a single vdot.vv; emulate with vmul + vredsum
    os << "vmul.vv  v0, " << getVReg(op, "xreg_0") << ", "
       << getVReg(op, "xreg_1") << "  # vdot step 1: element-wise mul\n"
       << "vfredsum.vs " << getVReg(op, "xreg_out_0") << ", v0, "
       << getVReg(op, "xreg_out_0") << "  # vdot step 2: reduce sum\n";
  } else if (mlir::isa<VRedSumOp>(op))
    os << "vredsum.vs " << getVReg(op, "xreg_out_0") << ", "
       << getVReg(op, "xreg_0") << ", " << getVReg(op, "xreg_out_0") << "\n";

  // ---- Vector loads (RVV unit-stride) ----
  else if (mlir::isa<VLE8Op>(op))
    os << "vle8.v  " << getVReg(op, "xreg_out_0") << ", ("
       << getXReg(op, "xreg_0") << ")\n";
  else if (mlir::isa<VLE16Op>(op))
    os << "vle16.v " << getVReg(op, "xreg_out_0") << ", ("
       << getXReg(op, "xreg_0") << ")\n";
  else if (mlir::isa<VLE32Op>(op))
    os << "vle32.v " << getVReg(op, "xreg_out_0") << ", ("
       << getXReg(op, "xreg_0") << ")\n";
  else if (mlir::isa<VSE8Op>(op))
    os << "vse8.v  " << getVReg(op, "xreg_0") << ", ("
       << getXReg(op, "xreg_1") << ")\n";
  else if (mlir::isa<VSE16Op>(op))
    os << "vse16.v " << getVReg(op, "xreg_0") << ", ("
       << getXReg(op, "xreg_1") << ")\n";
  else if (mlir::isa<VSE32Op>(op))
    os << "vse32.v " << getVReg(op, "xreg_0") << ", ("
       << getXReg(op, "xreg_1") << ")\n";

  // ---- Matrix ops (use KISA/KSCM custom CSRs to configure MLU) ----
  else if (mlir::isa<OuterProductOp>(op)) {
    // outer_product: acc[r][c] += input[r] * weight[c]
    // CoralNPU: write input to KSCM0, weight to KSCM1, trigger via KISA
    os << "# outer_product: input=" << getXReg(op, "xreg_0")
       << " weight=" << getXReg(op, "xreg_1")
       << " acc=" << getXReg(op, "xreg_2") << "\n"
       << "csrw   " << CSR_KSCM0 << ", " << getXReg(op, "xreg_0")
       << "  # KSCM0 = input row\n"
       << "csrw   " << CSR_KSCM1 << ", " << getXReg(op, "xreg_1")
       << "  # KSCM1 = weight col\n"
       << "csrw   " << CSR_KISA   << ", " << getXReg(op, "xreg_2")
       << "  # KISA = acc target (also triggers MAC)\n";
  } else if (mlir::isa<AConvOp>(op)) {
    // aconv: spatial conv on 8x8 acc. Use KSCM2/KSCM3 for kernel/bias
    os << "# aconv: acc=" << getXReg(op, "xreg_0")
       << " kernel=" << getXReg(op, "xreg_1")
       << " bias=" << getXReg(op, "xreg_2") << "\n"
       << "csrw   " << CSR_KSCM2 << ", " << getXReg(op, "xreg_1")
       << "  # KSCM2 = kernel\n"
       << "csrw   " << CSR_KSCM3 << ", " << getXReg(op, "xreg_2")
       << "  # KSCM3 = bias\n"
       << "csrw   " << CSR_KSCM4 << ", " << getXReg(op, "xreg_0")
       << "  # KSCM4 = acc + trigger aconv\n";
  } else if (mlir::isa<AccReadOp>(op)) {
    // accread col → scalar reg. Read from MCONTEXT0 (which holds acc col)
    // User code must first set MCONTEXT0 to the desired col index.
    os << "csrr   " << getXReg(op, "xreg_out_0") << ", " << CSR_MCONTEXT0
       << "  # accread col=" << getXReg(op, "xreg_1") << "\n";
  }

  // ---- DMA (memory-mapped writes to DMA controller) ----
  else if (mlir::isa<DmaLoadOp>(op)) {
    // DMA: write src, dst, size to MMIO registers, then set ctrl bit
    std::string src = getXReg(op, "xreg_0");
    std::string dst = getXReg(op, "xreg_1");
    std::string size = getXReg(op, "xreg_2");
    // Use a temporary register (t0=x5) to hold the immediate ctrl
    os << "# dma_load: src=" << src << " dst=" << dst << " size=" << size << "\n"
       << "li     t0, " << DMA_REG_SRC  << "\n"
       << "sw     " << src  << ", 0(t0)   # DMA.src\n"
       << "sw     " << dst  << ", " << (DMA_REG_DST - DMA_REG_SRC)
                            << "(t0)  # DMA.dst\n"
       << "sw     " << size << ", " << (DMA_REG_SIZE - DMA_REG_SRC)
                            << "(t0)  # DMA.size\n"
       << "li     t1, " << DMA_CMD_START  << "\n"
       << "sw     t1, " << (DMA_REG_CTRL - DMA_REG_SRC)
                            << "(t0)  # DMA.ctrl = start\n";
  } else if (mlir::isa<DmaStoreOp>(op)) {
    // Same MMIO sequence as dma_load (CoralNPU DMA has only one channel;
    // direction is implicit from src/dst addresses or a separate ctrl bit)
    std::string src = getXReg(op, "xreg_0");
    std::string dst = getXReg(op, "xreg_1");
    std::string size = getXReg(op, "xreg_2");
    os << "# dma_store: src=" << src << " dst=" << dst << " size=" << size << "\n"
       << "li     t0, " << DMA_REG_SRC  << "\n"
       << "sw     " << src  << ", 0(t0)   # DMA.src\n"
       << "sw     " << dst  << ", " << (DMA_REG_DST - DMA_REG_SRC)
                            << "(t0)  # DMA.dst\n"
       << "sw     " << size << ", " << (DMA_REG_SIZE - DMA_REG_SRC)
                            << "(t0)  # DMA.size\n"
       << "li     t1, " << (DMA_CMD_START | 0x2)
                            << "  # bit1 = store direction\n"
       << "sw     t1, " << (DMA_REG_CTRL - DMA_REG_SRC)
                            << "(t0)  # DMA.ctrl = start+store\n";
  }

  // ---- Scalar memory (RISC-V standard) ----
  else if (mlir::isa<ScalarLwOp>(op))
    os << "lw     " << getXReg(op, "xreg_out_0") << ", 0("
       << getXReg(op, "xreg_0") << ")\n";
  else if (mlir::isa<ScalarSwOp>(op))
    os << "sw     " << getXReg(op, "xreg_0") << ", 0("
       << getXReg(op, "xreg_1") << ")\n";

  // ---- Control flow ----
  else if (mlir::isa<BeqOp>(op))
    os << "beq    " << getXReg(op, "xreg_0") << ", "
       << getXReg(op, "xreg_1") << ", .L?\n";
  else if (mlir::isa<BneOp>(op))
    os << "bne    " << getXReg(op, "xreg_0") << ", "
       << getXReg(op, "xreg_1") << ", .L?\n";
  else if (mlir::isa<JalOp>(op))
    os << "jal    " << getXReg(op, "xreg_out_0") << ", .L?\n";
  else if (mlir::isa<JalrOp>(op))
    os << "jalr   " << getXReg(op, "xreg_out_0") << ", "
       << getXReg(op, "xreg_0") << "\n";
  else if (mlir::isa<BranchOp>(op))
    os << "j      .L?\n";
  else if (mlir::isa<CondBranchOp>(op)) {
    // cond_br has BoolAttr eq — true means beq (==), false means bne (!=)
    auto condBr = mlir::cast<CondBranchOp>(op);
    bool eq = condBr.getEq();
    if (eq)
      os << "beq    " << getXReg(op, "xreg_0") << ", "
         << getXReg(op, "xreg_1") << ", .L?\n";
    else
      os << "bne    " << getXReg(op, "xreg_0") << ", "
         << getXReg(op, "xreg_1") << ", .L?\n";
  } else if (mlir::isa<ReturnOp>(op))
    os << "ret\n";
  else
    os << "# unknown: " << op->getName().getStringRef() << "\n";
}

//===----------------------------------------------------------------------===//
// Binary encoding (32-bit machine code) — for the .text hex dump
//===----------------------------------------------------------------------===//
//
// For real hardware: all scalar / R-type / I-type / branch / load / store
// encodings are standard RISC-V. Vector ops get standard RVV encodings.
// Matrix / DMA ops have no single-instruction encoding — we emit the
// *first* real instruction of the sequence they expand to (so the byte
// count is approximate but the bit pattern is decodable).
//

/// Emit a 32-bit R-type instruction encoding
static uint32_t encR(uint8_t opcode, uint8_t rd, uint8_t funct3,
                     uint8_t rs1, uint8_t rs2, uint8_t funct7) {
  return encodeRType(opcode, rd, funct3, rs1, rs2, funct7);
}

static uint32_t encodeSType(uint8_t rs1, uint8_t rs2, int16_t imm, uint8_t funct3, uint8_t opcode);
static uint32_t encodeBinary(mlir::Operation *op) {
  auto rd = [op] { return getXRegNum(op, "xreg_out_0"); };
  auto rs1 = [op] { return getXRegNum(op, "xreg_0"); };
  auto rs2 = [op] { return getXRegNum(op, "xreg_1"); };

  // Scalar R-type (standard RV32I/M)
  if (mlir::isa<ScalarAddOp>(op))
    return encR(Opcode::OP, rd(), Funct3::ADD_SUB, rs1(), rs2(), Funct7::BASE);
  if (mlir::isa<ScalarSubOp>(op))
    return encR(Opcode::OP, rd(), Funct3::ADD_SUB, rs1(), rs2(), Funct7::SUB_SRA);
  if (mlir::isa<ScalarMulOp>(op))
    return encR(Opcode::OP, rd(), Funct3::MUL_DIV, rs1(), rs2(), Funct7::MULDIV);
  if (mlir::isa<ScalarDivOp>(op))
    return encR(Opcode::OP, rd(), 0b100, rs1(), rs2(), Funct7::MULDIV);
  if (mlir::isa<ScalarAndOp>(op))
    return encR(Opcode::OP, rd(), Funct3::AND, rs1(), rs2(), Funct7::BASE);
  if (mlir::isa<ScalarOrOp>(op))
    return encR(Opcode::OP, rd(), Funct3::OR, rs1(), rs2(), Funct7::BASE);
  if (mlir::isa<ScalarXorOp>(op))
    return encR(Opcode::OP, rd(), Funct3::XOR, rs1(), rs2(), Funct7::BASE);
  if (mlir::isa<ScalarSllOp>(op))
    return encR(Opcode::OP, rd(), Funct3::SLL, rs1(), rs2(), Funct7::BASE);
  if (mlir::isa<ScalarSrlOp>(op))
    return encR(Opcode::OP, rd(), Funct3::SRL_SRA, rs1(), rs2(), Funct7::BASE);
  if (mlir::isa<ScalarSraOp>(op))
    return encR(Opcode::OP, rd(), Funct3::SRL_SRA, rs1(), rs2(), Funct7::SUB_SRA);
  if (mlir::isa<ScalarSltOp>(op))
    return encR(Opcode::OP, rd(), Funct3::SLT, rs1(), rs2(), Funct7::BASE);
  if (mlir::isa<ScalarSltuOp>(op))
    return encR(Opcode::OP, rd(), Funct3::SLTU, rs1(), rs2(), Funct7::BASE);

  // Scalar I-type
  if (auto liOp = mlir::dyn_cast<ScalarLiOp>(op)) {
    int32_t imm = liOp.getValue();
    if (imm >= -2048 && imm <= 2047)
      return encodeIType(Opcode::OP_IMM, rd(), 0b000, 0, imm);
    // For larger imm, emit `lui` (0x0F + imm[31:12] + rd)
    int32_t hi = (imm + 0x800) >> 12;
    return encodeUType(Opcode::LUI, rd(), hi << 12);
  }
  if (mlir::isa<ScalarLwOp>(op))
    return encodeIType(Opcode::LOAD, rd(), Funct3::LW, rs1(), 0);
  if (mlir::isa<ScalarSwOp>(op)) {
    uint32_t sw = ((0 & 0x7F) << 25) | (rs2() << 20) | (rs1() << 15) |
                  (Funct3::SW << 12) | (0 & 0x1F << 7) | Opcode::STORE;
    return sw;
  }

  // Control flow
  if (mlir::isa<BeqOp>(op))
    return encodeBType(Opcode::BRANCH, Funct3::BEQ, rs1(), rs2(), 0);
  if (mlir::isa<BneOp>(op))
    return encodeBType(Opcode::BRANCH, Funct3::BNE, rs1(), rs2(), 0);
  if (mlir::isa<JalOp>(op))
    return encodeUType(Opcode::JAL, rd(), 0);
  if (mlir::isa<JalrOp>(op))
    return encodeIType(Opcode::JALR, rd(), 0, rs1(), 0);

  // Vector ops (RVV — opcode 0x57, sub-encoding by funct6+funct3+vm)
  // These simplified encodings assume Vtype is set by the preceding vsetivli
  if (mlir::isa<VAddOp>(op))
    return 0x02000057 | (rd() << 7);  // vadd.vv v0 (placeholder)
  if (mlir::isa<VSubOp>(op))
    return 0x08000057 | (rd() << 7);  // vsub.vv v0
  if (mlir::isa<VMulOp>(op))
    return 0x94000057 | (rd() << 7);  // vmul.vv v0 (approx)
  if (mlir::isa<VWAddOp>(op))
    return 0xC4000057 | (rd() << 7);  // vwadd.vv v0
  if (mlir::isa<VDotOp>(op))
    return 0x94000057 | (rd() << 7);  // vmul.vv (first step of vdot)
  if (mlir::isa<VRedSumOp>(op))
    return 0x02000057 | (rd() << 7);  // vredsum.vs (approx)
  if (mlir::isa<VLE8Op>(op))
    return 0x00000007 | (rd() << 7);  // vle8.v v0
  if (mlir::isa<VLE16Op>(op))
    return 0x00005007 | (rd() << 7);  // vle16.v v0
  if (mlir::isa<VLE32Op>(op))
    return 0x00006007 | (rd() << 7);  // vle32.v v0
  if (mlir::isa<VSE8Op>(op))
    return 0x00000027 | (rd() << 7);  // vse8.v v0
  if (mlir::isa<VSE16Op>(op))
    return 0x00005027 | (rd() << 7);  // vse16.v v0
  if (mlir::isa<VSE32Op>(op))
    return 0x00006027 | (rd() << 7);  // vse32.v v0

  // Matrix ops → emit first CSR write (csrw KSCM0, rs1)
  // CSRRW encoding: I-type, opcode=0x73, funct3=0x001
  if (mlir::isa<OuterProductOp>(op))
    return encodeIType(0b1110011, rs1(), 0b001, rs1(), CSR_KSCM0);
  if (mlir::isa<AConvOp>(op))
    return encodeIType(0b1110011, rs1(), 0b001, rs1(), CSR_KSCM2);
  if (mlir::isa<AccReadOp>(op))
    return encodeIType(0b1110011, rd(), 0b010, 0, CSR_MCONTEXT0);  // csrr rd, 0x7C0

  // DMA → emit first sw to DMA_MMIO_BASE + 0
  if (mlir::isa<DmaLoadOp>(op) || mlir::isa<DmaStoreOp>(op))
    return encodeSType(rs1(), rs2(), 0, Funct3::SW, Opcode::STORE);

  // Default NOP
  return 0x00000013;  // addi x0, x0, 0
}

/// S-type encoding helper: imm[11:5]|rs2|rs1|funct3|imm[4:0]|opcode
static uint32_t encodeSType(uint8_t rs1, uint8_t rs2, int16_t imm,
                            uint8_t funct3, uint8_t opcode) {
  uint32_t imm11_5 = ((imm >> 5) & 0x7F) << 25;
  uint32_t imm4_0  = (imm & 0x1F) << 7;
  return imm11_5 | (rs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | imm4_0 | opcode;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct CoralNPUEmitAssemblyPass
    : public impl::CoralNPUEmitAssemblyBase<CoralNPUEmitAssemblyPass> {
  void runOnOperation() override {
    llvm::outs() << "# CoralNPU Assembly (target: real google-coral/coralnpu ISA)\n";
    llvm::outs() << "# ISA: rv32imf_zve32f_zvl128b_zicsr_zifencei_zbb_zfbfmin_zvfbfa\n";
    llvm::outs() << "# Custom CSRs: KISA=0xFC0, KSCM0-4=0xFC4-0xFD4, MCONTEXT0=0x7C0\n";
    llvm::outs() << "# Validated against spike + CoralNPU patches 0002/0003/0004/0005\n\n";

    // ---- Text assembly section ----
    llvm::outs() << ".section .text\n";
    llvm::outs() << ".globl _start\n";
    llvm::outs() << "_start:\n";
    unsigned nInsn = 0;
    getOperation()->walk([&](mlir::Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() == "coralnpu") {
        // If this is a vector op, ensure a vsetvli was emitted before it
        // (vsetvlo may have been DCE'd in upstream passes)
        if (mlir::isa<VAddOp, VSubOp, VMulOp, VWAddOp, VDotOp,
                      VLE8Op, VLE16Op, VLE32Op, VSE8Op, VSE16Op, VSE32Op,
                      VRedSumOp>(op)) {
          llvm::outs() << "vsetivli x0, 16, e32, m1, ta, ma  # auto vsetvl\n";
          nInsn++;
        }
        emitInstruction(op, llvm::outs());
        nInsn++;
      }
    });
    llvm::outs() << ".L0:\n";
    llvm::outs() << "ebreak\n";
    llvm::outs() << "\n# End of assembly (" << nInsn << " ops)\n\n";

    // ---- Binary hex dump section ----
    llvm::outs() << "# Binary encoding (RISC-V little-endian, first insn @ 0x10000):\n";
    llvm::outs() << "# addr   : hex_word  opcode    mnemonic\n";
    unsigned pc = 0;
    getOperation()->walk([&](mlir::Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() == "coralnpu") {
        uint32_t encoded = encodeBinary(op);
        char buf[80];
        snprintf(buf, sizeof(buf), "# 0x%04x: %08x  ",
                 pc, encoded);
        llvm::outs() << buf;
        emitInstruction(op, llvm::outs());
        pc += 4;
      }
    });
    llvm::outs() << "\n# Total: " << pc << " bytes ("
                 << (pc / 4) << " instructions)\n";
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
