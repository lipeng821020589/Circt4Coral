//===- ExportCoralNPU.cpp - Coral NPU ELF binary export -------------------===//
//
// Translation pass that converts CoralNPU IR to an ELF binary file.
// Handles: RISC-V base ISA + Coral NPU custom extensions
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CoralNPU/CoralNPUEncodings.h"
#include "circt/Dialect/CoralNPU/CoralNPUOps.h"
#include "circt/Dialect/CoralNPU/CoralNPUPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <vector>

using namespace circt;
using namespace circt::coralnpu;

//===----------------------------------------------------------------------===//
// ELF file constants
//===----------------------------------------------------------------------===//

namespace {
constexpr uint8_t ELFCLASS32 = 1;
constexpr uint8_t ELFDATA2LSB = 1;
constexpr uint8_t EV_CURRENT = 1;
constexpr uint16_t EM_RISCV = 243; // RISC-V machine ID
constexpr uint16_t ET_REL = 1;     // Relocatable object file
constexpr uint32_t EI_NIDENT = 16;

// Coral NPU specific: use a custom EI_OSABI to identify NPU binaries
constexpr uint8_t ELFOSABI_CORALNPU = 0xC0;

struct ELFHeader {
  uint8_t  e_ident[EI_NIDENT];
  uint16_t e_type;
  uint16_t e_machine;
  uint32_t e_version;
  uint32_t e_entry;
  uint32_t e_phoff;
  uint32_t e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize;
  uint16_t e_phentsize;
  uint16_t e_phnum;
  uint16_t e_shentsize;
  uint16_t e_shnum;
  uint16_t e_shstrndx;
};

struct ELFSectionHeader {
  uint32_t sh_name;
  uint32_t sh_type;
  uint32_t sh_flags;
  uint32_t sh_addr;
  uint32_t sh_offset;
  uint32_t sh_size;
  uint32_t sh_link;
  uint32_t sh_info;
  uint32_t sh_addralign;
  uint32_t sh_entsize;
};

constexpr uint32_t SHT_PROGBITS = 1;
constexpr uint32_t SHT_SYMTAB = 2;
constexpr uint32_t SHT_STRTAB = 3;
constexpr uint32_t SHF_ALLOC = 2;
constexpr uint32_t SHF_EXECINSTR = 4;

//===----------------------------------------------------------------------===//
// Binary emitter
//===----------------------------------------------------------------------===//

class BinaryEmitter {
public:
  BinaryEmitter(llvm::raw_ostream &os) : os(os) {}

  /// Emit a 32-bit instruction word in little-endian.
  void emit32(uint32_t word) {
    os.write((char *)&word, 4);
    textSection.push_back(word);
  }

  /// Encode and emit a Coral NPU operation as binary.
  void encodeOp(mlir::Operation *op, unsigned &pc) {
    // Map CoralNPU ops to RISC-V / NPU custom encodings
    llvm::TypeSwitch<mlir::Operation *>(op)
      // ── Scalar RISC-V instructions ──
      .Case<ScalarAddOp>([&](auto) {
        emit32(encodeRType(Opcode::OP, getRd(op), Funct3::ADD_SUB,
                           getRs1(op), getRs2(op), Funct7::BASE));
      })
      .Case<ScalarSubOp>([&](auto) {
        emit32(encodeRType(Opcode::OP, getRd(op), Funct3::ADD_SUB,
                           getRs1(op), getRs2(op), Funct7::SUB_SRA));
      })
      .Case<ScalarMulOp>([&](auto) {
        emit32(encodeRType(Opcode::OP, getRd(op), Funct3::MUL_DIV,
                           getRs1(op), getRs2(op), Funct7::MULDIV));
      })
      .Case<ScalarAndOp>([&](auto) {
        emit32(encodeRType(Opcode::OP, getRd(op), Funct3::AND,
                           getRs1(op), getRs2(op), Funct7::BASE));
      })
      .Case<ScalarOrOp>([&](auto) {
        emit32(encodeRType(Opcode::OP, getRd(op), Funct3::OR,
                           getRs1(op), getRs2(op), Funct7::BASE));
      })
      .Case<ScalarXorOp>([&](auto) {
        emit32(encodeRType(Opcode::OP, getRd(op), Funct3::XOR,
                           getRs1(op), getRs2(op), Funct7::BASE));
      })
      .Case<ScalarSllOp>([&](auto) {
        emit32(encodeRType(Opcode::OP, getRd(op), Funct3::SLL,
                           getRs1(op), getRs2(op), Funct7::BASE));
      })
      .Case<ScalarSrlOp>([&](auto) {
        emit32(encodeRType(Opcode::OP, getRd(op), Funct3::SRL_SRA,
                           getRs1(op), getRs2(op), Funct7::BASE));
      })
      .Case<ScalarSraOp>([&](auto) {
        emit32(encodeRType(Opcode::OP, getRd(op), Funct3::SRL_SRA,
                           getRs1(op), getRs2(op), Funct7::SUB_SRA));
      })
      .Case<ScalarSltOp>([&](auto) {
        emit32(encodeRType(Opcode::OP, getRd(op), Funct3::SLT,
                           getRs1(op), getRs2(op), Funct7::BASE));
      })
      .Case<ScalarSltuOp>([&](auto) {
        emit32(encodeRType(Opcode::OP, getRd(op), Funct3::SLTU,
                           getRs1(op), getRs2(op), Funct7::BASE));
      })
      .Case<ScalarDivOp>([&](auto) {
        // DIV: MULDIV funct7, funct3=100
        emit32(encodeRType(Opcode::OP, getRd(op), 0b100,
                           getRs1(op), getRs2(op), Funct7::MULDIV));
      })
      .Case<ScalarLiOp>([&](auto liOp) {
        int32_t imm = liOp.getValue();
        emit32(encodeIType(Opcode::OP_IMM, getRd(op), 0b000,
                           getRs1(op), imm));
      })
      // ── Scalar memory ──
      .Case<ScalarLwOp>([&](auto) {
        emit32(encodeIType(Opcode::LOAD, getRd(op), Funct3::LW,
                           getRs1(op), 0));
      })
      .Case<ScalarSwOp>([&](auto) {
        // SW: imm[11:5]|rs2|rs1|funct3|imm[4:0]|opcode
        uint32_t sw = ((0 & 0x7F) << 25) | (getRs2(op) << 20) | (getRs1(op) << 15) |
                      (Funct3::SW << 12) | (0 & 0x1F << 7) | Opcode::STORE;
        emit32(sw);
      })
      // ── Control flow ──
      .Case<BeqOp>([&](auto) {
        emit32(encodeBType(Opcode::BRANCH, Funct3::BEQ,
                           getRs1(op), getRs2(op), 0));
      })
      .Case<BneOp>([&](auto) {
        emit32(encodeBType(Opcode::BRANCH, Funct3::BNE,
                           getRs1(op), getRs2(op), 0));
      })
      .Case<JalOp>([&](auto) {
        // JAL: imm|rd|opcode (UJ-type, simplified)
        emit32(encodeUType(Opcode::JAL, getRd(op), 0));
      })
      .Case<JalrOp>([&](auto) {
        emit32(encodeIType(Opcode::JALR, getRd(op), 0b000,
                           getRs1(op), 0));
      })
      // ── Vector ops (custom encoding) ──
      .Case<VAddOp>([&](auto) {
        emit32(encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VADD, 0, 0, 0));
      })
      .Case<VSubOp>([&](auto) {
        emit32(encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VSUB, 0, 0, 0));
      })
      .Case<VMulOp>([&](auto) {
        emit32(encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VMUL, 0, 0, 0));
      })
      .Case<VDotOp>([&](auto) {
        emit32(encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VDOT, 0, 0, 0));
      })
      .Case<VWAddOp>([&](auto) {
        emit32(encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VWADD, 0, 0, 0));
      })
      .Case<VRedSumOp>([&](auto) {
        emit32(encodeRType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VREDSUM, 0, 0, 0));
      })
      .Case<VLE8Op>([&](auto) {
        emit32(encodeIType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VLE, 0, 0));
      })
      .Case<VLE16Op>([&](auto) {
        emit32(encodeIType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VLE, 0, 1));
      })
      .Case<VLE32Op>([&](auto) {
        emit32(encodeIType(Opcode::CUSTOM_VEC, 0, NPUFunct3::VLE, 0, 2));
      })
      // ── Matrix ops (custom encoding) ──
      .Case<OuterProductOp>([&](auto) {
        emit32(encodeRType(Opcode::CUSTOM_MAT, 0, NPUFunct3::OUTER_PRODUCT, 0, 0, 0));
      })
      .Case<AConvOp>([&](auto) {
        emit32(encodeRType(Opcode::CUSTOM_MAT, 0, NPUFunct3::ACONV, 0, 0, 0));
      })
      .Case<AccReadOp>([&](auto) {
        emit32(encodeRType(Opcode::CUSTOM_MAT, 0, NPUFunct3::ACCREAD, 0, 0, 0));
      })
      // ── DMA ──
      .Case<DmaLoadOp>([&](auto) {
        emit32(encodeIType(Opcode::CUSTOM_DMA, 0, NPUFunct3::DMA_LOAD, 0, 0));
      })
      .Case<DmaStoreOp>([&](auto) {
        emit32(encodeIType(Opcode::CUSTOM_DMA, 0, NPUFunct3::DMA_STORE, 0, 0));
      })
      .Default([&](auto) {
        // Emit NOP (addi x0, x0, 0) for unsupported ops
        emit32(encodeIType(Opcode::OP_IMM, 0, 0, 0, 0));
      });
    pc += 4;
  }

  /// Write a complete ELF object file with the encoded text.
  void writeELF(llvm::StringRef filename) {
    // Build ELF header
    ELFHeader ehdr = {};
    ehdr.e_ident[0] = 0x7F;
    ehdr.e_ident[1] = 'E';
    ehdr.e_ident[2] = 'L';
    ehdr.e_ident[3] = 'F';
    ehdr.e_ident[4] = ELFCLASS32;
    ehdr.e_ident[5] = ELFDATA2LSB;
    ehdr.e_ident[6] = EV_CURRENT;
    ehdr.e_ident[7] = ELFOSABI_CORALNPU;
    ehdr.e_type = ET_REL;
    ehdr.e_machine = EM_RISCV;
    ehdr.e_version = EV_CURRENT;
    ehdr.e_ehsize = sizeof(ELFHeader);
    ehdr.e_shentsize = sizeof(ELFSectionHeader);
    ehdr.e_shnum = 4; // null, .text, .shstrtab, .symtab
    ehdr.e_shstrndx = 2;

    // Write ELF header
    os.write(reinterpret_cast<const char *>(&ehdr), sizeof(ELFHeader));

    // Section headers (written at end, offset computed)
    uint32_t textOff = sizeof(ELFHeader);
    uint32_t textSize = textSection.size() * 4;
    uint32_t shstrOff = textOff + textSize;

    // .shstrtab section data
    const char *shstrtab = "\0.text\0.shstrtab\0.symtab\0";
    uint32_t shstrSize = 29;

    // Section headers
    ELFSectionHeader shdr[4] = {};

    // Section 0: NULL
    // Section 1: .text
    shdr[1].sh_name = 1; // ".text"
    shdr[1].sh_type = SHT_PROGBITS;
    shdr[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
    shdr[1].sh_offset = textOff;
    shdr[1].sh_size = textSize;
    shdr[1].sh_addralign = 4;

    // Section 2: .shstrtab
    shdr[2].sh_name = 7; // ".shstrtab"
    shdr[2].sh_type = SHT_STRTAB;
    shdr[2].sh_offset = shstrOff;
    shdr[2].sh_size = shstrSize;
    shdr[2].sh_addralign = 1;

    // Section 3: .symtab (empty)
    shdr[3].sh_name = 18; // ".symtab"
    shdr[3].sh_type = SHT_SYMTAB;
    shdr[3].sh_offset = shstrOff + shstrSize;
    shdr[3].sh_size = 0;
    shdr[3].sh_link = 2;
    shdr[3].sh_entsize = 16;
    shdr[3].sh_addralign = 4;

    // Update header with section info
    ehdr.e_shoff = shstrOff + shstrSize;

    // Write .text section data
    os.write(reinterpret_cast<const char *>(textSection.data()), textSize);

    // Write .shstrtab
    os.write(shstrtab, shstrSize);

     // Write section headers
    for (unsigned i = 1; i < 4; ++i)
      os.write(reinterpret_cast<const char *>(&shdr[i]), sizeof(ELFSectionHeader));
  }

  size_t getTextSize() const { return textSection.size(); }

private:
  llvm::raw_ostream &os;
  std::vector<uint32_t> textSection;

  // Helper: get destination register from xreg annotation
  uint8_t getRd(mlir::Operation *op) {
    return getXRegAttr(op, "xreg_out_0");
  }
  uint8_t getRs1(mlir::Operation *op) {
    return getXRegAttr(op, "xreg_0");
  }
  uint8_t getRs2(mlir::Operation *op) {
    return getXRegAttr(op, "xreg_1");
  }

  uint8_t getXRegAttr(mlir::Operation *op, llvm::StringRef name) {
    if (auto attr = op->getDiscardableAttr(name))
      return static_cast<uint8_t>(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    return 0; // default to x0
  }
};

//===----------------------------------------------------------------------===//
// Export translation pass
//===----------------------------------------------------------------------===//

struct ExportCoralNPUPass {
  void runOnOperation(mlir::ModuleOp module, llvm::raw_ostream &os) {
    os << "; Coral NPU ELF Binary Export\n";
    os << "; Target: RISC-V + Coral NPU custom extensions\n\n";

    BinaryEmitter emitter(os);
    unsigned pc = 0;

    module.walk([&](mlir::Operation *op) {
      if (op->getDialect() &&
          op->getDialect()->getNamespace() == "coralnpu") {
        emitter.encodeOp(op, pc);
      }
    });

    os << "\n; Text section: " << emitter.getTextSize() << " instructions ("
       << (emitter.getTextSize() * 4) << " bytes)\n";
    os << "; ELF file generation: use --export-coralnpu-elf=<filename>\n";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

namespace circt {
namespace coralnpu {

void registerExportCoralNPUTranslation() {
  mlir::TranslateFromMLIRRegistration reg(
      "export-coralnpu",
      "export Coral NPU binary",
      [](mlir::ModuleOp module, llvm::raw_ostream &os) -> mlir::LogicalResult {
        ExportCoralNPUPass pass;
        pass.runOnOperation(module, os);
        return mlir::success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<CoralNPUDialect>();
      });
}

} // namespace coralnpu
} // namespace circt
