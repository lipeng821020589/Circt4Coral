//===- CoralNPUEncodings.h - Coral NPU instruction encodings -----*- C++ -*-===//
//
// RISC-V compatible instruction encodings for Coral NPU.
// Coral NPU recycles the C-extension encoding space for custom SIMD/Matrix ops.
//
// Base RISC-V opcodes:
//   R-type:  0110011  (funct7|rs2|rs1|funct3|rd|opcode)
//   I-type:  0010011  (imm[11:0]|rs1|funct3|rd|opcode)
//   Load:    0000011
//   Store:   0100011
//   Branch:  1100011
//   JAL:     1101111
//   JALR:    1100111
//   LUI:     0110111
//
// Coral NPU custom opcodes (reuse C-extension space 00/01/10):
//   V-ops:   custom-0 (0001011) with funct3/funct7 differentiation
//   M-ops:   custom-1 (0101011) for matrix (outer_product, aconv)
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CORALNPU_CORALNPUENCODINGS_H
#define CIRCT_DIALECT_CORALNPU_CORALNPUENCODINGS_H

#include <cstdint>
#include <string>

namespace llvm {
class StringRef;
}

namespace circt {
namespace coralnpu {

//===----------------------------------------------------------------------===//
// RISC-V Base Opcodes
//===----------------------------------------------------------------------===//

namespace Opcode {
constexpr uint8_t LOAD    = 0b0000011;
constexpr uint8_t STORE   = 0b0100011;
constexpr uint8_t BRANCH  = 0b1100011;
constexpr uint8_t JALR    = 0b1100111;
constexpr uint8_t JAL     = 0b1101111;
constexpr uint8_t OP_IMM  = 0b0010011;
constexpr uint8_t OP      = 0b0110011;
constexpr uint8_t LUI     = 0b0110111;
constexpr uint8_t AUIPC   = 0b0010111;

// Coral NPU custom opcodes (reuse reserved C-extension space)
constexpr uint8_t CUSTOM_VEC   = 0b0001011; // Vector/SIMD ops
constexpr uint8_t CUSTOM_MAT   = 0b0101011; // Matrix ops
constexpr uint8_t CUSTOM_DMA   = 0b1101011; // DMA ops
} // namespace Opcode

//===----------------------------------------------------------------------===//
// RISC-V funct3 values
//===----------------------------------------------------------------------===//

namespace Funct3 {
constexpr uint8_t ADD_SUB = 0b000;
constexpr uint8_t SLL     = 0b001;
constexpr uint8_t SLT     = 0b010;
constexpr uint8_t SLTU    = 0b011;
constexpr uint8_t XOR     = 0b100;
constexpr uint8_t SRL_SRA = 0b101;
constexpr uint8_t OR      = 0b110;
constexpr uint8_t AND     = 0b111;
constexpr uint8_t MUL_DIV = 0b000; // funct7 distinguishes
constexpr uint8_t BEQ     = 0b000;
constexpr uint8_t BNE     = 0b001;
constexpr uint8_t LW      = 0b010;
constexpr uint8_t SW      = 0b010;
} // namespace Funct3

//===----------------------------------------------------------------------===//
// RISC-V funct7 values
//===----------------------------------------------------------------------===//

namespace Funct7 {
constexpr uint8_t BASE    = 0b0000000;
constexpr uint8_t SUB_SRA = 0b0100000;
constexpr uint8_t MULDIV  = 0b0000001;
} // namespace Funct7

//===----------------------------------------------------------------------===//
// Coral NPU custom funct3 values (for custom opcodes)
//===----------------------------------------------------------------------===//

namespace NPUFunct3 {
// Vector ops (CUSTOM_VEC)
constexpr uint8_t VADD   = 0b000;
constexpr uint8_t VSUB   = 0b001;
constexpr uint8_t VMUL   = 0b010;
constexpr uint8_t VDOT   = 0b011;
constexpr uint8_t VWADD  = 0b100;
constexpr uint8_t VREDSUM = 0b101;
constexpr uint8_t VLE    = 0b000; // funct7: 0=8b, 1=16b, 2=32b
constexpr uint8_t VSE    = 0b001; // funct7: 0=8b, 1=16b, 2=32b

// Matrix ops (CUSTOM_MAT)
constexpr uint8_t OUTER_PRODUCT = 0b000;
constexpr uint8_t ACONV         = 0b001;
constexpr uint8_t ACCREAD       = 0b010;

// DMA ops (CUSTOM_DMA)
constexpr uint8_t DMA_LOAD  = 0b000;
constexpr uint8_t DMA_STORE = 0b001;
} // namespace NPUFunct3

//===----------------------------------------------------------------------===//
// Instruction encoding helpers
//===----------------------------------------------------------------------===//

/// Encode an R-type instruction: funct7|rs2|rs1|funct3|rd|opcode
inline uint32_t encodeRType(uint8_t opcode, uint8_t rd, uint8_t funct3,
                            uint8_t rs1, uint8_t rs2, uint8_t funct7) {
  return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | (rd << 7) | opcode;
}

/// Encode an I-type instruction: imm[11:0]|rs1|funct3|rd|opcode
inline uint32_t encodeIType(uint8_t opcode, uint8_t rd, uint8_t funct3,
                            uint8_t rs1, int16_t imm) {
  return ((imm & 0xFFF) << 20) | (rs1 << 15) |
         (funct3 << 12) | (rd << 7) | opcode;
}

/// Encode a U-type instruction: imm[31:12]|rd|opcode
inline uint32_t encodeUType(uint8_t opcode, uint8_t rd, int32_t imm) {
  return (imm & 0xFFFFF000) | (rd << 7) | opcode;
}

/// Encode a branch instruction: imm|rs2|rs1|funct3|imm|opcode
inline uint32_t encodeBType(uint8_t opcode, uint8_t funct3,
                            uint8_t rs1, uint8_t rs2, int16_t offset) {
  uint32_t imm12 = (offset & 0x1000) >> 12;
  uint32_t imm10_5 = (offset & 0x7E0) >> 5;
  uint32_t imm4_1 = (offset & 0x1E) >> 1;
  uint32_t imm11 = (offset & 0x800) >> 11;
  return (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | (imm4_1 << 8) | (imm11 << 7) | opcode;
}

/// Get mnemonic string for a Coral NPU op name (for disassembly/metadata)
std::string getOpMnemonic(llvm::StringRef opName);

} // namespace coralnpu
} // namespace circt

#endif // CIRCT_DIALECT_CORALNPU_CORALNPUENCODINGS_H
