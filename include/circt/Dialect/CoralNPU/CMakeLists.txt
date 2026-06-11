# CMakeLists.txt for CoralNPU dialect headers

add_circt_dialect(CoralNPU coralnpu)

set(LLVM_TARGET_DEFINITIONS CoralNPU.td)
mlir_tablegen(CoralNPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(CoralNPUEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCoralNPUEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS CoralNPUPasses.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTCoralNPUTransformsIncGen)
