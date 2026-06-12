# CMakeLists.txt for CoralNPU dialect headers

add_circt_dialect(CoralNPU coralnpu)

set(LLVM_TARGET_DEFINITIONS CoralNPUPasses.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTCoralNPUTransformsIncGen)

set(LLVM_TARGET_DEFINITIONS CoralNPU.td)
mlir_tablegen(CoralNPUAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=coralnpu)
mlir_tablegen(CoralNPUAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=coralnpu)
add_public_tablegen_target(MLIRCoralNPUAttrDefsIncGen)
