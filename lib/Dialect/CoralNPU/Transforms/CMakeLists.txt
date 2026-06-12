# CMakeLists.txt for CoralNPU transforms

add_circt_dialect_library(CIRCTCoralNPUTransforms
  TosaToCoralNPU.cpp
  CoralNPUStripmine.cpp
  CoralNPULegalize.cpp
  CoralNPUEmitAssembly.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/CoralNPU

  DEPENDS
  CIRCTCoralNPUTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTCoralNPU
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  MLIRTransforms
  MLIRSupport
)
