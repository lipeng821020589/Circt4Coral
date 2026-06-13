# CMakeLists.txt for CoralNPU transforms

add_circt_dialect_library(CIRCTCoralNPUTransforms
  TosaToCoralNPU.cpp
  CoralNPUStripmine.cpp
  CoralNPULegalize.cpp
  CoralNPURegAlloc.cpp
  CoralNPUEmitAssembly.cpp
  ExportCoralNPU.cpp
  CheckCoralNPUHW.cpp
  CoralNPUFormalVerify.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/CoralNPU

  DEPENDS
  CIRCTCoralNPUTransformsIncGen

  LINK_COMPONENTS
  Object
  Support

  LINK_LIBS PUBLIC
  CIRCTCoralNPU
  CIRCTHW
  CIRCTVerif
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  MLIRTranslateLib
  MLIRTransforms
  MLIRSupport
)
