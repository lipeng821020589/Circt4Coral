# CMakeLists.txt for CoralNPU dialect library

add_circt_dialect_library(CIRCTCoralNPU
  CoralNPUDialect.cpp
  CoralNPUOps.cpp
  CoralNPUTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/CoralNPU

  DEPENDS
  MLIRCoralNPUAttrDefsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

add_subdirectory(Transforms)
