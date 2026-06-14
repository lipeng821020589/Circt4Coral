// TOSA add (entry-point for ML/LLVM frontends)
func.func @tosa_add(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
