// Scalar benchmark: 16-element i32 add (returns last value to avoid DCE)
func.func @scalar_add16_bench() -> i32 {
  %a = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tensor<16xi32>
  %b = arith.constant dense<[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]> : tensor<16xi32>
  %c = tosa.add %a, %b : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
  %v = tensor.extract %c[%c0] : tensor<16xi32>
  return %v : i32
}
