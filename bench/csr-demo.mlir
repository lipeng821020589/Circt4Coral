// Demonstrate CoralNPU custom CSR codegen
// These high-level coralnpu ops expand to csrw/csrr sequences
// targeting the custom KISA/KSCM/MCONTEXT0 CSRs.

// Test 1: outer_product → mul + csrw KSCM0
func.func @demo_outer_product() -> i32 {
  %a = coralnpu.li 2 : i32
  %b = coralnpu.li 3 : i32
  %acc = coralnpu.li 0 : i32
  %r = coralnpu.outer_product %a, %b, %acc : i32
  coralnpu.return %r : i32
}

// Test 2: 1D conv → csrw KSCM1+KSCM2
func.func @demo_aconv() -> i32 {
  %input = coralnpu.li 10 : i32
  %weight = coralnpu.li 1 : i32
  %acc = coralnpu.li 0 : i32
  %r = coralnpu.aconv %input, %weight, %acc : i32
  coralnpu.return %r : i32
}
