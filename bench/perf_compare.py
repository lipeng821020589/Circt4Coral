#!/usr/bin/env python3
"""
GEMV Performance Comparison: MAC CSR (int8) vs vfmacc i32
Measures assembly instruction count as proxy for compute efficiency.
"""
import subprocess, tempfile, os
from pathlib import Path

CIRCT_OPT = Path("/home/radxa/Work/Circt4Coral/build/bin/circt-opt")

GEMV_INT8 = """\
func.func @gemv_int8(%a: tensor<1x1x8xi8>, %b: tensor<1x8x8xi8>,
                     %az: tensor<1xi8>, %bz: tensor<1xi8>) -> tensor<1x1x8xi32> {
  %0 = tosa.matmul %a, %b, %az, %bz
    : (tensor<1x1x8xi8>, tensor<1x8x8xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x8xi32>
  func.return %0 : tensor<1x1x8xi32>
}
"""

GEMV_I32 = """\
func.func @gemv_i32(%a: tensor<1x1x8xi32>, %b: tensor<1x8x8xi32>,
                    %az: tensor<1xi32>, %bz: tensor<1xi32>) -> tensor<1x1x8xi32> {
  %0 = tosa.matmul %a, %b, %az, %bz
    : (tensor<1x1x8xi32>, tensor<1x8x8xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x8xi32>
  func.return %0 : tensor<1x1x8xi32>
}
"""

PASSES = ["--tosa-to-coralnpu", "--coralnpu-legalize",
          "--coralnpu-regalloc", "--emit-coralnpu-assembly"]

def compile_and_analyze(mlir_text, label):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_text); fname = f.name
    r = subprocess.run([str(CIRCT_OPT)] + PASSES + [fname],
                       capture_output=True, text=True)
    os.unlink(fname)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed for {label}:\n{r.stderr[:300]}")
    lines = [l.strip() for l in r.stdout.splitlines()
             if l.strip() and not l.strip().startswith('//') and
             not l.strip().startswith('module') and
             not l.strip().startswith('func.func') and
             not l.strip().startswith('}') and
             not l.strip().startswith('#')]
    # Count instructions that translate to hardware ops
    hw_ops = [l for l in lines if any(l.startswith(p) for p in
              ('csrw', 'csrr', 'vle32', 'vse32', 'vmul', 'vadd', 'vredsum',
               'vsetvl', 'addi', 'lui', 'li ', 'sw ', 'lw ', 'add ', 'mul',
               'ret', 'ebreak'))]
    return r.stdout, len(lines), len(hw_ops)

if __name__ == "__main__":
    print("=" * 62)
    print("GEMV 1×8×8  Compilation Efficiency Comparison")
    print("Coral NPU Circt4Coral compiler — Jul 2026")
    print("=" * 62)

    asm_mac, n_all_mac, n_hw_mac = compile_and_analyze(GEMV_INT8, "int8 MAC")
    asm_vec, n_all_vec, n_hw_vec = compile_and_analyze(GEMV_I32, "i32 vfmacc")

    print(f"\n{'Path':<28} {'Total IR lines':>14} {'HW instr':>10}")
    print("-" * 55)
    print(f"{'int8 → OuterProduct MAC CSR':<28} {n_all_mac:>14} {n_hw_mac:>10}")
    print(f"{'i32  → vle+vmul+vredsum':<28} {n_all_vec:>14} {n_hw_vec:>10}")
    print("-" * 55)
    if n_hw_mac > 0 and n_hw_vec > 0:
        ratio = n_hw_vec / n_hw_mac
        print(f"i32/MAC HW instruction ratio: {ratio:.1f}×  "
              f"({'MAC more compact' if ratio > 1 else 'vfmacc more compact'})")

    print("\nKey insight:")
    print("  MAC CSR path uses coralnpu.outer_product → csrw KSCM0/KSCM1/KISA")
    print("  Executes 64 multiply-accumulates per CSR trigger (8×8 outer product)")
    print("  vs i32 path: 2 vle32 + 1 vmul + 1 vredsum per 4-element tile")
    print("  MAC advantage grows with K dimension (more tiles = more CSR amortization)")

    print("\n--- int8 MAC CSR assembly excerpt (first 15 instructions) ---")
    for line in asm_mac.splitlines()[:25]:
        if any(kw in line for kw in ('csrw', 'vle32', 'csrr', 'vsetvl', 'vse32', 'addi', 'lui')):
            print(f"  {line.strip()}")
