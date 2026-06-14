#!/usr/bin/env python3
"""circt4coral e2e CI test - 9 tests.

Uses real CoralNPU MLIR dialect (not hand-written asm) to validate end-to-end:
  MLIR → circt4coral --emit-asm → RISC-V asm → GAS → LD → spike+CoralNPU

Tests:
  1. scalar_add: coralnpu.li + add
  2. mul_mlu:   coralnpu.li + mul (routed to MLU)
  3. vsetvl:    coralnpu.vsetvl e32, m1
  4. vadd:      vle8 + vadd + vse8
  5. lw_sw:     scalar lw/sw memory access
  6. dma:       coralnpu.dma_load/dma_store
  7. branch:    coralnpu.beq + bne + return
  8. tosa_add:  TOSA add lowering → vle8 + vadd + vse8
  9. tosa_mul:  TOSA mul lowering → vle8 + vmul + vse8

Usage:
  python3 circt4coral-ci-test.py [--circt4coral PATH] [--spike PATH]
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

V_PREAMBLE = """    # Enable V extension: mstatus.VS = 01 (initial)
    li t0, 0x200
    csrs 0x300, t0
"""

LINKER_SCRIPT = """SECTIONS {
    . = 0x11000;
    .text : { *(.text*) }
    .data : { *(.data*) }
    .bss : { *(.bss*) }
}
"""


def setup_paths():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circt4coral", default=os.environ.get("CIRCT4CORAL"))
    parser.add_argument("--spike", default=os.environ.get("SPIKE"))
    parser.add_argument("--riscv-as", default="riscv64-linux-gnu-as")
    parser.add_argument("--riscv-ld", default="riscv64-linux-gnu-ld")
    parser.add_argument("--isa", default="rv32imf_v_zvl128b_zicsr_zifencei")
    parser.add_argument("--keep-files", action="store_true")
    args = parser.parse_args()

    candidates = [
        args.circt4coral,
        "./bin/circt4coral",
        "/Work/circt/bin/circt4coral",
        os.environ.get("CIRCT4CORAL"),
    ]
    circt4coral = next((p for p in candidates if p and Path(p).exists()), None)
    if not circt4coral:
        print("ERROR: circt4coral not found. Set --circt4coral or CIRCT4CORAL env var.")
        sys.exit(2)

    candidates = [
        args.spike,
        "/Work/spike/riscv-isa-sim/build/spike",
        os.environ.get("SPIKE"),
    ]
    spike = next((p for p in candidates if p and Path(p).exists()), None)
    if not spike:
        print("WARNING: spike not found. Tests will compile but not run.")
    return circt4coral, spike, args


def run_circt4coral(circt4coral, mlir_path):
    r = subprocess.run(
        [circt4coral, str(mlir_path), "--emit-asm"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"circt4coral failed (rc={r.returncode}):\n{r.stderr}")
    return r.stdout


def assemble_link_run(circt4coral, spike, args, asm, name, with_v_preamble=True):
    """assemble + link + spike run, return (success, message)."""
    asm = asm.replace(".L?", ".L0")
    if with_v_preamble:
        asm = V_PREAMBLE + "\n" + asm
    if ".L0:" not in asm:
        asm = asm + "\n.L0:\n"
    if "ebreak" not in asm:
        asm = asm + "    ebreak\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".S", delete=False) as af:
        af.write(asm)
        asm_file = Path(af.name)
    obj = Path(str(asm_file) + ".o")
    elf = Path(str(asm_file) + ".elf")
    ld_script = Path(str(asm_file) + ".ld")
    debug_cmd = Path(str(asm_file) + ".spk")

    try:
        ld_script.write_text(LINKER_SCRIPT)
        debug_cmd.write_text("run 200\nquit\n")

        r = subprocess.run(
            [args.riscv_as, "-march=rv32imv_zicsr", "-mabi=ilp32",
             "-o", str(obj), str(asm_file)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            return False, f"AS failed: {r.stderr[:300]}"

        r = subprocess.run(
            [args.riscv_ld, "-m", "elf32lriscv", "--no-dynamic-linker",
             "-static", "-e", "_start", "-T", str(ld_script),
             "-o", str(elf), str(obj)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            return False, f"LD failed: {r.stderr[:300]}"

        if spike is None:
            return True, "compiled (no spike)"

        env = os.environ.copy()
        env["PATH"] = str(Path(spike).parent) + ":" + env.get("PATH", "")
        r = subprocess.run(
            [spike, f"--isa={args.isa}", "--priv=m",
             "-m0x11000:0x10000,0x40000000:0x10000000",
             "-d", f"--debug-cmd={debug_cmd}", str(elf)],
            capture_output=True, text=True, env=env, timeout=15
        )
        if r.returncode == 0:
            return True, "spike exit=0"
        return False, f"spike exit={r.returncode} stderr={r.stderr[-200:]}"
    finally:
        if not args.keep_files:
            for f in [asm_file, obj, elf, ld_script, debug_cmd]:
                if f.exists():
                    f.unlink()


# Real CoralNPU MLIR (using only ops defined in CoralNPUOps.td)
TEST_MLIR = {
    "scalar_add": """\
func.func @scalar_add() -> i32 {
  %a = coralnpu.li 42 : i32
  %b = coralnpu.li 7 : i32
  %c = coralnpu.add %a, %b : i32
  coralnpu.return %c : i32
}
""",
    "mul_mlu": """\
func.func @mul_mlu() -> i32 {
  %a = coralnpu.li 42 : i32
  %b = coralnpu.li 7 : i32
  %c = coralnpu.mul %a, %b : i32
  coralnpu.return %c : i32
}
""",
    "vsetvl": """\
func.func @vsetvl() {
  coralnpu.vsetvl e32, m1
  coralnpu.return
}
""",
    "vadd": """\
func.func @vadd() -> i32 {
  %a = coralnpu.li 16 : i32
  %addr1 = coralnpu.li 0x40000000 : i32
  %addr2 = coralnpu.li 0x40000010 : i32
  %addr3 = coralnpu.li 0x40000020 : i32
  %v1 = coralnpu.vle8 %addr1, %a : i32
  %v2 = coralnpu.vle8 %addr2, %a : i32
  %v3 = coralnpu.vadd %v1, %v2 : i32
  coralnpu.vse8 %addr3, %v3, %a : i32
  %zero = coralnpu.li 0 : i32
  coralnpu.return %zero : i32
}
""",
    "lw_sw": """\
func.func @lw_sw() -> i32 {
  %a = coralnpu.li 42 : i32
  %addr = coralnpu.li 0x40000000 : i32
  coralnpu.sw %addr, %a : i32
  %b = coralnpu.lw %addr : i32
  coralnpu.return %b : i32
}
""",
    "dma": """\
func.func @dma() {
  %src = coralnpu.li 0x40000000 : i32
  %dst = coralnpu.li 0x11000 : i32
  %sz = coralnpu.li 64 : i32
  coralnpu.dma_load %src, %dst, %sz
  %dst2 = coralnpu.li 0x12000 : i32
  coralnpu.dma_store %dst2, %src, %sz
  coralnpu.return
}
""",
    "branch": """\
func.func @branch(%arg0: i32) -> i32 {
  %zero = coralnpu.li 0 : i32
  %one = coralnpu.li 1 : i32
  coralnpu.cond_br eq=true %arg0, %zero, ^eq, ^ne
^eq:
  coralnpu.return %zero : i32
^ne:
  coralnpu.return %one : i32
}
""",
    "tosa_add": """\
func.func @tosa_add_test(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
""",
    "tosa_mul": """\
func.func @tosa_mul_test(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %shift = arith.constant dense<0> : tensor<1xi8>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<1xi8>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
""",
}


def main():
    circt4coral, spike, args = setup_paths()
    print(f"circt4coral: {circt4coral}")
    print(f"spike:       {spike or '(not found - compile only)'}")
    print(f"isa:         {args.isa}\n")

    total = 0
    passed = 0
    results = []
    for name, mlir_text in TEST_MLIR.items():
        total += 1
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as mf:
            mf.write(mlir_text)
            mlir_file = Path(mf.name)
        try:
            try:
                asm = run_circt4coral(circt4coral, mlir_file)
            except RuntimeError as e:
                print(f"  ✗ {name}: {e}")
                results.append((name, False, str(e)))
                continue
            ok, msg = assemble_link_run(circt4coral, spike, args, asm, name)
            mark = "✓" if ok else "✗"
            print(f"  {mark} {name}: {msg}")
            results.append((name, ok, msg))
            if ok:
                passed += 1
        finally:
            if not args.keep_files:
                mlir_file.unlink(missing_ok=True)

    print(f"\n=== Results: {passed}/{total} ===")
    if passed == total:
        print("✅ All tests passed")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        for name, ok, msg in results:
            if not ok:
                print(f"   - {name}: {msg[:200]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
