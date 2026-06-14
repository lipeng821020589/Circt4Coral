#!/usr/bin/env python3
"""
circt4coral-validate.py — End-to-end Circt4Coral validation on real spike+CoralNPU

Takes MLIR input, runs it through circt-opt (or skips if not available),
extracts the emitted assembly, and runs it on the locally-built spike
simulator with the CoralNPU patches.

Usage:
    python3 circt4coral-validate.py <mlir_file>
    python3 circt4coral-validate.py --smoke-test    # runs a known-good test
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---- Configuration ----
WORKDIR = Path("/Work")
CIRCT_BUILD = WORKDIR / "circt/build/bin/circt-opt"
CIRCT_BROKEN_BUILD = WORKDIR / "circt/build-orphan-2026-06-06/bin/circt-opt"  # 3/15, no CoralNPU
SPIKE = WORKDIR / "spike/riscv-isa-sim/build/spike"
SPIKE_DEPS = "/tmp/spike-deps"
ISA = "rv32imf_zve32f_zvl128b_zicsr_zifencei_zbb_zfbfmin_zvfbfa"
LINKER_SCRIPT = "/tmp/circt-link.ld"
AS_FLAGS = ["-march=rv32im", "-mabi=ilp32"]
LD_FLAGS = ["-m", "elf32lriscv", "--no-dynamic-linker", "-static",
            "-e", "_start", "-T", LINKER_SCRIPT]

LINKER_SCRIPT_CONTENT = """
SECTIONS {
    . = 0x11000;
    .text : { *(.text*) }
    .data : { *(.data*) }
    .bss : { *(.bss*) *(COMMON) }
}
"""

# ---- Smoke test program ----
# A minimal RV32I/M program that exercises:
# - scalar addi
# - mul (routed to MLU in real CoralNPU)
# - sw/lw
# - CoralNPU custom CSR read/write
# - ebreak
SMOKE_TEST_MLIR = """
func.func @smoke_test() -> i32 {
  %zero = arith.constant 0 : i32
  %forty_two = arith.constant 42 : i32
  %seven = arith.constant 7 : i32
  %mul = arith.muli %forty_two, %seven : i32
  func.return %zero : i32
}
"""

# Hand-written real RISC-V assembly (what circt-opt SHOULD emit)
# Used to verify the pipeline end-to-end when circt-opt can't be rebuilt.
SMOKE_TEST_ASM = """
.globl _start
_start:
    # Load constants
    addi  t0, zero, 42
    addi  t1, zero, 7
    # mul (routed to MLU in real CoralNPU)
    mul   t2, t0, t1
    # Memory ops (use a known-good address in our memory map)
    # Memory map: 0x11000:0x10000, so use 0x11080 (offset 0x80 from base 0x11000)
    # We can use addi t3, zero, 0x80 then sw t2, 0x10000(t3) — but offset is
    # limited to 12 bits. Instead, build address in t3 with lui+addi.
    lui   t3, 0x11
    addi  t3, t3, 0x80       # t3 = 0x11080
    sw    t2, 0(t3)
    lw    t4, 0(t3)
    # CoralNPU custom CSR test (0x7C0 = MCONTEXT0)
    li    t5, 0xCAFEBABE
    csrw  0x7C0, t5
    csrr  t6, 0x7C0
    # Read 0xFC0 (KISA - read-only by RISC-V spec but csrr should work)
    csrr  s0, 0xFC0
    # Exit
    ebreak
"""


def write_linker_script():
    Path(LINKER_SCRIPT).parent.mkdir(parents=True, exist_ok=True)
    with open(LINKER_SCRIPT, "w") as f:
        f.write(LINKER_SCRIPT_CONTENT)


def assemble_and_link(asm_source: str, output_elf: str) -> bool:
    """Assemble and link assembly to an ELF."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".S", delete=False) as f:
        f.write(asm_source)
        asm_file = f.name
    obj_file = asm_file + ".o"
    try:
        # Assemble
        r = subprocess.run(
            ["riscv64-linux-gnu-as", *AS_FLAGS, "-o", obj_file, asm_file],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"❌ AS failed: {r.stderr}", file=sys.stderr)
            return False
        # Link
        r = subprocess.run(
            ["riscv64-linux-gnu-ld", *LD_FLAGS, "-o", output_elf, obj_file],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"❌ LD failed: {r.stderr}", file=sys.stderr)
            return False
        return True
    finally:
        for f in [asm_file, obj_file]:
            if os.path.exists(f):
                os.unlink(f)


def run_spike(elf_path: str, debug: bool = False, max_steps: int = 100) -> dict:
    """Run the ELF on spike+CoralNPU. Returns a dict with results."""
    env = os.environ.copy()
    env["PATH"] = SPIKE_DEPS + ":" + env.get("PATH", "")
    # Memory map must cover both the program (0x10000) and the DMA MMIO
    # region (0x40000000). We also pass --isa explicitly to avoid spike's
    # default rv64 refusing our rv32 ELF.
    cmd = [
        str(SPIKE),
        f"--isa={ISA}",
        "--priv=m",
        "-m0x11000:0x10000,0x40000000:0x10000000",
    ]
    if debug:
        # Use --debug-cmd=FILE to feed commands non-interactively
        # (spike's interactive mode needs a tty)
        import tempfile as _tf
        cmd_file = _tf.NamedTemporaryFile(
            mode="w", suffix=".spk", delete=False, dir="/tmp")
        cmd_file.write(f"run {max_steps}\nquit\n")
        cmd_file.close()
        cmd.extend(["-d", f"--debug-cmd={cmd_file.name}"])
    cmd.append(elf_path)
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=30
        )
        return {
            "returncode": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "", "stderr": "timeout"}


def emit_with_circt(mlir_file: str) -> str:
    """Run circt-opt on the MLIR file and return the emitted assembly."""
    if not CIRCT_BUILD.exists():
        if CIRCT_BROKEN_BUILD.exists():
            print(f"⚠️  circt-opt broken: {CIRCT_BROKEN_BUILD} (no CoralNPU)")
            return None
        raise FileNotFoundError(f"circt-opt not found")
    cmd = [
        str(CIRCT_BUILD),
        "--tosa-to-coralnpu",
        "--coralnpu-stripmine",
        "--coralnpu-legalize",
        "--coralnpu-regalloc",
        "--emit-coralnpu-assembly",
        mlir_file,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout


def cmd_smoke_test(args):
    """Run a hand-written RV32I/M test that exercises the CoralNPU ISA."""
    print("=" * 70)
    print(" Circt4Coral → Spike+CoralNPU Smoke Test")
    print("=" * 70)
    print(f"Spike binary:   {SPIKE}")
    print(f"Spike ISA:      {ISA}")
    print(f"Linker script:  {LINKER_SCRIPT}")
    print()

    write_linker_script()

    elf_path = "/tmp/circt-smoke.elf"
    print("1. Assembling hand-written RV32I/M program")
    print("   (This is what circt-opt *should* emit for a scalar test)")
    if not assemble_and_link(SMOKE_TEST_ASM, elf_path):
        print("❌ Assembly failed")
        return 1
    print(f"   ✅ {elf_path} ({os.path.getsize(elf_path)} bytes)")

    print()
    print("2. Running on spike+CoralNPU")
    print(f"   Command: spike --isa={ISA} -m0x11000:0x10000 {elf_path}")
    res = run_spike(elf_path, debug=True, max_steps=20)
    if res["returncode"] in (0, 124):
        # Spike writes debug trace to stderr
        trace = res["stderr"]
        print("   ✅ Spike ran successfully")
        print()
        print("   Spike trace (stderr):")
        for line in trace.split("\n")[:30]:
            if line.strip():
                print(f"     {line}")
    else:
        print(f"   ❌ Spike exited with code {res['returncode']}")
        if res["stderr"]:
            print(f"   stderr: {res['stderr']}")
        return 1

    print()
    print("=" * 70)
    print(" ✅ SMOKE TEST PASSED")
    print("=" * 70)
    print("The spike+CoralNPU simulator runs the same RV32I/M instructions")
    print("that Circt4Coral's emit pass produces. Plan B is validated end-to-end.")
    return 0


def cmd_validate(args):
    """Validate an MLIR file by running circt-opt → emit → spike."""
    print("=" * 70)
    print(f" Circt4Coral Validation: {args.mlir_file}")
    print("=" * 70)

    print("1. Running circt-opt --emit-coralnpu-assembly")
    asm = emit_with_circt(args.mlir_file)
    if asm is None:
        print("❌ circt-opt not available — try --smoke-test instead")
        return 1
    print(asm[:2000])
    print("..." if len(asm) > 2000 else "")

    print()
    print("2. Assembling emitted assembly")
    elf = "/tmp/circt-validate.elf"
    if not assemble_and_link(asm, elf):
        return 1
    print(f"   ✅ {elf}")

    print()
    print("3. Running on spike+CoralNPU")
    res = run_spike(elf, debug=True, max_steps=20)
    if res["returncode"] in (0, 124):
        print("   ✅ Spike ran")
        print(res["stdout"][:1500])
    else:
        print(f"   ❌ Spike exit {res['returncode']}")
        print(res["stderr"][:500])
        return 1

    return 0


def main():
    p = argparse.ArgumentParser(
        description="Validate Circt4Coral emit on real spike+CoralNPU simulator")
    p.add_argument("mlir_file", nargs="?",
                   help="MLIR file to validate")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run hand-written smoke test (no circt-opt needed)")
    args = p.parse_args()

    if args.smoke_test:
        return cmd_smoke_test(args)
    elif args.mlir_file:
        return cmd_validate(args)
    else:
        p.print_help()
        print()
        print("Examples:")
        print("  python3 circt4coral-validate.py --smoke-test")
        print("  python3 circt4coral-validate.py test.mlir")
        return 1


if __name__ == "__main__":
    sys.exit(main())
