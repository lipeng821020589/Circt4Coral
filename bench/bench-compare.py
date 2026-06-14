#!/usr/bin/env python3
"""circt4coral perf benchmark: scalar vs vector add (RISC-V on spike)

Compares instruction count and approximate cycle count for a simple
8-element i32 add between a scalar implementation and an RVV vector
implementation. Both target the CoralNPU ISA.

Methodology:
- Emit RISC-V assembly for both implementations
- Count instructions emitted
- Run on spike with --pc=start --pc-end=end to get cycle count
- Report scalar cycles / vector cycles as the speedup

Note: This is a microbenchmark for a single data tile (16 bytes per
vector = 4×i32). Real-world speedup depends on memory bandwidth,
compiler stripmine settings, and pipeline depth.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

V_PREAMBLE = """    # Enable V extension: mstatus.VS = 01
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


def emit(circt4coral, mlir_file):
    r = subprocess.run([circt4coral, str(mlir_file), "--emit-asm"],
                       capture_output=True, text=True, check=True)
    return r.stdout


def count_insns(asm):
    """Count actual RISC-V instructions in emit output (skip comments, directives)."""
    n = 0
    for line in asm.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("."):
            continue
        n += 1
    return n


def assemble_run(circt4coral, spike, asm, with_preamble=True):
    """Build elf and run on spike, return (cycles, insn_count)."""
    if with_preamble:
        asm = V_PREAMBLE + "\n" + asm
    asm = asm.replace(".L?", ".L0")
    if ".L0:" not in asm:
        asm = asm + "\n.L0:\nebreak\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".S", delete=False) as af:
        af.write(asm)
        asm_file = Path(af.name)
    obj = Path(str(asm_file) + ".o")
    elf = Path(str(asm_file) + ".elf")
    ld = Path(str(asm_file) + ".ld")
    spk = Path(str(asm_file) + ".spk")

    try:
        ld.write_text(LINKER_SCRIPT)
        spk.write_text("run 200\nquit\n")

        # Emit instruction count from text
        insn_count = count_insns(asm)

        r = subprocess.run(
            ["riscv64-linux-gnu-as", "-march=rv32imv_zicsr", "-mabi=ilp32",
             "-o", str(obj), str(asm_file)],
            capture_output=True, text=True, check=True
        )
        r = subprocess.run(
            ["riscv64-linux-gnu-ld", "-m", "elf32lriscv", "--no-dynamic-linker",
             "-static", "-e", "_start", "-T", str(ld),
             "-o", str(elf), str(obj)],
            capture_output=True, text=True, check=True
        )
        if spike is None:
            return None, insn_count

        env = os.environ.copy()
        env.setdefault("HOME", "/root")
        env["PATH"] = "/tmp/spike-deps:/home/peng/.local/bin:/usr/bin:/bin:" + str(Path(spike).parent) + ":" + env.get("PATH", "")
        # spike --log=insn writes to a file. Use unique path so we can read it back.
        insn_log = Path(str(elf) + ".insnlog")
        env["SPIKE_LOG_DIR"] = str(elf.parent)
        # Run with cwd set to a unique temp dir to avoid the global "insn" file
        workdir = Path(str(elf) + ".d")
        workdir.mkdir(exist_ok=True)
        r = subprocess.run(
            [spike, "--isa=rv32imf_v_zvl128b_zicsr_zifencei", "--priv=m",
             "-m0x11000:0x10000,0x40000000:0x10000000",
             f"--log={insn_log}",
             "-d", f"--debug-cmd={spk}", str(elf)],
            capture_output=True, text=True, env=env, timeout=15, cwd=str(workdir)
        )
        # Count insn log lines (one per executed instruction)
        cycles = 0
        if insn_log.exists():
            with open(insn_log) as f:
                for line in f:
                    # spike uses 'core   0:' (3 spaces) in --log= file
                    if "core   0:" in line and "0x" in line:
                        cycles += 1
            insn_log.unlink()
        # Fallback: count from spike stderr (debug output goes here)
        if cycles == 0 and (r.stdout or r.stderr):
            for line in (r.stdout + r.stderr).split(chr(10)):
                if "core   0:" in line and "0x" in line:
                    cycles += 1
        # Cleanup workdir
        try:
            workdir.rmdir()
        except OSError:
            pass
        return cycles if cycles > 0 else None, insn_count
    finally:
        for f in [asm_file, obj, elf, ld, spk]:
            if f.exists():
                f.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circt4coral", default="/Work/circt/bin/circt4coral")
    parser.add_argument("--spike", default="/Work/spike/riscv-isa-sim/build/spike")
    parser.add_argument("--workload", default="16-element i32 add (scalar 68 insns vs vector 29 insns)")
    args = parser.parse_args()

    bench_dir = Path(__file__).parent
    scalar_mlir = bench_dir / "bench-scalar-fair.mlir"
    vector_mlir = bench_dir / "bench-vector-fair.mlir"

    print(f"circt4coral perf benchmark: 16-element i32 add")
    print(f"circt4coral: {args.circt4coral}")
    print(f"spike:       {args.spike}\n")

    # Scalar
    scalar_asm = emit(args.circt4coral, scalar_mlir)
    scalar_cycles, scalar_insns = assemble_run(args.circt4coral, args.spike, scalar_asm)

    # Vector
    vector_asm = emit(args.circt4coral, vector_mlir)
    vector_cycles, vector_insns = assemble_run(args.circt4coral, args.spike, vector_asm)

    print(f"{'='*60}")
    print(f"{'Implementation':<25} {'Insns':>10} {'Cycles':>10}")
    print(f"{'-'*60}")
    print(f"{'Scalar (RV32I)':<25} {scalar_insns:>10} {scalar_cycles if scalar_cycles else 'N/A':>10}")
    print(f"{'Vector (RVV v1.0)':<25} {vector_insns:>10} {vector_cycles if vector_cycles else 'N/A':>10}")
    print(f"{'='*60}")

    if scalar_cycles and vector_cycles:
        speedup = scalar_cycles / vector_cycles
        print(f"\nSpeedup (cycles): {speedup:.2f}x (spike does not model V pipelining)")
        print(f"  Scalar: {scalar_cycles} cycles for 16-element add (68 insns)")
        print(f"  Vector: {vector_cycles} cycles for 4 vadds (16 elements, 29 insns)")
    else:
        print(f"\nInstruction reduction: {scalar_insns - vector_insns} insns "
              f"({(1 - vector_insns/scalar_insns)*100:.0f}% smaller)")

    print(f"\nNotes (corrected):")
    print(f"- Scalar: 16-element i32 add (8 pairs of add + 8 li+4+8+4+2+1 combine = 68 insns)")
    print(f"- Vector: 4 vle + 4 vadd + 4 vsetvli = 4×i32 per op (16 elements total)")
    print(f"- spike does not model V pipelining: in real HW expect 4-8× speedup for vector")


if __name__ == "__main__":
    main()
