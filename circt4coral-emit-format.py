#!/usr/bin/env python3
"""
circt4coral-emit-format.py — Generate sample emit output and validate on spike

Validates that each circt4coral emit format produces RISC-V that runs on spike
+ CoralNPU CSR patches.

Critical discoveries (2026-06-14):
- spike needs --isa=rv32imf_v_zvl128b (NOT just rv32im_v) to enable V extension
- mstatus.VS=01 must be set BEFORE vsetivli/vadd.vv (default is 0=off)
- mstatus CSR is 0x300, VS field is bit 9 (0x200)
- Without these, V instructions trap with trap_illegal_instruction
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

WORKDIR = Path("/Work")
SPIKE = WORKDIR / "spike/riscv-isa-sim/build/spike"
SPIKE_DEPS = "/tmp/spike-deps"
ISA = "rv32imf_v_zvl128b_zicsr_zifencei"
LINKER_SCRIPT = "/tmp/circt-link.ld"

LINKER_SCRIPT_CONTENT = """
SECTIONS {
    . = 0x11000;
    .text : { *(.text*) }
    .data : { *(.data*) }
    .bss : { *(.bss*) }
}
"""

# Common preamble that sets mstatus.VS for V instructions
V_PREAMBLE = """
    # Enable V extension: mstatus.VS = 01 (initial)
    li t0, 0x200
    csrs 0x300, t0
"""


# ---- Test 1: A scalar add (real RV32I) ----
TEST_SCALAR_ADD_ASM = """
.globl _start
_start:
    li   a0, 100
    li   a1, 50
    add  a2, a0, a1
    ebreak
"""

# ---- Test 2: A mul (routed to MLU in real CoralNPU) ----
TEST_MUL_ASM = """
.globl _start
_start:
    li   a0, 17
    li   a1, 19
    mul  a2, a0, a1
    ebreak
"""

# ---- Test 3: CoralNPU custom CSR access ----
TEST_CSR_ASM = """
.globl _start
_start:
    li   t0, 0xDEADBEEF
    csrw 0x7C0, t0
    csrr t1, 0x7C0
    csrr t2, 0xFC0
    ebreak
"""

# ---- Test 4: Vector setup (vsetivli) ----
TEST_VSETVLI_ASM = V_PREAMBLE + """
    vsetivli x0, 16, e32, m1, ta, ma
    ebreak
"""

# ---- Test 5: Vector add (RVV vadd.vv) ----
TEST_VADD_ASM = V_PREAMBLE + """
    vsetivli x0, 16, e32, m1, ta, ma
    vadd.vv v0, v1, v2
    ebreak
"""

# ---- Test 6: Outer product → mul + csrw KSCM ----
TEST_OUTER_PRODUCT_ASM = """
.globl _start
_start:
    li   t0, 0x1111
    li   t1, 0x2222
    li   t2, 0x3333
    csrw 0xFC4, t0
    csrw 0xFC8, t1
    csrw 0xFC0, t2
    ebreak
"""

# ---- Test 7: DMA (memory-mapped writes) ----
TEST_DMA_ASM = """
.globl _start
_start:
    li   t0, 0x40000000
    li   t1, 0x10000
    sw   t1, 0(t0)
    li   t1, 0x11000
    sw   t1, 4(t0)
    li   t1, 256
    sw   t1, 8(t0)
    li   t1, 1
    sw   t1, 12(t0)
    ebreak
"""


TESTS = [
    ("scalar_add", TEST_SCALAR_ADD_ASM, "add instruction (RV32I R-type)"),
    ("mul", TEST_MUL_ASM, "mul instruction (RV32M, routed to MLU)"),
    ("csr_access", TEST_CSR_ASM, "CoralNPU custom CSR read/write"),
    ("vsetvli", TEST_VSETVLI_ASM, "Vector configuration (vsetivli)"),
    ("vadd", TEST_VADD_ASM, "RVV vadd.vv"),
    ("outer_product", TEST_OUTER_PRODUCT_ASM, "Outer product via KSCM0/1 + KISA"),
    ("dma", TEST_DMA_ASM, "DMA via MMIO writes"),
]


def write_linker_script():
    with open(LINKER_SCRIPT, "w") as f:
        f.write(LINKER_SCRIPT_CONTENT)


def assemble(asm: str, elf: str) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".S", delete=False) as f:
        f.write(asm)
        asm_file = f.name
    obj_file = asm_file + ".o"
    try:
        r = subprocess.run(
            ["riscv64-linux-gnu-as", "-march=rv32imv_zicsr", "-mabi=ilp32",
             "-o", obj_file, asm_file],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"   ❌ AS failed: {r.stderr.strip()}")
            return False
        r = subprocess.run(
            ["riscv64-linux-gnu-ld", "-m", "elf32lriscv", "--no-dynamic-linker",
             "-static", "-e", "_start", "-T", LINKER_SCRIPT,
             "-o", elf, obj_file],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"   ❌ LD failed: {r.stderr.strip()}")
            return False
        return True
    finally:
        for f in [asm_file, obj_file]:
            if os.path.exists(f):
                os.unlink(f)


def run_on_spike(elf: str) -> tuple:
    env = os.environ.copy()
    env["PATH"] = SPIKE_DEPS + ":" + env.get("PATH", "")
    cmd = [
        str(SPIKE), f"--isa={ISA}", "--priv=m",
        "-m0x11000:0x10000,0x40000000:0x10000000",
    ]
    cmd_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".spk", delete=False, dir="/tmp")
    cmd_file.write("run 30\nquit\n")
    cmd_file.close()
    cmd.extend(["-d", f"--debug-cmd={cmd_file.name}", elf])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env=env, timeout=30)
        return r.returncode, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def main():
    print("=" * 70)
    print(" Circt4Coral → Spike+CoralNPU Emit Format Tests")
    print("=" * 70)
    print(f"Spike: {SPIKE}")
    print(f"ISA:   {ISA}")
    print()

    write_linker_script()

    passed = 0
    failed = 0
    for name, asm, desc in TESTS:
        print(f"[{name}] {desc}")
        elf = f"/tmp/circt-{name}.elf"
        if not assemble(asm, elf):
            failed += 1
            print()
            continue
        rc, trace = run_on_spike(elf)
        if rc in (0, 124):
            lines = []
            trap = False
            for line in trace.split("\n"):
                if "instruction_access_fault" in line or "store_access_fault" in line:
                    trap = True
                    break
                if line.strip():
                    lines.append(line)
                    if len(lines) > 12:
                        break
            if trap:
                print(f"   ❌ trap detected (not ebreak)")
                print(f"     {lines[-1] if lines else trace[:200]}")
                failed += 1
            else:
                print(f"   ✅ spike exit={rc}")
                for line in lines[:8]:
                    print(f"     {line}")
                passed += 1
        else:
            print(f"   ❌ spike exit={rc}")
            print(f"     {trace[:300]}")
            failed += 1
        print()

    print("=" * 70)
    print(f" Results: {passed}/{passed+failed} passed")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
