# From TOSA to google-coral/coralnpu: A standalone MLIR Compiler in 2 Days

*Draft for MLIR Discourse forum - https://discourse.llvm.org/c/mlir/*

## TL;DR

We built a standalone MLIR compiler (`circt4coral`) that lowers
TOSA → CoralNPU dialect → RISC-V assembly with **RVV** + custom
**KISA/KSCM CSRs** in 2 days, with a **7/7 e2e test pass** rate
on spike + CoralNPU patches. The entire compiler is a 211 MB
single binary that runs without an LLVM/MLIR install on the user
side. We think this is the first end-to-end working pipeline
targeting google-coral/coralnpu publicly available.

## Why this matters

`google-coral/coralnpu` is a 6-month-old open-source hardware
accelerator (announced November 2025 in partnership with
VeriSilicon). It's based on the RISC-V Vector Extension (RVV)
plus a small matrix engine with custom CSRs. As of mid-2026:

- **google-coral** publishes: hardware RTL + spike patches + LiteRT
  reference compiler (C-intrinsic-based, not MLIR)
- **IREE** has RISC-V64 microkernel experiments but **no CoralNPU
  backend**
- **No public MLIR→CoralNPU e2e flow** has been demonstrated

This is a window. We built one.

## The compiler

```
TOSA MLIR
   │  tosa-to-coralnpu
   ▼
CoralNPU dialect (custom)
   │  stripmine + legalize + regalloc
   ▼
CoralNPU canonical form
   │  emit-coralnpu-assembly
   ▼
RISC-V assembly (RV32IMV + KISA/KSCM CSRs)
   │  riscv64-linux-gnu-as
   ▼
ELF binary
   │  spike --isa=rv32imf_v_zvl128b_zicsr_zifencei
   ▼
exit 0 ✓
```

The compiler is a 211 MB ELF executable that statically links
against orphan-built MLIR/LLVM. We had to do **partial link + gold
linker** because the standard `circt-opt` build is broken with our
LLVM submodule mismatch.

## The 5 most painful bugs we hit

### 1. The 0x0B/0x2B/0xEB disaster

The original `circt4coral` had fake opcodes 0x0B, 0x2B, 0xEB that
**do not exist in RISC-V**. They were invented in the upstream
circt4coral project. We rewrote 649 lines of emit-pass to use
real RVV + custom CSRs (`csrw 0xFC0, x1` for KISA writes).

Lesson: never invent opcodes. RISC-V has both a standard
extension set AND a CSR escape hatch — use them.

### 2. PropertyRef ABI break

Our LLVM submodule (main, +69780be) introduced a `PropertyRef`
type that doesn't exist in the orphan build (MLIR 23.0). This
broke `circt-opt` compilation. Workaround: build a standalone
binary instead of using `circt-opt --load-plugin`.

### 3. DCE eating vsetvl

Our `tosa-to-coralnpu` pass uses `mlir::applyPatternsGreedily`
which runs DCE. The vsetvl instruction has no result, so DCE
removes it. Then the vector op that needs it traps with
`trap_illegal_instruction`.

Fix: in the emit pass, auto-insert `vsetivli` before every vector
op. Hacky but works.

Better fix (future): implement `MemoryEffectOpInterface` properly
on CoralNPU ops so DCE understands side effects.

### 4. vle8 DCE'd because result unused

Same root cause. Fix: chain results through the MLIR so the
last vle8 result is the function return value.

### 5. `_start` at wrong address

Spike runs the boot stub at 0x1000, then jumps to 0x11000.
If our linker script puts `_start` at 0x0, we get
`trap_instruction_access_fault` (the boot stub jumps into
our code before it's loaded).

Fix: linker script `SECTIONS { . = 0x11000; .text : ... }`.

## What works (7/7 e2e tests)

| Test        | MLIR feature used               | Result |
|-------------|---------------------------------|--------|
| scalar_add  | coralnpu.li, add, return        | ✓      |
| mul_mlu     | coralnpu.mul (routed to MLU)    | ✓      |
| vsetvl      | coralnpu.vsetvl                 | ✓      |
| vadd        | coralnpu.vle8, vadd, vse8       | ✓      |
| lw_sw       | coralnpu.lw, sw                 | ✓      |
| dma         | coralnpu.dma_load               | ✓      |
| branch      | coralnpu.cond_br                | ✓      |

We also have a TOSA test that lowers `tosa.add` through the full
pipeline.

## What doesn't work yet

- The patched spike with CoralNPU CSR support is required to run
  the `outer_product` demo. Plain spike reports
  `trap_illegal_instruction` on `csrw 0xFC0, x1`. The patches are
  in google-coral/coralnpu; we're working on a one-click install.
- No automated register spilling yet. Our `regalloc` is a
  linear-scan that fails on programs with >16 simultaneous live
  values.

## What we want from the MLIR community

- **Feedback on `CoralNPU.td` design**: should this be a new
  dialect, an extension of `vector`, or a `tosa` profile?
- **Co-design of the CSR codegen interface**: should the
  accelerator ops carry a `target_csr` attribute, or should we
  add a new `csr_region` op kind?
- **A clean way to bundle custom CSRs with the ISA string**:
  currently we hardcode `rv32imf_v_zvl128b_zicsr_zifencei` in
  the emit pass.

## How to try it

```bash
git clone https://github.com/lipeng821020589/circt4coral.git
cd circt4coral
./bin/build-circt4coral.sh
./bin/circt4coral test/Dialect/CoralNPU/ops.mlir --emit-asm | head
```

## Acknowledgments

- The CIRCT team for the MLIR infrastructure
- google-coral for open-sourcing the hardware
- The spike team for the RISC-V ISA simulator
- Our local spike+CoralNPU patches 0002-0005 (in development)

## About the author

Built in 2 days by [peng](https://github.com/lipeng821020589)
as part of an MLIR learning project. Reach out on MLIR
Discourse or GitHub Issues.
