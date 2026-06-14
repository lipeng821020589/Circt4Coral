# CSR Codegen Demo (Direction 3)

Demonstrates that circt4coral correctly emits RISC-V assembly for
**CoralNPU custom CSRs** that aren't part of standard RISC-V ISA:

| CSR Address | Name        | Purpose                          |
|-------------|-------------|----------------------------------|
| 0xFC0       | KISA        | Kernel/ISA configuration         |
| 0xFC4-0xFD4 | KSCM0-4     | Kernel/Scratch/Config Matrix     |
| 0x7C0       | MCONTEXT0   | Mode/Context register            |

## Source MLIR

```mlir
func.func @demo_outer_product() -> i32 {
  %a = coralnpu.li 2 : i32
  %b = coralnpu.li 3 : i32
  %acc = coralnpu.li 0 : i32
  %r = coralnpu.outer_product %a, %b, %acc : i32
  coralnpu.return %r : i32
}
```

## Generated Assembly

```asm
addi   x1, zero, 2
addi   x2, zero, 3
addi   x3, zero, 0
# outer_product: input=x1 weight=x2 acc=x3
csrw   4036, x1  # KSCM0 = input row
csrw   4040, x2  # KSCM1 = weight col
csrw   4032, x3  # KISA = acc target (also triggers MAC)
ret
```

The `csrw 4036/4040/4032` are non-standard CSR addresses that the
**google-coral/coralnpu** hardware and **spike+CoralNPU patches 0002-0005**
recognize as accelerator control registers.

## Running on Spike

### Plain spike (without CoralNPU patches)
Does NOT recognize these CSRs — reports `trap_illegal_instruction`.

### Patched spike (with CoralNPU patches 0002-0005 from google-coral/coralnpu)
The upstream patches have a **bug**: they register the 0xFC0-0xFD4 CSRs
as `basic_csr_t`, which is marked read-only because the address falls
in the 0xC00-0xFFF read-only range per RISC-V convention.

**Fix**: We add a custom `coralnpu_csr_t` class that overrides the
read-only check (see `bench/0006-coralnpu-csr-writable.patch`).

To apply all 6 patches (0001-0005 from upstream + our 0006 fix):

```bash
cd /Work/spike/riscv-isa-sim
git apply /Work/coralnpu-google/third_party/spike/0001-Add-mpause.patch
git apply /Work/coralnpu-google/third_party/spike/0002-Coral-Deviations.patch
git apply /Work/coralnpu-google/third_party/spike/0003-Dump-GPRs-on-EBREAK.patch
git apply /Work/coralnpu-google/third_party/spike/0004-Add-custom-CoralNPU-CSRs-and-update-MVENDORID-MARCHI.patch
git apply /Work/coralnpu-google/third_party/spike/0005-Force-logging-in-vcompress.patch
git apply /Work/circt/bench/0006-coralnpu-csr-writable.patch
cd build && make spike -j$(nproc)
```

Then re-run:
```bash
python3 /Work/circt/bench/csr-demo.py
# Output: CSR codegen worked on spike!
```

## Why this is a competitive advantage

- **Standard RISC-V toolchain compatibility**: `csrw 0xFC0, x1` is valid
  RISC-V assembly that any toolchain can assemble
- **CoralNPU-specific semantics**: The 0xFC0-0xFD4 range is reserved by
  google-coral/coralnpu for accelerator control
- **No fake opcodes**: We use real RISC-V CSRs instead of inventing
  custom opcodes (the previous circt4coral had 0x0B/0x2B/0xEB which
  don't exist in RISC-V)

## Comparison with IREE

IREE generates RISC-V assembly for *standard* RVV + standard CSRs.
To target CoralNPU custom CSRs, IREE would need:
1. A new backend plugin (not implemented upstream)
2. Or an inline asm hack in the input MLIR

circt4coral has CoralNPU-specific ops (`coralnpu.outer_product`) that
natively understand the custom CSRs.
