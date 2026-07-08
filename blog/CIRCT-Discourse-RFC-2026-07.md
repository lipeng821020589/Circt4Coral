# [RFC] CoralNPU Dialect: TOSA → RISC-V Vector + Custom-CSR Lowering for google-coral/coralnpu

*Category: CIRCT / New Dialect Proposal*
*Thread: https://discourse.llvm.org/c/mlir/31 (to be posted)*

---

## Summary

We propose upstreaming **CoralNPU dialect** into CIRCT — a new MLIR dialect that
compiles TOSA IR to RISC-V assembly targeting the
[google-coral/coralnpu](https://github.com/google-coral/coralnpu) edge-AI
accelerator. The dialect covers:

- **39 TOSA lowering patterns** (elementwise, conv2d, depthwise conv2d with real
  KH×KW, pooling, rescale, activation, reduce, concat, slice, cast, select, …)
- **Custom hardware types**: `!coralnpu.vreg<sew, lmul>` for vector registers,
  `!coralnpu.xreg` for scalar registers, `!coralnpu.acc<8,8>` for the MAC
  accumulator
- **Full pipeline**: TOSA → regalloc → RISC-V text assembly + ELF binary export
- **39/39 spike E2E tests** bit-exact; TFLite flatbuffer weight injection

The implementation is at
[github.com/lipeng821020589/Circt4Coral](https://github.com/lipeng821020589/Circt4Coral),
branch `feat/real-elementwise-lowering`, v0.7.4.

---

## Motivation

### The hardware

`google-coral/coralnpu` (announced November 2025) is an open-source edge-AI
accelerator combining:

| Component | Spec |
|-----------|------|
| Base ISA | RV32IMF + Zve32f + Zvl128b + Zicsr + Zifencei + Zbb |
| Vector unit | 128-bit VLEN, SEW ∈ {e8, e16, e32}, LMUL ∈ {mf4…m8} |
| Matrix engine | 8×8 outer-product MAC via custom CSRs KISA/KSCM |
| TCM | 64 KB ITCM (0x10000–0x1FFFF), 4 KB per tensor slot |

The hardware already has an RTL release, a
[spike simulator patch set](https://github.com/google-coral/coralnpu/tree/main/patches)
(0002–0006), and a LiteRT-micro C-intrinsic backend. What it **lacks** is an
MLIR-based compilation path.

### The gap

As of July 2026:

- IREE has RV64 microkernels but **no CoralNPU backend**
- No public MLIR→CoralNPU end-to-end flow exists
- The official LiteRT-micro backend is C-intrinsic–only — no graph-level
  optimisation, no type safety, no composability with other MLIR passes

CIRCT already contains hardware-facing dialects (FIRRTL, ESI, HW, LLHD).
A CoralNPU dialect fits naturally as the lowest-level software-interface
layer for this hardware.

---

## Design

### Dialect structure

```
CoralNPU dialect
├── Scalar ops    ScalarAddOp / SubOp / MulOp / DivOp / MulhOp / …  (i32 → i32)
├── Memory ops    ScalarLwOp / SwOp  (TCM load/store)
├── Vector ops    VLE32Op / VMulOp / VAddOp / VRedSumOp / VMaxVXOp / …
│                 (args: AnyType;  result: !coralnpu.vreg<sew, lmul>)
├── Matrix ops    OuterProductOp / AConvOp / AccReadOp  (i32 → i32)
├── DMA ops       DmaLoadOp / DmaStoreOp
└── Control       VSetVLOp / ReturnOp / BranchOps
```

**Type system** (`CoralNPUTypes.td`):

```
!coralnpu.xreg             — 32-bit scalar register (x0–x31)
!coralnpu.vreg<sew, lmul>  — vector register group (v0–v63),
                             e.g. !coralnpu.vreg<e32, m1>
!coralnpu.acc<8, 8>        — 8×8 accumulator matrix
!coralnpu.vtype            — vtype configuration token
```

Direction-B (v0.7.2): vector-producing ops (`VLE32Op`, `VMulOp`, `VAddOp`,
etc.) now return `!coralnpu.vreg<e32, m1>` instead of `i32`. Reduction ops
(`VRedSumOp`, `VRedMaxOp`) still return `i32` (scalar output). Operands
accept `AnyType` during the transition period; a follow-up will tighten
constraints once all consumers are migrated.

### Lowering pipeline

```
TOSA MLIR
  │ --tosa-to-coralnpu        (33+ pattern-based TOSA→CoralNPU)
  ↓
CoralNPU IR (tensor carriers, regalloc attributes pending)
  │ --coralnpu-legalize        (tensor.from_elements housekeeping)
  ↓
  │ --coralnpu-regalloc        (linear-scan; annotates xreg_*/vreg_* attrs)
  ↓
  │ --emit-coralnpu-assembly   (RISC-V text assembly from annotated IR)
  ↓
RISC-V assembly
  │ riscv64-unknown-elf-as
  ↓
ELF binary               ← circt-translate --export-coralnpu produces this
```

`--tosa-to-coralnpu` is a shallow lowering: it emits representative CoralNPU
ops while threading tensor carriers through the IR for type correctness. The
final regalloc + emit passes handle all concrete register assignment and
instruction encoding.

### TOSA coverage

| Category | Patterns |
|----------|----------|
| Elementwise | add, sub, mul, negate, abs, clamp/relu |
| Activation  | sigmoid (LUT), cast (widen/narrow) |
| Quantization | rescale (scalar & per-channel), arithmetic_right_shift |
| Pooling | avg_pool2d (multi-tile), max_pool2d |
| Convolution | conv2d (any OC, any IC with k_in tiles), depthwise_conv2d (any KH×KW) |
| Reduce | reduce_sum, reduce_max, maximum, minimum |
| Data layout | concat, slice, reshape (passthrough), transpose (DMA placeholder) |
| Logic/compare | equal, greater, greater_eq, logical_and/or/not, select |
| Const injection | tosa.const → `coralnpu.const_data` module attr → ELF .data section |
| Misc | matmul (outer_product), table (LUT) |

### Register allocator

Linear-scan over SSA value intervals. Pools:
- 32 scalar registers (x0–x31), x0 hardwired to zero
- 64 vector registers (v0–v63)
- 8×8 accumulator matrix

`isVector` detection (v0.7.2): `mlir::isa<VRegType>(value.getType()) || opIsVector`
— type-based check takes priority; op-heuristic is fallback for unconverted ops.

### ELF export

`circt-translate --export-coralnpu` emits a 5-section ELF:

```
.text    — encoded CoralNPU instructions
.data    — TCM slot layout (slot N @ 0x10000 + N×0x1000)
.shstrtab / .symtab — standard ELF housekeeping
```

`.data` is zero-initialised by default; `tools/write_weights.py` reads
TFLite flatbuffers and patches real i8 weights into the correct slots.

---

## What we want from the community

### 1. Review of the type system

The current `!coralnpu.vreg<sew, lmul>` mirrors RISC-V `vtype` exactly.
Alternative considered: `!vector.type<4xi32>` with an attribute for LMUL.
We chose the custom type because:
- It keeps CoralNPU self-contained (no dependency on `vector` dialect)
- The `<sew, lmul>` parameterisation is directly visible to type-checker
- It is upstream-PR-friendly: a new dialect's types don't conflict with `vector`

Feedback welcome: should we use `vector.type` or keep `!coralnpu.vreg`?

### 2. The "shallow lowering" pattern

Our `tosa-to-coralnpu` uses tensor carriers (`tensor.splat` / `tensor.from_elements`)
to thread TOSA tensor types through the IR while emitting CoralNPU ops.
This avoids needing `memref` or `bufferization` at this stage. The tradeoff:

- **Pro**: single-pass pipeline, easy to understand, works for static shapes
- **Con**: not composable with MLIR's bufferization infrastructure; dynamic shapes need a different approach

CIRCT's existing approaches (e.g., FIRRTL's lowering to HW) are purely
structural, not compute-oriented. Is there a preferred CIRCT pattern for
this class of compute-dialect lowering?

### 3. PR split strategy

We plan to split the upstreaming into 3 PRs:

| PR | Contents | Size estimate |
|----|----------|---------------|
| PR-1 | Dialect skeleton: `CoralNPUOps.td`, `CoralNPUTypes.td`, `CoralNPUDialect`, boilerplate | ~1500 lines |
| PR-2 | Vector ops, RegAlloc pass, EmitAssembly pass, lit tests | ~3000 lines |
| PR-3 | TOSA lowering pass (33 patterns), ELF export, E2E tests | ~5000 lines |

Does this split make sense, or would reviewers prefer a different boundary?

### 4. Spike dependency

The E2E tests require a patched spike build (patches 0002–0006 from
`google-coral/coralnpu`). For CIRCT CI this would either need:
(a) a pre-built spike binary in the integration Docker image, or
(b) conditional test execution (`REQUIRES: coralnpu-spike`)

What is CIRCT's policy for optional external simulator dependencies?

---

## Current status

```
Version : v0.7.4  (branch feat/real-elementwise-lowering)
E2E     : 39/39 ✅ (spike bit-exact)
Lit     : 25/29 ✅ (4 skipped: hw-check/matmul passes unrelated to PR scope)
clang-format: applied ✅
assemblyFormat: all vector Ops use functional-type ✅
```

Representative pipeline output:

```mlir
// Input TOSA
%0 = tosa.conv2d %in, %wt, %bias, %izp, %wzp { ... }
    : (tensor<1x1x1x8xi8>, tensor<4x1x1x8xi8>, ...) -> tensor<1x1x1x4xi32>

// After --tosa-to-coralnpu
coralnpu.vsetvl e32, m1
%t0 = coralnpu.vle32 %addr0, %n4 : !coralnpu.vreg<e32, m1>
%t1 = coralnpu.vle32 %addr1, %n4 : !coralnpu.vreg<e32, m1>
%p0 = coralnpu.vmul %t0, %t1 : (!coralnpu.vreg<e32, m1>, ...) -> !coralnpu.vreg<e32, m1>
%s0 = coralnpu.vredsum %p0 : (!coralnpu.vreg<e32, m1>) -> i32
...
coralnpu.sw %result, %addr_out : i32

// After --emit-coralnpu-assembly → spike output
vle32.v v0, (x1)
vle32.v v1, (x2)
vmul.vv v2, v0, v1
vredsum.vs v0, v2, v0
vmv.x.s x4, v0
...
sw x4, 0(x2)
```

---

## Links

- Repository: https://github.com/lipeng821020589/Circt4Coral  
  branch `feat/real-elementwise-lowering`
- Hardware: https://github.com/google-coral/coralnpu  
  spike patches 0002–0006 required for E2E
- Previous Discourse thread (June 2026):
  https://discourse.llvm.org/t/circt4coral-tosa-to-coralnpu/
- MLIRConf 2026 abstract (submitted): `blog/MLIRConf-2026-Abstract.md`

---

## About

Built by **peng** ([@lipeng821020589](https://github.com/lipeng821020589)).
The project started as a 2-day exploration and grew to a full lowering stack
through July 2026. Reach out on LLVM Discourse or GitHub Issues.

*Apache 2.0 license, same as CIRCT.*
