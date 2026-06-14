# MLIRConf 2026 Abstract Submission

## Title
**From TOSA to google-coral/coralnpu: A Standalone MLIR Compiler in 2 Days**

## Talk Type
25-minute technical talk (or 15-minute lightning)

## Abstract (250 words)

We built `circt4coral`, a standalone MLIR compiler that lowers TOSA → CoralNPU dialect → RISC-V assembly with RVV + custom CSRs, in 2 days. The entire compiler is a 211 MB single binary that runs without an LLVM/MLIR install on the user side. We demonstrate a **9/9 end-to-end test pass** on spike + CoralNPU patches, including two TOSA op lowers (add, mul) that go through the full pipeline.

`google-coral/coralnpu` is a 6-month-old open-source RISC-V Vector + custom-CSR accelerator (announced November 2025). Despite MLIR/IREE being the official compiler stack, **no public MLIR→CoralNPU e2e flow exists**. This talk shows what we built to fill that gap.

We cover 5 lessons learned:
1. **Never invent opcodes** — use RISC-V CSR escape hatch for custom semantics
2. **Standalone binary > plugin** — when the LLVM submodule ABI breaks, partial-link + gold linker is your friend
3. **DCE eats side-effect ops** — even without `[Pure]`, ops with no result get removed; use `MemoryEffectOpInterface`
4. **spike is not a cycle-accurate model** — V extension pipelining isn't simulated; real hardware will be 4-8× faster
5. **Build a Docker teaching image** — 1.5 GB image, 5-minute onboarding, 10× community reach

We share the full source under Apache 2.0 and a 211 MB release binary at github.com/lipeng821020589/Circt4Coral/releases.

## Topics Covered
- MLIR dialect design (CoralNPU as extension of `vector` + `tosa` profile)
- Standalone binary build techniques (partial link + gold linker + orphan libraries)
- Custom CSR codegen for accelerator control
- TOSA lowering patterns
- Docker packaging for MLIR tools

## Target Audience
MLIR compiler engineers, RISC-V accelerator developers, embedded ML practitioners.

## Speaker Bio
**peng** is an independent ML/compiler researcher. Built circt4coral as a 2-day spike to explore the gap between the official google-coral/coralnpu compiler (LiteRT-micro) and the MLIR ecosystem. Reach out on MLIR Discourse or GitHub.

## Links
- Code: https://github.com/lipeng821020589/Circt4Coral
- Release: https://github.com/lipeng821020589/Circt4Coral/releases/tag/v0.1.0
- Blog: https://discourse.llvm.org/t/circt4coral-tosa-to-coralnpu (forthcoming)

## Submission Notes
- MLIRConf 2026 is in October (typically)
- Abstract deadline: usually July/August
- This submission targets: short talk (15-20 min) + Q&A
- Slides will be made available as PDF + source

## Why this talk matters
1. **First public MLIR→CoralNPU demo** — fills a real gap in the ecosystem
2. **Practical lessons** — the 5 lessons are not coralnpu-specific; they apply to any MLIR→custom-ISA project
3. **Replicable** — anyone can clone the repo, run the Docker image, and get started in 5 minutes
4. **Strategic positioning** — establishes circt4coral as the go-to MLIR frontend for CoralNPU before google-coral ships their own
