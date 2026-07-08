#!/usr/bin/env python3
"""
run_tflite_layer.py — 完整端到端流水线：
  TFLite 权重 → MLIR → circt-opt assembly → ELF .data 注入 → spike

验证：从 TFLite flatbuffer 提取 buf[24]（1152 bytes i8）的前 4 个字节作为
conv2d 权重，构造一个 1x1 pointwise conv2d，通过 write_weights 注入 ELF，
然后 spike 运行，对比手算结果。

这是方向 A 的完整端到端验证：TOSA MLIR → CoralNPU ISA + 真实权重。
"""

import sys, os, struct, tempfile, subprocess
from pathlib import Path

sys.path.insert(0, '/home/radxa/Work/Circt4Coral/bench')
sys.path.insert(0, '/home/radxa/Work/Circt4Coral/tools')

# ── 常量（与 e2e_validate.py 一致）───────────────────────────────────────────
CIRCT_OPT   = Path("/home/radxa/Work/Circt4Coral/build/bin/circt-opt")
CIRCT_TRANS = Path("/home/radxa/Work/Circt4Coral/build/bin/circt-translate")
SPIKE       = Path("/home/radxa/Work/riscv-isa-sim/build/spike")
AS  = "riscv64-unknown-elf-as"
LD  = "riscv64-unknown-elf-ld"
TCM_BASE    = 0x10000
TCM_SLOT    = 0x1000
SPIKE_ISA   = "rv32imf_zve32f_zvl128b_zicsr_zifencei_zbb"
SPIKE_MEM   = "0x0:0x100000,0x20000000:0x4000000"
PASSES      = ["--tosa-to-coralnpu","--coralnpu-legalize",
               "--coralnpu-regalloc","--emit-coralnpu-assembly"]
TFLITE      = Path("/home/radxa/Work/coralnpu/tests/cocotb/tutorial/tfmicro/models/"
                   "mobilenet_v1_0.25_224_int8_dummy.tflite")

# ── 从 e2e_validate.py 复用函数 ───────────────────────────────────────────────
exec(open('/home/radxa/Work/Circt4Coral/bench/e2e_validate.py').read().split('if __name__')[0])
from write_weights import parse_tflite, patch_elf_data, i8_to_i32_le

# ── Step 1: 从 TFLite 提取 buf[24]（第一个 conv2d 权重）─────────────────────
print("Step 1: 从 TFLite 提取真实 i8 权重 buf[24]...")
tflite_data = TFLITE.read_bytes()
buffers, tensors = parse_tflite(tflite_data)

# buf[24] = 1152 bytes i8, 第一个 conv2d 权重（MobileNet v1 0.25x）
wt_raw = buffers[24]
print(f"  buf[24]: {len(wt_raw)} bytes, first 4 i8: {list(struct.unpack('4b', wt_raw[:4]))}")

# 取前 4 个 i8 作为 1x1 conv2d（IC=4, OC=1）的权重
wt_4 = list(struct.unpack('4b', wt_raw[:4]))  # signed i8
print(f"  使用前 4 个权重: {wt_4}")

# ── Step 2: 构造 MLIR（conv2d IC=4, OC=1, 输入全 1）────────────────────────
print("\nStep 2: 构造 MLIR...")

# 输入固定为 [1,1,1,1]，权重来自 TFLite，bias=0
# 期望结果 = dot([1,1,1,1], wt_4) = sum(wt_4)
expected = sum(wt_4)
print(f"  in=[1,1,1,1], wt={wt_4}, bias=0 → expected={expected}")

MLIR = f"""\
module {{
func.func @pw_conv_tflite(
  %in: tensor<1x1x1x4xi8>, %wt: tensor<1x1x1x4xi8>,
  %bias: tensor<1xi32>, %izp: tensor<1xi8>, %wzp: tensor<1xi8>
) -> tensor<1x1x1x1xi32> {{
  %0 = tosa.conv2d %in, %wt, %bias, %izp, %wzp {{
    acc_type = i32, dilation = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>
  }} : (tensor<1x1x1x4xi8>, tensor<1x1x1x4xi8>, tensor<1xi32>,
       tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x1x1xi32>
  func.return %0 : tensor<1x1x1x1xi32>
}}
}}
"""

# ── Step 3: circt-opt 生成汇编 ───────────────────────────────────────────────
print("Step 3: circt-opt TOSA → CoralNPU assembly...")
insns = extract_asm_instructions(run_circt_opt(MLIR, PASSES))
print(f"  {len(insns)} 条汇编指令")

# ── Step 4: 构建 ELF（.data 在 0x10000，.text 在 0x20000）──────────────────
print("Step 4: 构建 ELF（.data@0x10000, .text@0x20000）...")

prologue_hdr = [
    ".section .text", ".globl _start", "_start:",
    "    csrr  t0, mstatus", "    li    t1, 0x600",
    "    or    t0, t0, t1",  "    csrw  mstatus, t0",
]
data_init = "\n.section .data\n.balign 4\n.zero 36864\n"  # 9 × 0x1000
full_asm = "\n".join(prologue_hdr) + "\n" + "\n".join(insns) + "\n.Lexit:\n    ebreak\n" + data_init

ld_script = """
SECTIONS {
  . = 0x10000;
  .data : { *(.data*) *(.bss*) }
  . = 0x20000;
  .text : { *(.text*) }
}
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.S', delete=False) as af:
    af.write(full_asm); asm_file = af.name
obj_file = asm_file.replace('.S', '.o')
ld_file  = asm_file.replace('.S', '.ld')
open(ld_file, 'w').write(ld_script)

with tempfile.NamedTemporaryFile(suffix='.elf', delete=False) as ef:
    elf_path = Path(ef.name)

subprocess.run([AS, f'-march={SPIKE_ISA}', '-mabi=ilp32',
                '-o', obj_file, asm_file], check=True, capture_output=True)
subprocess.run([LD, '-m', 'elf32lriscv', '--no-dynamic-linker', '-static',
                '-e', '_start', '-T', ld_file,
                '-o', str(elf_path), obj_file], check=True, capture_output=True)
for p in [asm_file, obj_file, ld_file]:
    os.unlink(p)
print(f"  ELF 构建完成: {elf_path.stat().st_size} bytes")

# ── Step 5: 用 write_weights 注入真实 TFLite 权重 ───────────────────────────
print("Step 5: write_weights.patch_elf_data 注入真实权重...")

# slot 布局（5 args，result_slot=8）:
#   slot 0: in   = [1, 1, 1, 1]
#   slot 1: wt   = wt_4 (前4个 i8，sign-extend to i32)
#   slot 2: bias = [0]
#   slot 3: izp  = [0]
#   slot 4: wzp  = [0]

# 将 i8 权重 sign-extend 到 i32，放在 slot 1
def i8_list_to_i32_bytes(vals, pad_to=16):
    out = struct.pack(f'<{len(vals)}i', *vals)
    if len(vals) < pad_to:
        out += b'\x00' * (4 * (pad_to - len(vals)))
    return out

slot_data = {
    0: i8_list_to_i32_bytes([1, 1, 1, 1]),   # in
    1: i8_list_to_i32_bytes(wt_4),           # wt (真实 TFLite 权重)
    2: i8_list_to_i32_bytes([0]),            # bias
    3: i8_list_to_i32_bytes([0]),            # izp
    4: i8_list_to_i32_bytes([0]),            # wzp
}

print(f"  slot 1 (wt): {wt_4} → i32: {list(struct.unpack('<4i', slot_data[1][:16]))[:4]}")

elf_bytes   = elf_path.read_bytes()
patched     = patch_elf_data(elf_bytes, slot_data)
patched_path = Path(str(elf_path) + '.patched')
patched_path.write_bytes(patched)
print(f"  注入完成: {patched_path.stat().st_size} bytes")

# ── Step 6: spike 运行 ───────────────────────────────────────────────────────
print("Step 6: spike 运行，读取 result slot 0x18000...")

RESULT_ADDR = TCM_BASE + 8 * TCM_SLOT  # 0x18000
raw = run_spike_and_read_mem(patched_path, RESULT_ADDR, 4)
actual = struct.unpack('<i', raw)[0]

# ── 输出结果 ─────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"输入:    in=[1,1,1,1]")
print(f"权重:    wt={wt_4}  (TFLite buf[24] 前4字节 i8)")
print(f"期望:    dot([1,1,1,1], {wt_4}) + bias(0) = {expected}")
print(f"实际:    {actual}")
if actual == expected:
    print("✅  PASS — TFLite 权重注入 + spike 数值 bit-exact")
else:
    print(f"❌  FAIL — 期望 {expected}，实际 {actual}")

for p in [elf_path, patched_path]:
    if p.exists(): p.unlink()
