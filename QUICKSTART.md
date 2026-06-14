# circt4coral Quickstart

5 分钟上手指南：从 TOSA MLIR 编译到 spike 跑通。

## 前置条件

- Ubuntu 22.04 / Debian 12 / 任何 glibc 2.35+ Linux
- gcc-riscv64-unknown-elf apt 包
- spike 1.1.1-dev with RVV
- ~500MB 磁盘空间

## 安装

```bash
# 1. 克隆仓库（包含预构建的 circt4coral）
git clone https://github.com/lipeng821020589/circt4coral.git
cd circt4coral

# 2. 安装 RISC-V 工具链
sudo apt install gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf

# 3. 验证
./bin/circt4coral --help
```

## 编译第一个 MLIR

创建一个文件 `hello.mlir`：

```mlir
func.func @hello() -> i32 {
  %a = coralnpu.li 42 : i32
  %b = coralnpu.li 7 : i32
  %c = coralnpu.add %a, %b : i32
  coralnpu.return %c : i32
}
```

编译到汇编：

```bash
./bin/circt4coral hello.mlir --emit-asm
```

输出：

```asm
.section .text
.globl _start
_start:
addi   x1, zero, 42
addi   x2, zero, 7
add    x3, x1, x2
ret
.L0:
ebreak
```

## 汇编到 spike

```bash
# 汇编
riscv64-linux-gnu-as -march=rv32imv_zicsr -mabi=ilp32 -o hello.o <(cat << 'EOF'
.section .text
.globl _start
_start:
    # Enable V extension
    li t0, 0x200
    csrs 0x300, t0
addi   x1, zero, 42
addi   x2, zero, 7
add    x3, x1, x2
ret
.L0:
ebreak
