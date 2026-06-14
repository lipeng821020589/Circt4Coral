# circt4coral Docker 教学镜像

5 分钟内跑通 MLIR → google-coral/coralnpu RISC-V 汇编 → spike 仿真。

## 构建

```bash
make -f docker/Makefile build
```

需要先把 `bin/circt4coral` 复制到 `docker/circt4coral`：

```bash
cp bin/circt4coral docker/circt4coral
make -f docker/Makefile build
```

## 运行

```bash
make -f docker/Makefile run
```

进入容器后：

```bash
circt4coral /workspace/examples/hello.mlir --emit-asm
circt4coral /workspace/examples/vector-add.mlir --emit-asm
circt4coral /workspace/examples/tosa-add.mlir --emit-asm
```

## 跑测试

```bash
make -f docker/Makefile test
```

## 镜像内容

- `circt4coral` 编译器 (211 MB 静态二进制)
- spike RISC-V 仿真器 (RVV 启用)
- gcc-riscv64-unknown-elf 工具链
- dtc (device tree compiler)
- 示例代码 (`/workspace/examples/`)
- 测试套件 (`/workspace/tests/`)

镜像大小：~1.5GB

## 上游依赖

- Ubuntu 22.04
- riscv-software-src/riscv-isa-sim (master 分支)
- google-coral/coralnpu (CSR 地址规范)
- RISC-V apt 包: gcc-riscv64-unknown-elf
