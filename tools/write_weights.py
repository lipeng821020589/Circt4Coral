#!/usr/bin/env python3
"""
write_weights.py — 从 TFLite flatbuffer 提取权重，注入 Coral NPU ELF .data section。

用法：
  python3 write_weights.py --tflite MODEL.tflite --elf INPUT.elf --out OUTPUT.elf \
                           [--slot-map 1=0,2=1,3=2]   # arg_slot=tensor_idx

  --list-tensors   列出所有 tensor（buf_idx, bytes, type, 可读名字）后退出

.data section 布局（与 TosaToCoralNPU.cpp 约定一致）：
  slot N = TCM 地址 0x10000 + N * 0x1000（4KB/slot）

--slot-map: argSlot=tensorIdx 逗号分隔，把 MLIR 函数参数 N 对应的 TCM slot
映射到 TFLite tensor index M 的数据。省略则按 tensor 顺序从 slot 0 开始。
"""

import argparse
import struct
import sys
from pathlib import Path

import flatbuffers.table as fbt
import flatbuffers.number_types as N
import flatbuffers.encode as E

TCM_BASE = 0x10000
TCM_SLOT = 0x1000
ELF_MAGIC = b'\x7fELF'

# ── TFLite flatbuffer parser ──────────────────────────────────────────────────

TFLITE_TYPE = {0: 'f32', 1: 'f16', 2: 'i8', 3: 'i32', 4: 'u8', 9: 'i64'}

def _tbl(ba, pos):
    return fbt.Table(ba, pos)

def _vec_of_tables(ba, t, vtoff):
    """Return list of Table objects from a vector-of-tables field at vtable_offset."""
    off = t.Offset(vtoff)
    if not off:
        return []
    count = t.VectorLen(off)
    vec = t.Vector(off)
    return [_tbl(ba, vec + i * 4 + struct.unpack_from('<I', ba, vec + i * 4)[0])
            for i in range(count)]

def _bytes_field(ba, t, vtoff):
    off = t.Offset(vtoff)
    if not off:
        return b''
    return bytes(ba[t.Vector(off): t.Vector(off) + t.VectorLen(off)])

def _uint_field(t, vtoff):
    off = t.Offset(vtoff)
    if not off:
        return 0
    return struct.unpack_from('<I', t.Bytes, t.Pos + off)[0]

def _str_field(t, vtoff):
    off = t.Offset(vtoff)
    if not off:
        return ''
    try:
        raw = t.String(off)
        return raw.decode('utf-8', 'replace')
    except Exception:
        return ''


def parse_tflite(data: bytes):
    """
    Parse TFLite flatbuffer.
    Returns:
      buffers: list of bytes (raw buffer data, may be empty)
      tensors: list of (name, type_id, buffer_idx, raw_bytes)
    TFLite schema vtable offsets (fixed by schema compiler):
      Model:    version=4, operator_codes=6, subgraphs=8, description=10, buffers=12
      SubGraph: tensors=4, inputs=6, outputs=8, operators=10, name=12
      Tensor:   name=4, shape=6, type=8, buffer=10, ...
      Buffer:   data=4
    """
    ba = bytearray(data)
    root = E.Get(N.UOffsetTFlags.packer_type, ba, 0)
    model = _tbl(ba, root)

    # Model.buffers = vtable_offset 12 (field index 4)
    buf_tables = _vec_of_tables(ba, model, 12)
    buffers = [_bytes_field(ba, b, 4) for b in buf_tables]

    # Model.subgraphs = vtable_offset 8 (field index 2)
    sg_tables = _vec_of_tables(ba, model, 8)
    tensors = []
    if sg_tables:
        sg0 = sg_tables[0]
        # SubGraph.tensors = vtable_offset 4 (field index 0)
        t_tables = _vec_of_tables(ba, sg0, 4)
        for t in t_tables:
            name   = _str_field(t, 4)    # Tensor.name   vtoff=4
            typ    = _uint_field(t, 8)   # Tensor.type   vtoff=8
            bidx   = _uint_field(t, 10)  # Tensor.buffer vtoff=10
            raw    = buffers[bidx] if bidx < len(buffers) else b''
            tensors.append((name, typ, bidx, raw))

    return buffers, tensors


# ── ELF .data patcher ─────────────────────────────────────────────────────────

def patch_elf_data(elf: bytes, slot_data: dict) -> bytes:
    """
    Patch .data section of a 32-bit LE ELF in-place.
    slot_data: {arg_slot_index: raw_bytes}
    Returns patched ELF bytes.
    """
    if elf[:4] != ELF_MAGIC:
        raise ValueError("Not an ELF file")
    elf = bytearray(elf)
    if elf[4] != 1:
        raise ValueError(f"Only ELF32 supported (class={elf[4]})")

    e_shoff    = struct.unpack_from('<I', elf, 0x20)[0]
    e_shentsize = struct.unpack_from('<H', elf, 0x2e)[0]
    e_shnum    = struct.unpack_from('<H', elf, 0x30)[0]
    e_shstrndx = struct.unpack_from('<H', elf, 0x32)[0]

    shstr_hdr = e_shoff + e_shstrndx * e_shentsize
    shstr_off  = struct.unpack_from('<I', elf, shstr_hdr + 0x10)[0]
    shstr_size = struct.unpack_from('<I', elf, shstr_hdr + 0x14)[0]
    shstrtab = bytes(elf[shstr_off: shstr_off + shstr_size])

    def sec_name(sh_name):
        end = shstrtab.index(b'\x00', sh_name)
        return shstrtab[sh_name:end].decode()

    data_off = data_size = 0
    data_hdr_pos = None
    for i in range(e_shnum):
        hdr = e_shoff + i * e_shentsize
        if sec_name(struct.unpack_from('<I', elf, hdr)[0]) == '.data':
            data_off  = struct.unpack_from('<I', elf, hdr + 0x10)[0]
            data_size = struct.unpack_from('<I', elf, hdr + 0x14)[0]
            data_hdr_pos = hdr
            break

    if data_off == 0:
        raise ValueError("No .data section found in ELF")

    # Ensure .data is large enough
    needed = max(
        slot * TCM_SLOT + len(raw)
        for slot, raw in slot_data.items()
    )
    if needed > data_size:
        extra = needed - data_size
        insert_at = data_off + data_size
        elf = elf[:insert_at] + bytearray(extra) + elf[insert_at:]
        # Update .data size in section header
        if data_hdr_pos is not None:
            struct.pack_into('<I', elf, data_hdr_pos + 0x14, needed)
        # Shift e_shoff if headers come after inserted region
        if e_shoff >= insert_at:
            struct.pack_into('<I', elf, 0x20, e_shoff + extra)
        data_size = needed

    for slot, raw in slot_data.items():
        start = data_off + slot * TCM_SLOT
        elf[start: start + len(raw)] = raw

    return bytes(elf)


# ── dtype converters ──────────────────────────────────────────────────────────

def i8_to_i32_le(raw: bytes) -> bytes:
    """Sign-extend each i8 byte to i32 LE (4 bytes per element)."""
    out = bytearray(len(raw) * 4)
    for i, b in enumerate(raw):
        v = b if b < 128 else b - 256
        struct.pack_into('<i', out, i * 4, v)
    return bytes(out)

def raw_i32_le(raw: bytes) -> bytes:
    """Pass i32 LE data through, padding to 4-byte boundary."""
    pad = (-len(raw)) % 4
    return raw + b'\x00' * pad


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tflite', required=True)
    ap.add_argument('--elf',    default='',  help='Input ELF (required unless --list-tensors)')
    ap.add_argument('--out',    default='',  help='Output ELF (required unless --list-tensors)')
    ap.add_argument('--slot-map', default='',
                    help='Comma-separated argSlot=tensorIdx pairs, e.g. 1=2,2=3')
    ap.add_argument('--list-tensors', action='store_true')
    ap.add_argument('--dtype', default='i8', choices=['i8', 'i32'])
    args = ap.parse_args()

    data = Path(args.tflite).read_bytes()
    buffers, tensors = parse_tflite(data)

    if args.list_tensors:
        print(f"{'idx':>4}  {'buf':>4}  {'bytes':>8}  {'type':>4}  name")
        print('-' * 60)
        for i, (name, typ, bidx, raw) in enumerate(tensors):
            tname = TFLITE_TYPE.get(typ, str(typ))
            display = repr(name) if name and not name.isprintable() else name
            print(f"{i:4d}  {bidx:4d}  {len(raw):8d}  {tname:>4}  {display[:30]}")
        return

    if not args.elf or not args.out:
        ap.error("--elf and --out are required")

    slot_data: dict = {}
    convert = i8_to_i32_le if args.dtype == 'i8' else raw_i32_le

    if args.slot_map:
        for pair in args.slot_map.split(','):
            pair = pair.strip()
            if not pair:
                continue
            slot_str, tidx_str = pair.split('=')
            slot = int(slot_str.strip())
            tidx = int(tidx_str.strip())
            name, typ, bidx, raw = tensors[tidx]
            if not raw:
                print(f"WARNING: tensor {tidx} has no data, skipping slot {slot}")
                continue
            out_bytes = convert(raw)
            slot_data[slot] = out_bytes
            tname = TFLITE_TYPE.get(typ, str(typ))
            print(f"  slot {slot:2d} ← tensor {tidx:3d} ({tname}, {len(raw)} bytes"
                  f" → {len(out_bytes)} i32 bytes)")
    else:
        slot = 0
        for i, (name, typ, bidx, raw) in enumerate(tensors):
            if raw:
                out_bytes = convert(raw)
                slot_data[slot] = out_bytes
                tname = TFLITE_TYPE.get(typ, str(typ))
                print(f"  slot {slot:2d} ← tensor {i:3d} ({tname}, {len(raw)} bytes"
                      f" → {len(out_bytes)} i32 bytes)")
                slot += 1

    if not slot_data:
        print("No weight data found; ELF not modified.")
        sys.exit(1)

    elf = Path(args.elf).read_bytes()
    patched = patch_elf_data(elf, slot_data)
    Path(args.out).write_bytes(patched)
    print(f"\nWrote {len(patched)} bytes → {args.out}")
    print(f"Patched {len(slot_data)} TCM slot(s).")


if __name__ == '__main__':
    main()
