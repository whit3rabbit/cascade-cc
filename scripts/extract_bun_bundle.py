#!/usr/bin/env python3
"""Extract Bun standalone bundle contents from a Mach-O Claude binary."""

import argparse
import os
import struct
import sys
from pathlib import Path

LC_SEGMENT_64 = 0x19
MACHO_MAGIC_64 = 0xFEEDFACF
MACHO_MAGIC_64_BE = 0xCFFAEDFE


def read_u32(data, offset, endian):
    return struct.unpack_from(endian + "I", data, offset)[0]


def read_u64(data, offset, endian):
    return struct.unpack_from(endian + "Q", data, offset)[0]


def find_bun_section(binary_path):
    data = Path(binary_path).read_bytes()
    if len(data) < 32:
        raise ValueError("Binary too small to be Mach-O.")

    magic = struct.unpack_from("<I", data, 0)[0]
    if magic == MACHO_MAGIC_64:
        endian = "<"
    elif magic == MACHO_MAGIC_64_BE:
        endian = ">"
    else:
        raise ValueError("Unsupported binary format (expected Mach-O 64-bit).")

    ncmds = read_u32(data, 16, endian)
    offset = 32

    for _ in range(ncmds):
        cmd = read_u32(data, offset, endian)
        cmdsize = read_u32(data, offset + 4, endian)
        if cmd == LC_SEGMENT_64:
            segname = data[offset + 8: offset + 24].split(b"\x00", 1)[0].decode("utf-8", "ignore")
            nsects = read_u32(data, offset + 64, endian)
            sect_offset = offset + 72
            if segname == "__BUN":
                for _ in range(nsects):
                    sectname = data[sect_offset: sect_offset + 16].split(b"\x00", 1)[0].decode("utf-8", "ignore")
                    segname2 = data[sect_offset + 16: sect_offset + 32].split(b"\x00", 1)[0].decode("utf-8", "ignore")
                    size = read_u64(data, sect_offset + 40, endian)
                    fileoff = read_u32(data, sect_offset + 48, endian)
                    if segname2 == "__BUN" and sectname == "__bun":
                        return data[fileoff:fileoff + size]
                    sect_offset += 80
        offset += cmdsize

    raise ValueError("Failed to locate __BUN,__bun section in Mach-O.")


def iter_bun_paths(blob):
    prefixes = [b"file:///", b"/$bunfs/root/"]
    for prefix in prefixes:
        start = 0
        while True:
            idx = blob.find(prefix, start)
            if idx == -1:
                break
            end = blob.find(b"\x00", idx)
            if end == -1:
                start = idx + len(prefix)
                continue
            raw_path = blob[idx:end]
            if idx >= 4:
                length = struct.unpack_from("<I", blob, idx - 4)[0]
                if length != len(raw_path):
                    start = end + 1
                    continue
            try:
                path = raw_path.decode("utf-8")
            except UnicodeDecodeError:
                start = end + 1
                continue
            yield path, idx, end
            start = end + 1


def parse_entry(blob, path_end):
    cursor = path_end
    while cursor < len(blob) and blob[cursor] == 0:
        cursor += 1
    if cursor + 16 > len(blob):
        return None
    size = struct.unpack_from("<I", blob, cursor + 12)[0]
    data_start = cursor + 16
    data_end = data_start + size
    if size == 0 or data_end > len(blob):
        return None
    return data_start, size


def normalize_path(raw_path):
    if raw_path.startswith("file:///"):
        rel = raw_path[len("file:///"):]
    elif raw_path.startswith("file://"):
        rel = raw_path[len("file://"):]
    elif raw_path.startswith("/$bunfs/root/"):
        rel = os.path.join("bunfs_root", raw_path[len("/$bunfs/root/"):])
    else:
        rel = raw_path.lstrip("/")
    return rel.replace("..", "__")


def extract_entries(blob, outdir, targets, extract_all=False):
    outdir.mkdir(parents=True, exist_ok=True)
    extracted = []
    for raw_path, _, path_end in iter_bun_paths(blob):
        if not extract_all and raw_path not in targets:
            continue
        entry = parse_entry(blob, path_end)
        if not entry:
            continue
        data_start, size = entry
        rel_path = normalize_path(raw_path)
        dest = outdir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob[data_start:data_start + size])
        extracted.append((raw_path, str(dest), size))
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract Bun standalone bundle contents.")
    parser.add_argument("version", nargs="?", help="Version folder under claude-analysis/")
    parser.add_argument("--version", dest="version_flag", help="Version folder under claude-analysis/")
    parser.add_argument("--binary", help="Path to Claude binary (defaults to claude-analysis/<version>/binary/claude)")
    parser.add_argument("--outdir", help="Output directory (defaults to claude-analysis/<version>/binary)")
    parser.add_argument("--all", action="store_true", help="Extract all detected bunfs entries")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted files")
    args = parser.parse_args()

    version = args.version_flag or args.version
    if not args.binary and not version:
        print("[!] Provide --version <ver> or --binary <path>.")
        return 1

    binary_path = args.binary
    if not binary_path:
        binary_path = os.path.join("claude-analysis", version, "binary", "claude")
        if sys.platform == "win32":
            exe_path = binary_path + ".exe"
            if os.path.exists(exe_path):
                binary_path = exe_path
        if not os.path.exists(binary_path):
            print(f"[!] Binary not found: {binary_path}")
            return 1

    outdir = args.outdir
    if not outdir:
        if version:
            outdir = os.path.join("claude-analysis", version, "binary")
        else:
            outdir = os.path.join("claude-analysis", "unknown", "binary")

    try:
        blob = find_bun_section(binary_path)
    except Exception as exc:
        print(f"[!] Failed to locate Bun section: {exc}")
        return 1

    targets = {
        "file:///src/entrypoints/cli.js.jsc",
        "file:///src/entrypoints/cli.js",
        "file:///src/entrypoints/cli.mjs",
    }

    outdir_path = Path(outdir)
    cli_dest = outdir_path / "cli.js"
    if cli_dest.exists() and not args.force and not args.all:
        print(f"[=] Already extracted: {cli_dest} (use --force to overwrite)")
        return 0

    extracted = extract_entries(blob, outdir_path, targets, extract_all=args.all)
    if not extracted:
        print("[!] No entries extracted. Try --all if format changed.")
        return 1

    print("[+] Extracted:")
    for raw_path, dest, size in extracted:
        print(f"    - {raw_path} -> {dest} ({size} bytes)")

    # Keep cli.js.jsc intact for bytecode analysis; do not rename to cli.js.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
