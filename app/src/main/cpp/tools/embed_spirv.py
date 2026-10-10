#!/usr/bin/env python3
import pathlib, sys
out = pathlib.Path(sys.argv[1])
entries = [(sys.argv[2], "kMeshVert"), (sys.argv[3], "kMeshFrag"), (sys.argv[4], "kSkyVert"), (sys.argv[5], "kSkyFrag"), (sys.argv[6], "kShadowVert")]
lines = ["#pragma once", "#include <cstddef>", "#include <cstdint>"]
for source, name in entries:
    data = pathlib.Path(source).read_bytes()
    if len(data) % 4:
        raise SystemExit(f"{source}: SPIR-V size must be a multiple of four")
    words = [int.from_bytes(data[i:i+4], "little") for i in range(0, len(data), 4)]
    lines.append(f"inline constexpr std::uint32_t {name}[] = {{")
    for i in range(0, len(words), 8):
        lines.append("    " + ", ".join(f"0x{x:08x}u" for x in words[i:i+8]) + ",")
    lines.append("};")
    lines.append(f"inline constexpr std::size_t {name}Size = sizeof({name});")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
