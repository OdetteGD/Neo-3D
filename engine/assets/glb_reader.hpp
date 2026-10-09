#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace neo3d::assets {
struct GlbDocument { std::string_view json; std::vector<std::uint8_t> binary; };
inline std::uint32_t readU32(const std::uint8_t* p) { return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1])<<8u) | (static_cast<std::uint32_t>(p[2])<<16u) | (static_cast<std::uint32_t>(p[3])<<24u); }
// Parses the GLB container only; JSON semantics, accessor validation, and GPU upload belong to the importer layer.
inline GlbDocument readGlb(const std::vector<std::uint8_t>& bytes) {
    constexpr std::uint32_t magic=0x46546C67u, jsonChunk=0x4E4F534Au, binChunk=0x004E4942u;
    if (bytes.size()<12) throw std::runtime_error("GLB header is truncated");
    if (readU32(bytes.data())!=magic) throw std::runtime_error("Invalid GLB magic; expected glTF");
    if (readU32(bytes.data()+4)!=2u) throw std::runtime_error("Only GLB version 2 is supported");
    const std::uint32_t declared=readU32(bytes.data()+8);
    if (declared!=bytes.size()) throw std::runtime_error("GLB declared length does not match input size");
    std::size_t cursor=12; GlbDocument doc; bool sawJson=false, sawBin=false;
    while (cursor<bytes.size()) {
        if (bytes.size()-cursor<8) throw std::runtime_error("GLB chunk header is truncated");
        const std::uint32_t length=readU32(bytes.data()+cursor), type=readU32(bytes.data()+cursor+4); cursor+=8;
        if (length>bytes.size()-cursor) throw std::runtime_error("GLB chunk exceeds container bounds");
        const auto* chunk=bytes.data()+cursor;
        if (type==jsonChunk) {
            if (sawJson || cursor!=20) throw std::runtime_error("GLB JSON chunk must be the first and only JSON chunk");
            doc.json=std::string_view(reinterpret_cast<const char*>(chunk),length); sawJson=true;
        } else if (type==binChunk) {
            // Track presence independently of payload size: a zero-length BIN chunk
            // is still a chunk and must not permit a second BIN chunk.
            if (!sawJson || sawBin) throw std::runtime_error("GLB BIN chunk must follow JSON and appear at most once");
            sawBin=true;
            doc.binary.assign(chunk,chunk+length);
        }
        cursor+=length;
        if (cursor%4u!=0u && cursor<bytes.size()) throw std::runtime_error("GLB chunk length is not 4-byte aligned");
    }
    if (!sawJson || doc.json.empty()) throw std::runtime_error("GLB is missing its JSON chunk");
    return doc;
}
}
