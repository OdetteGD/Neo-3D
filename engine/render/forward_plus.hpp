#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace neo3d::render {
struct ScreenLight { float x=0, y=0, radiusPixels=0; float viewDepth=1; float depthRadius=1; };
struct TileLights { std::vector<std::uint16_t> indices; float minDepth=0, maxDepth=0; };

// Conservative screen-space tile assignment. Coordinates are pixels with origin at top-left.
// The renderer can upload each tile's compact index list to a storage buffer for fragment shading.
inline std::vector<TileLights> buildLightTiles(std::uint32_t width, std::uint32_t height,
                                               std::uint32_t tileSize,
                                               const std::vector<ScreenLight>& lights,
                                               std::uint32_t maxLightsPerTile=256) {
    if (width==0 || height==0 || tileSize==0) return {};
    const std::uint32_t cols=(width+tileSize-1)/tileSize, rows=(height+tileSize-1)/tileSize;
    std::vector<TileLights> tiles(static_cast<std::size_t>(cols)*rows);
    for (std::uint32_t ty=0; ty<rows; ++ty) for (std::uint32_t tx=0; tx<cols; ++tx) {
        auto& tile=tiles[static_cast<std::size_t>(ty)*cols+tx];
        const float left=static_cast<float>(tx*tileSize), top=static_cast<float>(ty*tileSize);
        const float right=static_cast<float>(std::min(width,(tx+1)*tileSize));
        const float bottom=static_cast<float>(std::min(height,(ty+1)*tileSize));
        tile.minDepth=1.0e30f; tile.maxDepth=0.0f;
        for (std::uint32_t i=0; i<lights.size() && i<65535u; ++i) {
            const auto& l=lights[i]; const float nx=std::max(left,std::min(l.x,right)); const float ny=std::max(top,std::min(l.y,bottom));
            const float dx=l.x-nx, dy=l.y-ny;
            if (dx*dx+dy*dy > l.radiusPixels*l.radiusPixels) continue;
            if (tile.indices.size() >= maxLightsPerTile) break;
            tile.indices.push_back(static_cast<std::uint16_t>(i));
            tile.minDepth=std::min(tile.minDepth,std::max(0.0f,l.viewDepth-l.depthRadius));
            tile.maxDepth=std::max(tile.maxDepth,l.viewDepth+l.depthRadius);
        }
        if (tile.indices.empty()) tile.minDepth=tile.maxDepth=0.0f;
    }
    return tiles;
}
}
