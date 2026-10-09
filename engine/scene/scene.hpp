#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

namespace neo3d {
using EntityId = std::uint64_t;
struct Transform { std::array<float,3> position{0,0,0}; std::array<float,4> rotation{0,0,0,1}; std::array<float,3> scale{1,1,1}; };
struct PbrMaterial { std::array<float,4> baseColor{1,1,1,1}; float metallic=0.0f; float roughness=0.6f; float normalScale=1.0f; float occlusionStrength=1.0f; std::string baseColorTexture, metallicRoughnessTexture, normalTexture, emissiveTexture; };
struct MeshComponent { std::string assetUri; std::uint32_t firstIndex=0, indexCount=0; PbrMaterial material; };
struct Entity { EntityId id=0; std::string name="Entity"; Transform transform; bool visible=true; MeshComponent mesh; };
class Scene {
public:
 EntityId create(std::string name) { Entity e; e.id=next_++; e.name=std::move(name); entities_.push_back(std::move(e)); return entities_.back().id; }
 bool erase(EntityId id) { for (auto i=entities_.begin(); i!=entities_.end(); ++i) if (i->id==id) { entities_.erase(i); return true; } return false; }
 Entity* find(EntityId id) { for (auto& e:entities_) if (e.id==id) return &e; return nullptr; }
 const std::vector<Entity>& entities() const { return entities_; }
private: EntityId next_=1; std::vector<Entity> entities_;
};
}
