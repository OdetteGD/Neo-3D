#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 baseColor;
    vec4 material;     // x: metallic, y: roughness, z: ao, w: unused
    vec4 cameraPos;    // xyz: camPos, w: time
    vec4 sunDir;       // xyz: sunDir, w: exposure
    vec4 envParams;    // x: fogDensity, y: timeOfDay, zw: unused
} ubo;

layout(location = 0) out vec3 vNormalWorld;
layout(location = 1) out vec3 vPositionWorld;
layout(location = 2) out vec4 vColor;

void main() {
    vec4 worldPos = ubo.model * vec4(inPos, 1.0);
    vPositionWorld = worldPos.xyz;
    vNormalWorld = normalize(mat3(ubo.model) * inNormal);
    vColor = ubo.baseColor;
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
}
