#version 450

layout(push_constant) uniform SkyPush {
    mat4 invViewProj;
    vec4 cameraPos;   // xyz: cameraPos, w: time
    vec4 sunDir;      // xyz: sunDir, w: exposure
    vec4 envParams;   // x: fogDensity, y: timeOfDay, z: cloudCoverage, w: windSpeed
} skyData;

layout(location = 0) out vec3 vRayDir;
layout(location = 1) out vec3 vCamPos;

void main() {
    // Hardware Fullscreen Triangle trick gamit ang gl_VertexIndex (0, 1, 2)
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec4 clip = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 world = skyData.invViewProj * clip;
    
    vCamPos = skyData.cameraPos.xyz;
    // Unprojected ray direction mula camera papuntang Far Plane
    vRayDir = normalize(world.xyz / world.w - skyData.cameraPos.xyz);
    
    // Depth clamp sa Far Clip Plane
    gl_Position = vec4(uv * 2.0 - 1.0, 0.99999, 1.0);
}
