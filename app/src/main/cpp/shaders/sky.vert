#version 450

layout(push_constant) uniform SkyPush {
    mat4 invViewProj;
    vec4 sunDir;
} skyData;

layout(location = 0) out vec3 vRayDir;

void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec4 clip = vec4(p * 2.0 - 1.0, 1.0, 1.0);
    vec4 world = skyData.invViewProj * clip;
    vRayDir = world.xyz / world.w;
    gl_Position = vec4(p * 2.0 - 1.0, 0.9999, 1.0);
}
