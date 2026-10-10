#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

// Descriptor Set 0 - Global Scene Uniforms mula sa native_engine.cpp
layout(std140, set = 0, binding = 0) uniform SceneUniformBuffer {
    mat4 viewProj;
    mat4 invViewProj;
    mat4 shadowViewProj;

    vec4 cameraPosition;      // xyz: eye pos, w: time
    vec4 sunDirection;       // xyz: sun dir, w: sun intensity
    vec4 sunColor;           // rgb: light color, w: ambient intensity

    vec4 fogAtmosphereParams;// x: density, y: timeOfDay, z: Rayleigh, w: Mie
    vec4 cloudParameters;    // x: coverage, y: speed, z: altitude, w: density
    vec4 renderingSettings;  // x: exposure, y: shadowBias, z: pcfRadius, w: unused
} scene;

// Push Constants - 112 bytes exact match sa ObjectPushConstants
layout(push_constant) uniform ObjectPushConstants {
    mat4 model;
    vec4 baseColor;
    vec4 pbrParams;      // x: metallic, y: roughness, z: ao, w: isWater
    vec4 emissive;
} obj;

layout(location = 0) out vec3 vPositionWorld;
layout(location = 1) out vec3 vNormalWorld;
layout(location = 2) out vec2 vUV;
layout(location = 3) out vec4 vShadowCoord;
layout(location = 4) out vec3 vTangentWorld;
layout(location = 5) out vec3 vBitangentWorld;

void main() {
    vec4 worldPos = obj.model * vec4(inPosition, 1.0);
    vPositionWorld = worldPos.xyz;
    vUV = inUV;

    // Normal at Tangent transformations
    mat3 normalMat = transpose(inverse(mat3(obj.model)));
    vec3 N = normalize(normalMat * inNormal);
    vec3 T = normalize(normalMat * inTangent.xyz);
    // Gram-Schmidt orthogonalization
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T) * inTangent.w;

    vNormalWorld = N;
    vTangentWorld = T;
    vBitangentWorld = B;

    // Light Space Shadow Coordinates (Naka-map sa Vulkan [0, 1] texture coordinates)
    vec4 sc = scene.shadowViewProj * worldPos;
    vShadowCoord = vec4(sc.xy * 0.5 + 0.5 * sc.w, sc.z, sc.w);

    gl_Position = scene.viewProj * worldPos;
}
