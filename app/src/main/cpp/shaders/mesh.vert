#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform DrawConstants {
    mat4 mvp;
    mat4 model;
    vec4 baseColor;
    vec4 material;   // x = metallic, y = roughness, z = ao, w = unused
    vec4 cameraPos;  // xyz = camera world pos, w = time
    vec4 sunDir;     // xyz = sun direction, w = sun intensity
} drawData;

layout(location = 0) out vec3 vNormalWorld;
layout(location = 1) out vec3 vPositionWorld;
layout(location = 2) out vec3 vCameraPos;
layout(location = 3) out vec3 vSunDir;
layout(location = 4) out float vSunIntensity;

void main() {
    vec4 worldPos = drawData.model * vec4(inPosition, 1.0);
    vPositionWorld = worldPos.xyz;

    // Normal transform using normal matrix (safe for non-uniform scaling)
    mat3 normalMat = transpose(inverse(mat3(drawData.model)));
    vNormalWorld = normalize(normalMat * inNormal);

    vCameraPos = drawData.cameraPos.xyz;
    vSunDir = normalize(drawData.sunDir.xyz);
    vSunIntensity = drawData.sunDir.w;

    gl_Position = drawData.mvp * vec4(inPosition, 1.0);
}
