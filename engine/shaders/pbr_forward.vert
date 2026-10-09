#version 450
layout(location=0) in vec3 inPosition;
layout(location=1) in vec3 inNormal;
layout(location=2) in vec2 inUv;
layout(set=0,binding=0) uniform Camera { mat4 viewProjection; } camera;
layout(push_constant) uniform Object { mat4 model; } objectData;
layout(location=0) out vec3 worldNormal;
layout(location=1) out vec2 uv;
void main() {
    vec4 world = objectData.model * vec4(inPosition, 1.0);
    gl_Position = camera.viewProjection * world;
    worldNormal = normalize(mat3(objectData.model) * inNormal);
    uv = inUv;
}
