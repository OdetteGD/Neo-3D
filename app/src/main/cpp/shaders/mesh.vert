#version 450
layout(location=0) in vec3 inPosition;
layout(location=1) in vec3 inNormal;
layout(push_constant) uniform DrawConstants { mat4 mvp; mat4 model; } drawData;
layout(location=0) out vec3 normalWorld;
layout(location=1) out vec3 positionWorld;

void main() {
    vec4 world = drawData.model * vec4(inPosition, 1.0);
    positionWorld = world.xyz;

    // Correctly transform normals when a model contains non-uniform scale.
    // This avoids lighting distortion from using the model matrix directly.
    mat3 normalMatrix = transpose(inverse(mat3(drawData.model)));
    normalWorld = normalize(normalMatrix * inNormal);

    gl_Position = drawData.mvp * vec4(inPosition, 1.0);
}
