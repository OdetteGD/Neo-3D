#version 450

layout(location = 0) in vec3 inPosition;

// The shadow pass intentionally has no scene descriptor set. The CPU supplies
// the complete light-view-projection * model matrix for each draw.
layout(push_constant) uniform ShadowPushConstants {
    mat4 lightMvp;
} shadowData;

void main() {
    gl_Position = shadowData.lightMvp * vec4(inPosition, 1.0);
}
