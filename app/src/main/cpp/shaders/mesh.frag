#version 450

layout(location=0) in vec3 normalWorld;
layout(location=1) in vec3 positionWorld;
layout(push_constant) uniform DrawConstants { mat4 mvp; mat4 model; vec4 baseColor; vec4 material; } drawData;
layout(location=0) out vec4 outColor;

const float PI = 3.14159265359;
const float MIN_ROUGHNESS = 0.045;

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float nh = max(dot(N, H), 0.0);
    float d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-5);
}

float geometrySchlickGGX(float nv, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return nv / max(nv * (1.0 - k) + k, 1e-5);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 fresnelSchlick(float cosine, vec3 f0) {
    float x = clamp(1.0 - cosine, 0.0, 1.0);
    float x2 = x * x;
    return f0 + (1.0 - f0) * (x2 * x2 * x);
}

// Roughness-aware Fresnel for the low-cost environment approximation.
// This is not a replacement for prefiltered cubemap IBL.
vec3 fresnelSchlickRoughness(float cosine, vec3 f0, float roughness) {
    vec3 grazing = max(vec3(1.0 - roughness), f0);
    float x = clamp(1.0 - cosine, 0.0, 1.0);
    float x2 = x * x;
    return f0 + (grazing - f0) * (x2 * x2 * x);
}

vec3 evaluateDirectLight(vec3 N, vec3 V, vec3 L, vec3 radiance,
                         vec3 albedo, float metallic, float roughness) {
    float nl = max(dot(N, L), 0.0);
    float nv = max(dot(N, V), 0.0);
    if (nl <= 0.0 || nv <= 0.0) return vec3(0.0);

    vec3 H = normalize(V + L);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), f0);
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 specular = (D * G * F) / max(4.0 * nv * nl, 1e-4);
    vec3 diffuseWeight = (vec3(1.0) - F) * (1.0 - metallic);
    return (diffuseWeight * albedo / PI + specular) * radiance * nl;
}

vec3 toneMapACES(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

void main() {
    vec3 N = normalize(normalWorld);
    vec3 V = normalize(vec3(0.0, 0.0, 5.0) - positionWorld);

    // glTF baseColorFactor is specified in sRGB space; lighting is linear.
    vec3 albedo = srgbToLinear(clamp(drawData.baseColor.rgb, 0.0, 1.0));
    float metallic = clamp(drawData.material.x, 0.0, 1.0);
    float roughness = clamp(drawData.material.y, MIN_ROUGHNESS, 1.0);

    vec3 color = vec3(0.0);
    color += evaluateDirectLight(N, V, normalize(vec3(-0.55, 0.85, 0.65)),
                                 vec3(3.4, 3.1, 2.7), albedo, metallic, roughness);
    color += evaluateDirectLight(N, V, normalize(vec3(0.72, 0.25, -0.48)),
                                 vec3(0.65, 0.82, 1.05), albedo, metallic, roughness);
    color += evaluateDirectLight(N, V, normalize(vec3(0.05, -0.72, 0.35)),
                                 vec3(0.16, 0.20, 0.28), albedo, metallic, roughness);

    // Mobile-friendly hemispherical environment lighting approximation.
    // Real image-based lighting still requires environment cubemaps and a BRDF LUT.
    float hemisphere = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 skyIrradiance = vec3(0.24, 0.38, 0.58);
    vec3 groundIrradiance = vec3(0.10, 0.085, 0.07);
    vec3 ambient = mix(groundIrradiance, skyIrradiance, hemisphere);

    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    float nv = max(dot(N, V), 0.0);
    vec3 Fambient = fresnelSchlickRoughness(nv, f0, roughness);
    vec3 diffuseAmbient = (vec3(1.0) - Fambient) * (1.0 - metallic) * albedo;
    vec3 specularAmbient = Fambient * (1.0 - roughness * 0.65);
    color += ambient * (diffuseAmbient * 0.7 + specularAmbient * 0.22);

    color = toneMapACES(color);
    color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}
