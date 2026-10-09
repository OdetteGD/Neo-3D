#version 450

layout(location=0) in vec3 normalWorld;
layout(location=1) in vec3 positionWorld;
layout(location=0) out vec4 outColor;

const float PI = 3.14159265359;

// Cook-Torrance microfacet BRDF: GGX normal distribution, Schlick-GGX
// visibility and Schlick Fresnel. Material maps are not bound yet, so these
// scalar material values are the explicit default material for the viewport.
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
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

vec3 evaluateDirectLight(vec3 N, vec3 V, vec3 L, vec3 radiance,
                         vec3 albedo, float metallic, float roughness) {
    float nl = max(dot(N, L), 0.0);
    if (nl <= 0.0) return vec3(0.0);

    vec3 H = normalize(V + L);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), f0);
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 specular = (D * G * F) /
        max(4.0 * max(dot(N, V), 0.0) * nl, 1e-4);
    vec3 diffuseWeight = (vec3(1.0) - F) * (1.0 - metallic);
    return (diffuseWeight * albedo / PI + specular) * radiance * nl;
}

// Filmic ACES approximation; tone mapping happens once, after light summation.
vec3 toneMapACES(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 N = normalize(normalWorld);
    vec3 V = normalize(vec3(0.0, 0.0, 5.0) - positionWorld);

    // A visible default material while glTF material/texture descriptors are
    // being integrated. Values are linear-space, not display-space colors.
    vec3 albedo = vec3(0.16, 0.58, 0.92);
    float metallic = 0.18;
    float roughness = 0.32;

    vec3 color = vec3(0.0);
    color += evaluateDirectLight(N, V, normalize(vec3(-0.55, 0.85, 0.65)),
                                 vec3(3.4, 3.1, 2.7), albedo, metallic, roughness);
    color += evaluateDirectLight(N, V, normalize(vec3(0.72, 0.25, -0.48)),
                                 vec3(0.65, 0.82, 1.05), albedo, metallic, roughness);
    color += evaluateDirectLight(N, V, normalize(vec3(0.05, -0.72, 0.35)),
                                 vec3(0.16, 0.20, 0.28), albedo, metallic, roughness);

    // Low-cost hemispherical environment approximation, not image-based
    // lighting: sky irradiance above the horizon and ground bounce below it.
    float hemisphere = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 skyIrradiance = vec3(0.24, 0.38, 0.58);
    vec3 groundIrradiance = vec3(0.10, 0.085, 0.07);
    vec3 ambient = mix(groundIrradiance, skyIrradiance, hemisphere);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 Fambient = fresnelSchlick(max(dot(N, V), 0.0), f0);
    vec3 diffuseAmbient = (1.0 - Fambient) * (1.0 - metallic) * albedo;
    vec3 specularAmbient = Fambient * (1.0 - roughness * 0.65);
    color += ambient * (diffuseAmbient * 0.7 + specularAmbient * 0.22);

    color = toneMapACES(color);
    color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}
