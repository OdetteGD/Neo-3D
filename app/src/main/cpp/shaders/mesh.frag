#version 450

layout(location = 0) in vec3 vNormalWorld;
layout(location = 1) in vec3 vPositionWorld;
layout(location = 2) in vec4 vColor;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 baseColor;
    vec4 material;     // x: metallic, y: roughness, z: ao, w: isWater
    vec4 cameraPos;    // xyz: camPos, w: time
    vec4 sunDir;       // xyz: sunDir, w: exposure
    vec4 envParams;    // x: fogDensity, y: timeOfDay, zw: unused
} ubo;

layout(location = 0) out vec4 outColor;

const float PI = 3.141592653589793;

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / max(PI * denom * denom, 1e-5);
}

float geometrySmithJoint(float NdotV, float NdotL, float roughness) {
    float a = roughness * roughness;
    float gV = NdotL * sqrt(NdotV * (NdotV - a * NdotV) + a);
    float gL = NdotV * sqrt(NdotL * (NdotL - a * NdotL) + a);
    return 0.5 / max(gV + gL, 1e-5);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 ACESFilm(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 N = normalize(vNormalWorld);
    vec3 V = normalize(ubo.cameraPos.xyz - vPositionWorld);
    vec3 L = normalize(ubo.sunDir.xyz);
    float time = ubo.cameraPos.w;

    // REALISTIC ANIMATED PBR WATER PLANE
    bool isWater = ubo.material.w > 0.5;
    if (isWater) {
        // Gerstner Animated Wave Perturbation
        float wave1 = sin(vPositionWorld.x * 1.5 + time * 2.5);
        float wave2 = cos(vPositionWorld.z * 1.2 + time * 1.8);
        vec3 waveNormal = normalize(vec3(wave1 * 0.08, 1.0, wave2 * 0.08));
        N = normalize(mat3(ubo.model) * waveNormal);
    }

    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 1e-4);
    float NdotL = max(dot(N, L), 0.0);

    vec3 albedo = pow(vColor.rgb, vec3(2.2));
    float metallic = isWater ? 0.05 : clamp(ubo.material.x, 0.0, 1.0);
    float roughness = isWater ? 0.03 : clamp(ubo.material.y, 0.04, 1.0);
    float ao = clamp(ubo.material.z, 0.1, 1.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    if (isWater) F0 = vec3(0.02); // Pure water dielectric reflectivity

    // Direct Lighting
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmithJoint(NdotV, NdotL, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
    vec3 diffuse = (kD * albedo) / PI;
    vec3 specular = (D * G * F);

    float sunElevation = clamp(L.y, 0.0, 1.0);
    vec3 sunColor = vec3(1.0, 0.94, 0.85) * (4.5 * sunElevation);
    vec3 direct = (diffuse + specular) * sunColor * NdotL;

    // Sky Reflections
    vec3 R = reflect(-V, N);
    float refUp = clamp(R.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 skyReflect = mix(vec3(0.1, 0.15, 0.25), vec3(0.4, 0.65, 0.95), refUp);
    vec3 ambient = (albedo * 0.15 + skyReflect * F) * ao;

    // Water Tropical Deep Blend
    if (isWater) {
        vec3 waterDeep = vec3(0.02, 0.12, 0.28);
        vec3 waterShallow = vec3(0.05, 0.38, 0.55);
        albedo = mix(waterDeep, waterShallow, clamp(NdotV, 0.0, 1.0));
    }

    vec3 finalLinear = direct + ambient;

    // Height Volumetric Fog
    float dist = length(ubo.cameraPos.xyz - vPositionWorld);
    float fogDensity = ubo.envParams.x;
    float fog = 1.0 - exp(-dist * fogDensity * exp(-vPositionWorld.y * 0.15));
    vec3 fogColor = vec3(0.65, 0.78, 0.92);
    finalLinear = mix(finalLinear, fogColor, clamp(fog, 0.0, 1.0));

    // Tonemap
    vec3 mapped = ACESFilm(finalLinear * ubo.sunDir.w);
    outColor = vec4(pow(mapped, vec3(1.0 / 2.2)), isWater ? 0.88 : vColor.a);
}
