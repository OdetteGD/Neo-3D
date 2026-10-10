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

const float PI = 3.14159265358979323846;

// --- COOK-TORRANCE SPECULAR BRDF ---
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / max(PI * denom * denom, 1e-6);
}

float geometrySmithJoint(float NdotV, float NdotL, float roughness) {
    float a = roughness * roughness;
    float gV = NdotL * sqrt(NdotV * (NdotV - a * NdotV) + a);
    float gL = NdotV * sqrt(NdotL * (NdotL - a * NdotL) + a);
    return 0.5 / max(gV + gL, 1e-6);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Roughness-dependent Fresnel para sa ambient IBL reflections
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ACES Filmic Tone Mapping (UE4 / Unity Industry Curve)
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Multi-octave Gerstner wave normal derivation
vec3 calculateWaterNormal(vec2 pos, float time) {
    float w1 = sin(pos.x * 0.9 + pos.y * 0.6 + time * 2.2);
    float w2 = cos(pos.x * 1.7 - pos.y * 1.3 + time * 2.8);
    float w3 = sin(pos.x * 3.4 + pos.y * 2.8 + time * 4.1);
    
    vec2 dH = vec2(
        0.9 * cos(pos.x * 0.9 + pos.y * 0.6 + time * 2.2) +
        1.7 * -sin(pos.x * 1.7 - pos.y * 1.3 + time * 2.8) +
        3.4 * cos(pos.x * 3.4 + pos.y * 2.8 + time * 4.1),

        0.6 * cos(pos.x * 0.9 + pos.y * 0.6 + time * 2.2) +
        -1.3 * -sin(pos.x * 1.7 - pos.y * 1.3 + time * 2.8) +
        2.8 * cos(pos.x * 3.4 + pos.y * 2.8 + time * 4.1)
    ) * 0.045;

    return normalize(vec3(-dH.x, 1.0, -dH.y));
}

void main() {
    vec3 N = normalize(vNormalWorld);
    vec3 V = normalize(ubo.cameraPos.xyz - vPositionWorld);
    vec3 L = normalize(ubo.sunDir.xyz);
    float time = ubo.cameraPos.w;

    bool isWater = ubo.material.w > 0.5;
    if (isWater) {
        vec3 localWaveNormal = calculateWaterNormal(vPositionWorld.xz, time);
        mat3 normalMatrix = transpose(inverse(mat3(ubo.model)));
        N = normalize(normalMatrix * localWaveNormal);
    }

    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 1e-4);
    float NdotL = max(dot(N, L), 0.0);

    // Linear albedo space
    vec3 albedo = pow(vColor.rgb, vec3(2.2));
    float metallic = isWater ? 0.02 : clamp(ubo.material.x, 0.0, 1.0);
    float roughness = isWater ? 0.025 : clamp(ubo.material.y, 0.035, 1.0);
    float ao = clamp(ubo.material.z, 0.05, 1.0);

    // Dielectric F0 = 0.04 (Water = 0.02), Metal F0 = Albedo
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    if (isWater) F0 = vec3(0.02);

    // --- DIRECT SUN LIGHTING ---
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmithJoint(NdotV, NdotL, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
    vec3 diffuse = (kD * albedo) / PI;
    vec3 specular = (D * G * F);

    float sunElevation = clamp(L.y, 0.0, 1.0);
    vec3 sunLightColor = vec3(1.0, 0.95, 0.88) * (5.5 * sunElevation);
    vec3 directLighting = (diffuse + specular) * sunLightColor * NdotL;

    // --- ANALYTIC IBL & SKY AMBIENT REFLECTION ---
    vec3 R = reflect(-V, N);
    float skyUp = clamp(R.y * 0.5 + 0.5, 0.0, 1.0);
    
    // Dynamic Sky Dome Reflection Colors
    vec3 skyZenithReflect = mix(vec3(0.05, 0.08, 0.15), vec3(0.25, 0.55, 0.95), sunElevation);
    vec3 skyHorizonReflect = mix(vec3(0.08, 0.07, 0.08), vec3(0.85, 0.72, 0.60), sunElevation);
    vec3 groundBounce = vec3(0.06, 0.05, 0.04);
    
    vec3 skyReflectColor = mix(groundBounce, mix(skyHorizonReflect, skyZenithReflect, skyUp), clamp(R.y + 0.2, 0.0, 1.0));
    vec3 ambientF = fresnelSchlickRoughness(NdotV, F0, roughness);
    
    vec3 ambientDiffuse = (albedo * 0.12 * mix(groundBounce, skyZenithReflect, N.y * 0.5 + 0.5)) * (1.0 - metallic);
    vec3 ambientSpecular = skyReflectColor * ambientF * (1.0 / (roughness * roughness + 1.0));
    vec3 ambientLighting = (ambientDiffuse + ambientSpecular) * ao;

    // Water Depth Color Absorption & Transmittance
    if (isWater) {
        vec3 deepWater = vec3(0.01, 0.08, 0.22);
        vec3 shallowWater = vec3(0.04, 0.35, 0.48);
        albedo = mix(deepWater, shallowWater, clamp(pow(NdotV, 2.5), 0.0, 1.0));
        directLighting += albedo * sunLightColor * 0.35;
    }

    vec3 finalLinear = directLighting + ambientLighting;

    // --- EXPONENTIAL ATMOSPHERIC HEIGHT FOG ---
    float dist = length(ubo.cameraPos.xyz - vPositionWorld);
    float fogDensity = ubo.envParams.x;
    float heightFog = exp(-max(vPositionWorld.y, 0.0) * 0.08);
    float fogFactor = 1.0 - exp(-dist * fogDensity * heightFog);
    
    vec3 inScatteringFog = mix(vec3(0.1, 0.12, 0.18), vec3(0.70, 0.82, 0.95), sunElevation);
    inScatteringFog = mix(inScatteringFog, vec3(1.0, 0.55, 0.25), pow(max(dot(-V, L), 0.0), 4.0) * 0.5);
    finalLinear = mix(finalLinear, inScatteringFog, clamp(fogFactor, 0.0, 1.0));

    // --- EXPOSURE & TONEMAPPING ---
    float exposure = ubo.sunDir.w;
    vec3 mapped = ACESFilm(finalLinear * exposure);
    
    // sRGB Gamma Correction
    outColor = vec4(pow(mapped, vec3(1.0 / 2.2)), isWater ? 0.88 : vColor.a);
}
