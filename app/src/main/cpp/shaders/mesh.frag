#version 450

layout(location = 0) in vec3 vNormalWorld;
layout(location = 1) in vec3 vPositionWorld;
layout(location = 2) in vec4 vColor;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 baseColor;
    vec4 material;     // x: metallic, y: roughness, z: ao, w: unused
    vec4 cameraPos;    // xyz: camPos, w: time
    vec4 sunDir;       // xyz: sunDir, w: exposure
    vec4 envParams;    // x: fogDensity, y: timeOfDay, zw: unused
} ubo;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

// GGX / Trowbridge-Reitz Normal Distribution Function
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / max(PI * denom * denom, 1e-5);
}

// Smith Joint Masking-Shadowing Function (Heitz 2014)
float geometrySmithJoint(float NdotV, float NdotL, float roughness) {
    float a = roughness * roughness;
    float gV = NdotL * sqrt(NdotV * (NdotV - a * NdotV) + a);
    float gL = NdotV * sqrt(NdotL * (NdotL - a * NdotL) + a);
    return 0.5 / max(gV + gL, 1e-5);
}

// Fresnel-Schlick with F0
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    float f = clamp(1.0 - cosTheta, 0.0, 1.0);
    float f2 = f * f;
    return F0 + (1.0 - F0) * (f2 * f2 * f);
}

// Roughness-aware Fresnel for ambient reflection
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    float f = clamp(1.0 - cosTheta, 0.0, 1.0);
    float f2 = f * f;
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * (f2 * f2 * f);
}

// ACES Filmic Tone Mapping (Academy Color Encoding System)
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
    vec3 N = normalize(vNormalWorld);
    vec3 V = normalize(ubo.cameraPos.xyz - vPositionWorld);
    float NdotV = max(dot(N, V), 1e-4);

    vec3 albedo = srgbToLinear(clamp(vColor.rgb, 0.0, 1.0));
    float metallic = clamp(ubo.material.x, 0.0, 1.0);
    float roughness = clamp(ubo.material.y, 0.045, 1.0);
    float ao = clamp(ubo.material.z > 0.0 ? ubo.material.z : 1.0, 0.0, 1.0);

    // Dielectrics use 0.04 base reflectivity, metals use albedo
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 Lo = vec3(0.0);

    // 1. DIRECTIONAL SUN LIGHT
    vec3 L = normalize(ubo.sunDir.xyz);
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);

    // Dynamic solar intensity based on angle above horizon
    float sunAltitude = clamp(L.y, 0.0, 1.0);
    float sunIntensity = 4.2 * sunAltitude;

    if (NdotL > 0.0) {
        float D = distributionGGX(N, H, roughness);
        float Vis = geometrySmithJoint(NdotV, NdotL, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 kS = F;
        vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

        vec3 diffuse = kD * albedo / PI;
        vec3 specular = D * Vis * F;

        vec3 sunColor = vec3(1.0, 0.95, 0.88) * sunIntensity;
        Lo += (diffuse + specular) * sunColor * NdotL;
    }

    // 2. SECONDARY FILL LIGHT (Sky bounce)
    vec3 Lfill = normalize(vec3(-L.x, 0.45, -L.z));
    float NdotLfill = max(dot(N, Lfill), 0.0);
    vec3 fillRadiance = vec3(0.35, 0.48, 0.65) * 0.75;
    Lo += (albedo / PI) * (1.0 - metallic) * fillRadiance * NdotLfill;

    // 3. PHYSICALLY-BASED AMBIENT SKY & GROUND BOUNCE
    vec3 skyColor = vec3(0.18, 0.32, 0.55);
    vec3 groundColor = vec3(0.08, 0.07, 0.06);
    float hemisphere = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ambientEnv = mix(groundColor, skyColor, hemisphere);

    vec3 Famb = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kDamb = (vec3(1.0) - Famb) * (1.0 - metallic);
    vec3 diffuseAmbient = kDamb * albedo * ambientEnv;

    vec3 R = reflect(-V, N);
    float reflectionUp = clamp(R.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 reflectionColor = mix(groundColor, skyColor * 1.6, reflectionUp);
    vec3 specularAmbient = Famb * reflectionColor * (1.0 - roughness * 0.5);

    vec3 ambient = (diffuseAmbient + specularAmbient) * ao;
    vec3 finalColor = Lo + ambient;

    // 4. VOLUMETRIC EXPONENTIAL HEIGHT FOG (Connected to Settings Slider)
    float dist = length(ubo.cameraPos.xyz - vPositionWorld);
    float fogDensity = ubo.envParams.x;
    float fogHeightFactor = exp(-vPositionWorld.y * 0.12);
    float fogFactor = 1.0 - exp(-dist * fogDensity * fogHeightFactor);
    vec3 fogColor = vec3(0.65, 0.76, 0.90);
    finalColor = mix(finalColor, fogColor, clamp(fogFactor, 0.0, 1.0));

    // 5. EXPOSURE + ACES TONEMAPPING + GAMMA CORRECTION
    float exposure = ubo.sunDir.w;
    finalColor = toneMapACES(finalColor * exposure);
    finalColor = pow(max(finalColor, vec3(0.0)), vec3(1.0 / 2.2));

    outColor = vec4(finalColor, vColor.a);
}
