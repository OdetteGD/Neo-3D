#version 450

layout(location = 0) in vec3 vPositionWorld;
layout(location = 1) in vec3 vNormalWorld;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec4 vShadowCoord;
layout(location = 4) in vec3 vTangentWorld;
layout(location = 5) in vec3 vBitangentWorld;

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

// Set 0, Binding 1 - Real-time Hardware PCF Shadow Sampler
layout(set = 0, binding = 1) uniform sampler2DShadow shadowMap;

layout(push_constant) uniform ObjectPushConstants {
    mat4 model;
    vec4 baseColor;
    vec4 pbrParams;      // x: metallic, y: roughness, z: ao, w: isWater
    vec4 emissive;
} obj;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

// 16-Tap Percentage Closer Soft Shadows (PCF) Filter
float calculateShadow(vec4 shadowCoord, vec3 N, vec3 L) {
    if (shadowCoord.w <= 0.0) return 1.0;
    vec3 proj = shadowCoord.xyz / shadowCoord.w;
    if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
        return 1.0; // Labas sa shadow map frustum
    }

    // Dynamic slope-scale bias para mawala ang shadow acne
    float bias = max(scene.renderingSettings.y * (1.0 - dot(N, L)), scene.renderingSettings.y * 0.2);
    float currentDepth = proj.z - bias;

    float shadow = 0.0;
    vec2 texelSize = vec2(1.0 / 2048.0) * scene.renderingSettings.z;

    // 4x4 Poisson disk / PCF kernel
    for (int x = -1; x <= 2; ++x) {
        for (int y = -1; y <= 2; ++y) {
            vec2 offset = vec2(float(x) - 0.5, float(y) - 0.5) * texelSize;
            shadow += texture(shadowMap, vec3(proj.xy + offset, currentDepth));
        }
    }
    return shadow / 16.0;
}

// Cook-Torrance Microfacet BRDF
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
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

vec3 ACESFilm(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 N = normalize(vNormalWorld);
    vec3 V = normalize(scene.cameraPosition.xyz - vPositionWorld);
    vec3 L = normalize(scene.sunDirection.xyz);
    float time = scene.cameraPosition.w;

    // Realistic Water Gerstner Waves
    bool isWater = obj.pbrParams.w > 0.5;
    if (isWater) {
        float phaseA = vPositionWorld.x * 0.72 + vPositionWorld.z * 0.48 + time * 1.35;
        float phaseB = vPositionWorld.z * 1.63 - vPositionWorld.x * 0.57 + time * 2.05;
        float phaseC = (vPositionWorld.x + vPositionWorld.z) * 3.1 - time * 2.8;
        float w1 = sin(phaseA) * 0.55 + sin(phaseB) * 0.30 + sin(phaseC) * 0.15;
        float w2 = cos(phaseA * 0.83) * 0.52 + cos(phaseB * 1.12) * 0.33 + cos(phaseC) * 0.15;
        vec3 waveNormal = normalize(vec3(w1 * 0.13, 1.0, w2 * 0.13));
        mat3 TBN = mat3(vTangentWorld, vBitangentWorld, vNormalWorld);
        N = normalize(TBN * waveNormal);
    }

    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 1e-4);
    float NdotL = max(dot(N, L), 0.0);

    vec3 albedo = pow(obj.baseColor.rgb, vec3(2.2));
    float metallic = isWater ? 0.03 : clamp(obj.pbrParams.x, 0.0, 1.0);
    float roughness = isWater ? 0.025 : clamp(obj.pbrParams.y, 0.04, 1.0);
    float ao = clamp(obj.pbrParams.z, 0.05, 1.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    if (isWater) F0 = vec3(0.02);

    // Direct Lighting with Real-Time Soft Shadows
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmithJoint(NdotV, NdotL, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
    vec3 diffuse = (kD * albedo) / PI;
    vec3 specular = (D * G * F);

    float shadowFactor = calculateShadow(vShadowCoord, N, L);
    vec3 directLight = (diffuse + specular) * scene.sunColor.rgb * scene.sunDirection.w * NdotL * shadowFactor;

    // Ambient & Sky Dome Reflections
    vec3 R = reflect(-V, N);
    float up = clamp(R.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 skyZenith = vec3(0.15, 0.35, 0.65);
    vec3 groundBounce = vec3(0.05, 0.04, 0.03);
    vec3 ambientReflect = mix(groundBounce, skyZenith, up);

    vec3 ambient = (albedo * scene.sunColor.w + ambientReflect * F) * ao;

    // Deep water color attenuation
    vec3 finalLinear = directLight + ambient + obj.emissive.rgb;
    if (isWater) {
        vec3 deepWater = vec3(0.008, 0.055, 0.14);
        vec3 shallowWater = vec3(0.035, 0.30, 0.39);
        float shorelineTint = clamp(pow(NdotV, 1.7), 0.0, 1.0);
        vec3 waterBody = mix(deepWater, shallowWater, shorelineTint);
        // Schlick Fresnel: reflections increase at shallow viewing angles.
        float waterFresnel = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);
        vec3 reflectedSky = mix(vec3(0.20, 0.34, 0.48), vec3(0.62, 0.76, 0.92),
                                clamp(R.y * 0.5 + 0.5, 0.0, 1.0));
        float sunGlint = pow(max(dot(reflect(-L, N), V), 0.0), 96.0) *
                         max(scene.sunDirection.y, 0.0) * 2.4;
        finalLinear = mix(waterBody + directLight * 0.20, reflectedSky, waterFresnel);
        finalLinear += scene.sunColor.rgb * sunGlint;
    }

    // Height Volumetric Fog
    float dist = length(scene.cameraPosition.xyz - vPositionWorld);
    float fogDensity = scene.fogAtmosphereParams.x;
    float fog = 1.0 - exp(-dist * fogDensity * exp(-max(vPositionWorld.y, 0.0) * 0.12));
    vec3 fogColor = mix(vec3(0.1, 0.12, 0.18), vec3(0.70, 0.82, 0.95), clamp(scene.sunDirection.y, 0.0, 1.0));
    finalLinear = mix(finalLinear, fogColor, clamp(fog, 0.0, 1.0));

    // Exposure at ACES Tonemapping
    float exposure = scene.renderingSettings.x;
    vec3 mapped = ACESFilm(finalLinear * exposure);
    outColor = vec4(pow(mapped, vec3(1.0 / 2.2)), isWater ? 0.88 : obj.baseColor.a);
}
