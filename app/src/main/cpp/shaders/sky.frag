#version 450

layout(location = 0) in vec3 vRayDir;
layout(location = 1) in vec3 vCamPos;

layout(push_constant) uniform SkyPush {
    mat4 invViewProj;
    vec4 cameraPos;   // xyz: cameraPos, w: time
    vec4 sunDir;      // xyz: sunDir, w: exposure
    vec4 envParams;   // x: fogDensity, y: timeOfDay, z: cloudCoverage, w: windSpeed
} skyData;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

// Mali/Adreno Mobile-Safe Hash
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

// 3D Smooth Interpolated Value Noise
float smoothNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // Hermite cubic curve

    return mix(
        mix(mix(hash13(i + vec3(0,0,0)), hash13(i + vec3(1,0,0)), f.x),
            mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), f.x),
            mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), f.x), f.y), f.z
    );
}

// 4-Octave Fractional Brownian Motion para sa makapal at malambot na ulap
float fbmCloud(vec3 p) {
    float v = 0.0;
    v += 0.5000 * smoothNoise(p); p = p * 2.02 + vec3(1.2, 3.4, 5.6);
    v += 0.2500 * smoothNoise(p); p = p * 2.03 + vec3(2.3, 4.5, 6.7);
    v += 0.1250 * smoothNoise(p); p = p * 2.01 + vec3(3.4, 5.6, 7.8);
    v += 0.0625 * smoothNoise(p);
    return v;
}

// Dual-Lobe Henyey-Greenstein Phase Function (Mabilisang rayleigh forward scattering)
float dualHenyeyGreenstein(float cosTheta) {
    float g1 = 0.82;
    float g2 = -0.35;
    float w = 0.65;
    
    float p1 = (1.0 - g1 * g1) / (4.0 * PI * pow(max(1.0 + g1 * g1 - 2.0 * g1 * cosTheta, 0.001), 1.5));
    float p2 = (1.0 - g2 * g2) / (4.0 * PI * pow(max(1.0 + g2 * g2 - 2.0 * g2 * cosTheta, 0.001), 1.5));
    return mix(p2, p1, w);
}

vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 ray = normalize(vRayDir);
    vec3 sun = normalize(skyData.sunDir.xyz);
    float time = skyData.cameraPos.w;
    float cosTheta = dot(ray, sun);

    // 1. PHYSICAL ATMOSPHERIC SCATTERING SKY MODEL
    float h = clamp(ray.y, 0.0, 1.0);
    float sunElev = sun.y;

    vec3 zenithDay = vec3(0.08, 0.32, 0.72);
    vec3 horizonDay = vec3(0.65, 0.80, 0.96);
    vec3 sunsetColor = vec3(1.0, 0.40, 0.10);
    vec3 nightZenith = vec3(0.004, 0.008, 0.018);
    vec3 nightHorizon = vec3(0.02, 0.03, 0.05);

    float dayBlend = smoothstep(-0.08, 0.22, sunElev);
    float sunsetBlend = smoothstep(0.35, 0.0, abs(sunElev - 0.06));

    vec3 skyBase = mix(horizonDay, zenithDay, pow(h, 0.68));
    skyBase = mix(skyBase, sunsetColor, sunsetBlend * (1.0 - h));
    vec3 nightSky = mix(nightHorizon, nightZenith, pow(h, 0.68));
    skyBase = mix(nightSky, skyBase, dayBlend);

    // Sun Disc at Solar Corona Halo
    float sunDisc = smoothstep(0.9994, 0.99985, cosTheta) * 60.0;
    float sunCorona = dualHenyeyGreenstein(cosTheta) * 4.2;
    vec3 sunLightColor = mix(vec3(1.0, 0.55, 0.2), vec3(1.0, 0.96, 0.85), clamp(sunElev * 2.0, 0.0, 1.0));
    vec3 sunAtmosphere = sunLightColor * (sunDisc + sunCorona) * dayBlend;
    vec3 finalSky = skyBase + sunAtmosphere;

    // Horizon Earth Nadir blend
    if (ray.y < 0.0) {
        vec3 groundTint = vec3(0.08, 0.07, 0.06);
        finalSky = mix(finalSky, groundTint, clamp(-ray.y * 3.5, 0.0, 1.0));
    }

    // 2. ULTRA-REALISTIC PROCEDURAL CLOUDS (Curved Stratocumulus Dome)
    if (ray.y > 0.01) {
        float cloudAltitude = 1.0 / (ray.y + 0.16);
        vec2 windOffset = vec2(time * 0.008, time * 0.0035);
        vec3 cloudCoord = vec3(ray.xz * cloudAltitude * 0.42 + windOffset, time * 0.0018);

        float density = fbmCloud(cloudCoord * 2.4);
        float coverageThreshold = 0.40;
        density = smoothstep(coverageThreshold, 0.82, density) * smoothstep(0.01, 0.28, ray.y);

        if (density > 0.001) {
            // Forward Mie Scattering Silver Lining
            float silverLining = pow(max(cosTheta, 0.0), 3.5) * (1.0 - density);
            
            vec3 cloudShadow = mix(vec3(0.25, 0.32, 0.45), vec3(0.70, 0.35, 0.25), sunsetBlend);
            vec3 cloudLit    = mix(vec3(1.0, 0.98, 0.95), vec3(1.0, 0.72, 0.38), sunsetBlend) * (1.6 + silverLining * 2.8);
            
            vec3 cloudColor = mix(cloudShadow, cloudLit, density);
            finalSky = mix(finalSky, cloudColor, density * 0.95);
        }
    }

    // 3. EXPOSURE & ACES TONEMAPPING
    float exposure = skyData.sunDir.w;
    vec3 mapped = ACESFilm(finalSky * exposure);
    outColor = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
