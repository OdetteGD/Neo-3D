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

const float PI = 3.141592653589793;

// Mali GPU Safe Analytical 3D Noise (Walang high-frequency black noise)
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float smoothNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // Hermite curve

    return mix(
        mix(mix(hash13(i + vec3(0,0,0)), hash13(i + vec3(1,0,0)), f.x),
            mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), f.x),
            mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), f.x), f.y), f.z
    );
}

float fbmCloud(vec3 p) {
    float v = 0.0;
    v += 0.5000 * smoothNoise(p); p = p * 2.02 + vec3(1.2, 3.4, 5.6);
    v += 0.2500 * smoothNoise(p); p = p * 2.03 + vec3(2.3, 4.5, 6.7);
    v += 0.1250 * smoothNoise(p); p = p * 2.01 + vec3(3.4, 5.6, 7.8);
    v += 0.0625 * smoothNoise(p);
    return v;
}

// Henyey-Greenstein Mie Phase (Solar halo & silver lining)
float hgPhase(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 0.01), 1.5));
}

vec3 ACESFilm(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 ray = normalize(vRayDir);
    vec3 sun = normalize(skyData.sunDir.xyz);
    float time = skyData.cameraPos.w;
    float cosTheta = dot(ray, sun);

    // 1. PHYSICAL ATMOSPHERIC SCATTERING
    float h = clamp(ray.y, 0.0, 1.0);
    float sunAltitude = clamp(sun.y, -0.2, 1.0);

    vec3 zenithDay = vec3(0.08, 0.28, 0.68);
    vec3 horizonDay = vec3(0.68, 0.82, 0.98);
    vec3 sunsetColor = vec3(1.0, 0.42, 0.12);
    vec3 nightZenith = vec3(0.005, 0.01, 0.025);

    // Day-Night-Sunset Blend
    float dayFactor = smoothstep(-0.05, 0.25, sunAltitude);
    float sunsetFactor = smoothstep(0.35, 0.0, abs(sunAltitude - 0.08));

    vec3 skyBase = mix(horizonDay, zenithDay, pow(h, 0.72));
    skyBase = mix(skyBase, sunsetColor, sunsetFactor * (1.0 - h));
    skyBase = mix(nightZenith, skyBase, dayFactor);

    // Direct Sun Disc & Atmospheric Flare
    float sunDisc = smoothstep(0.9993, 0.9998, cosTheta) * 50.0;
    float sunCorona = hgPhase(cosTheta, 0.85) * 3.5;
    vec3 sunLight = vec3(1.0, 0.94, 0.82) * (sunDisc + sunCorona) * dayFactor;
    vec3 sky = skyBase + sunLight;

    // Ground Horizon Haze
    if (ray.y < 0.0) {
        vec3 groundColor = vec3(0.09, 0.08, 0.07);
        sky = mix(sky, groundColor, clamp(-ray.y * 3.0, 0.0, 1.0));
    }

    // 2. ULTRA-SMOOTH PROCEDURAL CLOUD DOME (Walang Itim na Dither Artifacts)
    if (ray.y > 0.01) {
        float domeHeight = 1.0 / (ray.y + 0.18);
        vec2 wind = vec2(time * 0.008, time * 0.004);
        vec3 cloudUV = vec3(ray.xz * domeHeight * 0.45 + wind, time * 0.002);

        float density = fbmCloud(cloudUV * 2.2);
        density = smoothstep(0.42, 0.78, density) * smoothstep(0.01, 0.25, ray.y);

        if (density > 0.001) {
            // Mie forward silver lining
            float silverLining = pow(max(cosTheta, 0.0), 4.0) * (1.0 - density);
            
            // Soft lit cloud coloring (Hindi nagiging itim)
            vec3 cloudAmbient = mix(vec3(0.45, 0.55, 0.70), vec3(0.85, 0.50, 0.35), sunsetFactor);
            vec3 cloudDirect  = mix(vec3(1.0, 0.98, 0.95), vec3(1.0, 0.75, 0.45), sunsetFactor) * (1.5 + silverLining * 2.5);
            vec3 cloudFinal = mix(cloudAmbient, cloudDirect, density);

            sky = mix(sky, cloudFinal, density * 0.94);
        }
    }

    // 3. EXPOSURE + ACES FILMIC TONEMAPPING
    float exposure = skyData.sunDir.w;
    vec3 finalMapped = ACESFilm(sky * exposure);
    outColor = vec4(pow(finalMapped, vec3(1.0 / 2.2)), 1.0);
}
