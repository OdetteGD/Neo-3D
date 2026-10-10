#version 450

layout(location = 0) in vec3 vRayDir;
layout(location = 1) in vec3 vCamPos;

layout(push_constant) uniform SkyPushConstants {
    mat4 invViewProj;
    vec4 cameraPos;   // xyz: cameraPos, w: time
    vec4 sunDir;      // xyz: sunDir, w: exposure
    vec4 envParams;   // x: fogDensity, y: timeOfDay, z: cloudCoverage, w: windSpeed
} sky;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float smoothNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash13(i + vec3(0,0,0)), hash13(i + vec3(1,0,0)), f.x),
            mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), f.x),
            mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), f.x), f.y), f.z
    );
}

float fbmCloud(vec3 p) {
    // Multi-octave cloud field: broad cloud masses plus fine billows.
    float v = 0.0;
    v += 0.5000 * smoothNoise(p); p = p * 2.02 + vec3(1.2, 3.4, 5.6);
    v += 0.2500 * smoothNoise(p); p = p * 2.03 + vec3(2.3, 4.5, 6.7);
    v += 0.1250 * smoothNoise(p); p = p * 2.01 + vec3(3.4, 5.6, 7.8);
    v += 0.0625 * smoothNoise(p); p = p * 2.04 + vec3(4.7, 1.9, 3.1);
    v += 0.03125 * smoothNoise(p);
    return v / 0.96875;
}

vec3 ACESFilm(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 ray = normalize(vRayDir);
    vec3 sun = normalize(sky.sunDir.xyz);
    float time = sky.cameraPos.w;
    float cosTheta = dot(ray, sun);

    float h = clamp(ray.y, 0.0, 1.0);
    float sunElev = sun.y;

    vec3 zenithDay = vec3(0.08, 0.32, 0.72);
    vec3 horizonDay = vec3(0.65, 0.80, 0.96);
    vec3 sunsetColor = vec3(1.0, 0.40, 0.10);
    vec3 nightZenith = vec3(0.004, 0.008, 0.018);

    float dayBlend = smoothstep(-0.08, 0.22, sunElev);
    float sunsetBlend = smoothstep(0.35, 0.0, abs(sunElev - 0.06));

    vec3 skyBase = mix(horizonDay, zenithDay, pow(h, 0.68));
    skyBase = mix(skyBase, sunsetColor, sunsetBlend * (1.0 - h));
    skyBase = mix(nightZenith, skyBase, dayBlend);

    // Sun Disc & Corona
    float sunDisc = smoothstep(0.9994, 0.99985, cosTheta) * 60.0;
    float sunCorona = (1.0 - 0.7 * 0.7) / (4.0 * PI * pow(max(1.0 + 0.49 - 1.4 * cosTheta, 0.01), 1.5)) * 3.5;
    vec3 sunLight = vec3(1.0, 0.95, 0.85) * (sunDisc + sunCorona) * dayBlend;
    vec3 finalSky = skyBase + sunLight;

    if (ray.y < 0.0) {
        finalSky = mix(finalSky, vec3(0.08, 0.07, 0.06), clamp(-ray.y * 3.5, 0.0, 1.0));
    }

    // Bounded front-to-back volumetric cloud ray march. Each sample evaluates
    // a compact multi-octave density field in world space. Eight steps keep the
    // cost predictable on mobile tile-based GPUs such as Mali-G57.
    if (ray.y > 0.015) {
        const int CLOUD_STEPS = 8;
        const float CLOUD_NEAR = 38.0;
        const float CLOUD_FAR = 190.0;
        const float CLOUD_STEP = (CLOUD_FAR - CLOUD_NEAR) / float(CLOUD_STEPS);
        vec3 cloudAccum = vec3(0.0);
        float transmittance = 1.0;
        vec2 wind = vec2(time * sky.envParams.w * 0.018,
                         time * sky.envParams.w * 0.007);
        float cloudCoverage = clamp(sky.envParams.z, 0.05, 0.98);

        for (int i = 0; i < CLOUD_STEPS; ++i) {
            float distanceAlongRay = CLOUD_NEAR + (float(i) + 0.5) * CLOUD_STEP;
            vec3 samplePos = sky.cameraPos.xyz + ray * distanceAlongRay;
            // Two cloud decks with domain-warped detail create visible depth.
            vec3 p = vec3(samplePos.x * 0.010 + wind.x,
                          samplePos.y * 0.012 + time * 0.001,
                          samplePos.z * 0.010 + wind.y);
            float warp = smoothNoise(p * 1.35 + vec3(2.7, 0.0, 4.1)) - 0.5;
            vec3 warped = p + vec3(warp * 0.65, warp * 0.18, -warp * 0.55);
            float macroShape = fbmCloud(warped * 1.7);
            float fineShape = fbmCloud(warped * 4.1 + vec3(7.3, 1.8, 3.6));
            float field = macroShape * 0.72 + fineShape * 0.28;
            float deck = smoothstep(0.58 - cloudCoverage * 0.20,
                                    0.82 - cloudCoverage * 0.18, field);
            float altitude = smoothstep(22.0, 55.0, samplePos.y) *
                             (1.0 - smoothstep(165.0, 220.0, samplePos.y));
            float density = clamp(deck * altitude * 1.65, 0.0, 1.0);

            if (density > 0.002) {
                float stepOpacity = 1.0 - exp(-density * CLOUD_STEP * 0.018);
                float phaseForward = pow(max(dot(ray, sun), 0.0) * 0.5 + 0.5, 5.0);
                float phaseBack = pow(max(dot(ray, sun), 0.0) * 0.5 + 0.5, 1.5);
                float powder = clamp(0.30 + phaseForward * 0.85 + phaseBack * 0.20, 0.0, 1.0);
                vec3 cloudShadow = mix(vec3(0.12, 0.16, 0.24),
                                       vec3(0.34, 0.19, 0.17), sunsetBlend);
                vec3 cloudLit = mix(vec3(1.0, 0.985, 0.95),
                                    vec3(1.0, 0.65, 0.36), sunsetBlend);
                // Approximate self-shadowing through accumulated optical depth.
                float selfShadow = exp(-float(i) * 0.075 * density);
                vec3 lighting = mix(cloudShadow, cloudLit,
                                    clamp(powder * selfShadow, 0.0, 1.0));
                cloudAccum += transmittance * stepOpacity * lighting;
                transmittance *= (1.0 - stepOpacity);
                if (transmittance < 0.035) break;
            }
        }
        finalSky = cloudAccum + finalSky * transmittance;
    }

    float exposure = sky.sunDir.w;
    vec3 mapped = ACESFilm(finalSky * exposure);
    outColor = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
