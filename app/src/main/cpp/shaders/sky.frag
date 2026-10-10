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

    // Volumetric Stratocumulus Cloud Layer
    if (ray.y > 0.01) {
        float dome = 1.0 / (ray.y + 0.16);
        vec2 wind = vec2(time * 0.008, time * 0.0035);
        vec3 uvCloud = vec3(ray.xz * dome * 0.42 + wind, time * 0.0018);

        // Two altitude layers approximate depth and self-shadowing without a
        // costly 3D texture; deliberately bounded for mobile Mali GPUs.
        float broad = fbmCloud(uvCloud * 1.65);
        float detail = fbmCloud(uvCloud * 3.7 + vec3(8.1, 2.4, 5.7));
        float cloudShape = broad * 0.78 + detail * 0.22;
        float lowerLayer = smoothstep(0.47, 0.76, cloudShape);
        float upperField = fbmCloud(uvCloud * 0.92 + vec3(4.0, 7.0, 2.0));
        float upperLayer = smoothstep(0.58, 0.84, upperField) * 0.36;
        float verticalFade = smoothstep(0.008, 0.20, ray.y);
        float density = clamp(max(lowerLayer, upperLayer) * verticalFade *
                              mix(0.75, 1.25, sky.envParams.z), 0.0, 1.0);

        if (density > 0.001) {
            float phase = 0.35 + 0.65 * pow(max(cosTheta * 0.5 + 0.5, 0.0), 5.0);
            float silverLining = pow(max(cosTheta, 0.0), 5.0) * (1.0 - density);
            float powder = 1.0 - exp(-density * 2.2);
            vec3 cloudShadow = mix(vec3(0.16, 0.20, 0.29), vec3(0.42, 0.23, 0.20), sunsetBlend);
            vec3 cloudLit = mix(vec3(1.0, 0.985, 0.96), vec3(1.0, 0.68, 0.40), sunsetBlend);
            cloudLit *= (0.72 + phase * 0.75 + silverLining * 2.0);
            vec3 cloudColor = mix(cloudShadow, cloudLit, clamp(powder + phase * 0.18, 0.0, 1.0));
            // Approximate volumetric transmittance and soft edge scattering.
            float extinction = 1.0 - exp(-density * 2.4);
            finalSky = mix(finalSky, cloudColor, clamp(extinction * 0.94, 0.0, 0.96));
        }
    }

    float exposure = sky.sunDir.w;
    vec3 mapped = ACESFilm(finalSky * exposure);
    outColor = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
