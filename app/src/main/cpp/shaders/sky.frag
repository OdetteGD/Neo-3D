#version 450

layout(location = 0) in vec3 vRayDir;

layout(push_constant) uniform SkyPush {
    mat4 invViewProj;
    vec4 sunDir;      // xyz = sun direction, w = time
} skyData;

layout(location = 0) out vec4 outColor;

const float PI = 3.141592653589793;

// 3D Noise for Spherical Cloud Volumetrics
float hash31(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash31(i + vec3(0,0,0)), hash31(i + vec3(1,0,0)), f.x),
            mix(hash31(i + vec3(0,1,0)), hash31(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash31(i + vec3(0,0,1)), hash31(i + vec3(1,0,1)), f.x),
            mix(hash31(i + vec3(0,1,1)), hash31(i + vec3(1,1,1)), f.x), f.y), f.z
    );
}

float fbmCloud(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; ++i) {
        v += noise3D(p) * a;
        p = p * 2.15 + vec3(1.3, 3.7, 5.1);
        a *= 0.5;
    }
    return v;
}

// Mie Phase function for realistic solar halos (Henyey-Greenstein)
float miePhase(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5));
}

void main() {
    vec3 ray = normalize(vRayDir);
    vec3 sun = normalize(skyData.sunDir.xyz);
    float cosTheta = dot(ray, sun);
    float time = skyData.sunDir.w;

    // 1. PHYSICAL ATMOSPHERIC SCATTERING (Rayleigh & Horizon Blend)
    float h = clamp(ray.y, 0.0, 1.0);
    vec3 zenithColor  = vec3(0.08, 0.28, 0.62);  // Deep Blue Sky
    vec3 horizonColor = vec3(0.68, 0.82, 0.96);  // Atmospheric Horizon Haze
    vec3 groundColor  = vec3(0.12, 0.11, 0.10);  // Dark Ground

    vec3 sky = ray.y >= 0.0 ? mix(horizonColor, zenithColor, pow(h, 0.72))
                            : mix(horizonColor, groundColor, pow(-ray.y, 0.6));

    // 2. DIRECTIONAL SUN DISC & CORONA
    float sunDisc = smoothstep(0.9992, 0.9998, cosTheta);
    float mieHalo = miePhase(cosTheta, 0.82) * 0.12;
    vec3 sunGlowColor = vec3(1.0, 0.92, 0.78);
    sky += sunGlowColor * (sunDisc * 8.0 + mieHalo * 3.5);

    // 3. CURVED SPHERICAL DOME CLOUDS (Unity / UE4 Style)
    if (ray.y > 0.02) {
        // Project onto a curved spherical shell at sky height
        float domeHeight = 1.0 / (ray.y + 0.15);
        vec3 cloudCoords = vec3(ray.xz * domeHeight * 0.65, time * 0.015);

        float density = fbmCloud(cloudCoords * 1.8);
        density = smoothstep(0.46, 0.78, density) * smoothstep(0.02, 0.25, ray.y);

        if (density > 0.001) {
            // Light scattering inside clouds: Sunlit upper edges + Silver lining
            float silverLining = pow(max(cosTheta, 0.0), 6.0) * (1.0 - density);
            vec3 cloudDark   = vec3(0.48, 0.58, 0.72) * 0.8;
            vec3 cloudBright = mix(vec3(0.92, 0.95, 1.0), vec3(1.0, 0.92, 0.82), pow(max(cosTheta, 0.0), 3.0));
            vec3 cloudCol    = mix(cloudDark, cloudBright + vec3(silverLining * 1.5), density);

            // Transmittance blend
            sky = mix(sky, cloudCol, density * 0.92);
        }
    }

    outColor = vec4(sky, 1.0);
}
