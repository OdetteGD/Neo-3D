#version 450

layout(location = 0) in vec3 vRayDir;

layout(push_constant) uniform SkyPush {
    mat4 invViewProj;
    vec4 sunDir; // xyz = sun dir, w = intensity
} skyData;

layout(location = 0) out vec4 outColor;

float hash21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float noise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f); return mix(mix(hash21(i), hash21(i + vec2(1,0)), f.x), mix(hash21(i + vec2(0,1)), hash21(i + vec2(1,1)), f.x), f.y); }
float fbm(vec2 p) { float v = 0.0, a = 0.5; for (int i = 0; i < 4; ++i) { v += noise(p) * a; p = p * 2.05 + vec2(13.1, 7.2); a *= 0.5; } return v; }

void main() {
    vec3 ray = normalize(vRayDir);
    vec3 sun = normalize(skyData.sunDir.xyz);

    float h = clamp(ray.y, 0.0, 1.0);
    vec3 horizon = vec3(0.68, 0.81, 0.94);
    vec3 zenith  = vec3(0.04, 0.18, 0.46);
    vec3 ground  = vec3(0.12, 0.11, 0.10);

    vec3 sky = ray.y >= 0.0 ? mix(horizon, zenith, pow(h, 0.75)) : mix(horizon, ground, pow(-ray.y, 0.5));

    // Directional Sun Disc & Solar Glare in 3D Space
    float sunDot = max(dot(ray, sun), 0.0);
    float disc = smoothstep(0.9985, 0.9995, sunDot);
    float corona = pow(sunDot, 64.0) * 0.8;
    float glow = pow(sunDot, 8.0) * 0.35;
    sky += vec3(1.0, 0.88, 0.7) * (disc * 3.5 + corona + glow);

    // Procedural 3D Cloud Layer
    if (ray.y > 0.05) {
        vec2 plane = ray.xz / (ray.y + 0.12);
        float cloudPattern = fbm(plane * 0.85);
        float clouds = smoothstep(0.48, 0.75, cloudPattern) * smoothstep(0.05, 0.35, ray.y);
        vec3 cloudColor = mix(vec3(0.85, 0.9, 0.95), vec3(1.0, 0.95, 0.85), pow(sunDot, 4.0));
        sky = mix(sky, cloudColor, clouds * 0.85);
    }

    outColor = vec4(sky, 1.0);
}
