#version 450
layout(location=0) in vec2 uv;
layout(location=0) out vec4 outColor;
float hash21(vec2 p) { p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
float noise(vec2 p) { vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f); return mix(mix(hash21(i),hash21(i+vec2(1,0)),f.x),mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)),f.x),f.y); }
// Procedural atmosphere, layered cloud fields and a directional solar disc.
float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; ++i) {
        v += noise(p) * a;
        p = p * 2.03 + vec2(17.1, 9.2);
        a *= 0.5;
    }
    return v;
}
void main() {
    vec2 p = uv;
    float h = clamp(p.y, 0.0, 1.0);
    vec3 horizon = vec3(0.72, 0.83, 0.94);
    vec3 zenith = vec3(0.035, 0.16, 0.42);
    float atmosphere = pow(1.0 - h, 3.0);
    vec3 sky = mix(horizon, zenith, smoothstep(0.0, 0.96, h));
    sky += vec3(0.22, 0.12, 0.055) * atmosphere;

    vec2 sunPos = vec2(0.76, 0.78);
    float d = length((p - sunPos) * vec2(1.0, 1.45));
    float disc = 1.0 - smoothstep(0.018, 0.024, d);
    float corona = exp(-d * 38.0) * 0.7;
    float wideGlow = exp(-d * 7.0) * 0.2;
    sky += vec3(1.0, 0.72, 0.38) * (corona + wideGlow);
    sky = mix(sky, vec3(1.0, 0.96, 0.82), disc);

    // Multi-octave cloud fields with a soft underside and sun-lit upper edges.
    vec2 cloudP = vec2(p.x * 7.5 + 0.035 * p.y, p.y * 4.0);
    float field = fbm(cloudP) * 0.72 + fbm(cloudP * 2.4 + 4.7) * 0.28;
    float cloudBand = smoothstep(0.28, 0.48, h) * (1.0 - smoothstep(0.82, 0.96, h));
    float cloud = smoothstep(0.53, 0.68, field) * cloudBand;
    float cloudDetail = smoothstep(0.57, 0.73, fbm(cloudP * 3.1 + 8.0)) * cloud;
    vec3 cloudShadow = vec3(0.48, 0.57, 0.68);
    vec3 cloudLit = mix(vec3(0.82, 0.87, 0.92), vec3(1.0, 0.94, 0.81), exp(-d * 4.0));
    vec3 cloudColor = mix(cloudShadow, cloudLit, smoothstep(0.43, 0.68, field) + cloudDetail * 0.2);
    sky = mix(sky, cloudColor, clamp(cloud * 0.92, 0.0, 0.92));

    // Gentle horizon haze and stable output for the existing Vulkan sky pipeline.
    sky = mix(sky, vec3(0.76, 0.84, 0.91), atmosphere * 0.22);
    outColor = vec4(max(sky, vec3(0.0)), 1.0);
}
