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

// -----------------------------------------------------------------------------
// Procedural Fast 3D Hash & Noise Functions (Mobile Optimized)
// -----------------------------------------------------------------------------
float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
            mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
            mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z
    );
}

// Multi-octave FBM for Cloud Volume Shaping
float cloudFBM(vec3 p) {
    float v = 0.0;
    float amp = 0.5;
    mat3 rot = mat3(
        0.00,  0.80,  0.60,
       -0.80,  0.36, -0.48,
       -0.60, -0.48,  0.64
    );
    for (int i = 0; i < 4; ++i) {
        v += amp * noise3D(p);
        p = rot * p * 2.18 + vec3(0.15, 0.32, 0.45);
        amp *= 0.48;
    }
    return v;
}

// -----------------------------------------------------------------------------
// Atmospheric Scattering Phase Functions
// -----------------------------------------------------------------------------
// Henyey-Greenstein Phase Function
float hgPhase(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5));
}

// Dual-Lobe Phase Function: Silver lining forward peak + backward soft glow
float dualLobePhase(float cosTheta) {
    return mix(hgPhase(cosTheta, 0.78), hgPhase(cosTheta, -0.22), 0.35);
}

// Rayleigh Scattering Phase
float rayleighPhase(float cosTheta) {
    return (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta);
}

// -----------------------------------------------------------------------------
// Physical Sky Atmosphere (Rayleigh + Mie + Ozone)
// -----------------------------------------------------------------------------
vec3 computeAtmosphere(vec3 ray, vec3 sun, float cosTheta) {
    float sunAlt = clamp(sun.y, -0.15, 1.0);
    
    // Rayleigh scattering coefficients for Red, Green, Blue wavelengths
    vec3 betaR = vec3(5.8e-3, 13.5e-3, 33.1e-3);
    // Mie scattering coefficient
    vec3 betaM = vec3(4.0e-3);
    
    // Optical path length (airmass)
    float zenith = max(ray.y, 0.02);
    float airmass = 1.0 / (zenith + 0.15 * pow(93.885 - acos(zenith) * (180.0 / PI), -1.253));
    
    // Sun extinction through atmosphere
    float sunAirmass = 1.0 / max(sunAlt + 0.05, 0.05);
    vec3 opticalDepth = (betaR + betaM) * sunAirmass * 8.0;
    vec3 sunTransmittance = exp(-opticalDepth);

    // Day Zenith and Horizon color blending
    vec3 skyRayleigh = betaR * rayleighPhase(cosTheta) * airmass * 6.5;
    vec3 skyMie = betaM * hgPhase(cosTheta, 0.78) * airmass * 2.0 * sunTransmittance;

    // Sunset / Golden Hour redness boost
    vec3 sunsetColor = vec3(1.0, 0.38, 0.12) * pow(clamp(1.0 - max(sun.y, 0.0), 0.0, 1.0), 3.0) * 8.0;
    
    vec3 skyColor = (skyRayleigh + skyMie) * mix(vec3(0.4, 0.7, 1.0), vec3(1.0, 0.8, 0.5), pow(1.0 - zenith, 4.0));
    skyColor += sunsetColor * smoothstep(-0.1, 0.3, sun.y) * exp(-zenith * 4.0);

    // Ground horizon fog blend
    if (ray.y < 0.0) {
        vec3 groundColor = vec3(0.08, 0.07, 0.06);
        skyColor = mix(skyColor, groundColor, clamp(-ray.y * 3.5, 0.0, 1.0));
    }

    // Direct Sun Disc with Corona Flare
    float sunDisc = smoothstep(0.9994, 0.9998, cosTheta);
    vec3 sunCore = vec3(1.0, 0.96, 0.90) * sunTransmittance * 60.0 * sunDisc;
    skyColor += sunCore;

    // Night stars background kapag lumubog na ang araw
    if (sun.y < 0.15) {
        vec3 starCoord = ray * 180.0;
        float starNoise = hash(floor(starCoord));
        float star = step(0.994, starNoise) * pow(fract(starNoise * 43.0), 3.0);
        float nightFactor = clamp((0.15 - sun.y) * 4.0, 0.0, 1.0);
        skyColor += vec3(star * 1.8) * nightFactor * clamp(ray.y, 0.05, 1.0);
    }

    return skyColor;
}

// -----------------------------------------------------------------------------
// Volumetric Cloud Density Function
// -----------------------------------------------------------------------------
float sampleCloudDensity(vec3 p, vec3 weatherOffset) {
    // Physical cloud layer: 1500m to 4200m altitude
    const float cloudBase = 1500.0;
    const float cloudTop  = 4200.0;
    
    if (p.y < cloudBase || p.y > cloudTop) return 0.0;

    // Height gradient (Flat base cumulus shape with billowy tops)
    float heightFraction = (p.y - cloudBase) / (cloudTop - cloudBase);
    float heightGradient = smoothstep(0.0, 0.22, heightFraction) * smoothstep(1.0, 0.72, heightFraction);

    // Coordinate scaling and wind motion
    vec3 samplePos = (p + weatherOffset) * 0.00035;

    float baseDensity = cloudFBM(samplePos);
    
    // Cloud coverage threshold (default ~0.48)
    float coverage = 0.52;
    float density = smoothstep(coverage, 0.95, baseDensity * heightGradient);

    return density;
}

// -----------------------------------------------------------------------------
// Volumetric Cloud Raymarcher
// -----------------------------------------------------------------------------
vec4 raymarchClouds(vec3 rayOrigin, vec3 rayDir, vec3 sunDir, float time, vec3 skyColor) {
    if (rayDir.y <= 0.02) return vec4(0.0); // Horizon cutoff

    const float cloudBase = 1500.0;
    const float cloudTop  = 4200.0;

    // Ray intersection sa curved atmosphere cloud slab
    float tStart = (cloudBase - rayOrigin.y) / rayDir.y;
    float tEnd   = (cloudTop  - rayOrigin.y) / rayDir.y;
    
    if (tStart < 0.0) tStart = 0.0;
    if (tEnd <= tStart) return vec4(0.0);

    // Limit maximum distance sa abot-tanaw
    tStart = max(tStart, 0.0);
    tEnd   = min(tEnd, 35000.0);
    float rayLength = tEnd - tStart;

    // Mobile Raymarching Steps (16 Primary Steps, 4 Shadow Steps)
    const int STEPS = 16;
    float stepSize = rayLength / float(STEPS);

    // Ray Jittering (Bayer / Noise dithering para walang step banding)
    float jitter = hash(rayDir * 123.456 + time) * 0.85;
    vec3 p = rayOrigin + rayDir * (tStart + stepSize * jitter);

    // Animated Wind Displacement
    vec3 windVector = vec3(time * 35.0, 0.0, time * 18.0);

    float cosTheta = dot(rayDir, sunDir);
    float phase = dualLobePhase(cosTheta);

    vec3 sunLightColor = vec3(1.0, 0.94, 0.85) * clamp(sunDir.y * 3.5, 0.1, 4.5);
    vec3 ambientLight = skyColor * 0.45 + vec3(0.08, 0.12, 0.18);

    vec3 cloudEnergy = vec3(0.0);
    float transmittance = 1.0;

    for (int i = 0; i < STEPS; ++i) {
        float density = sampleCloudDensity(p, windVector);

        if (density > 0.001) {
            // Light Marching papunta sa sikat ng araw
            const int LIGHT_STEPS = 4;
            float lightStepSize = 180.0;
            float lightDensity = 0.0;
            vec3 lp = p + sunDir * lightStepSize * 0.5;

            for (int j = 0; j < LIGHT_STEPS; ++j) {
                lightDensity += sampleCloudDensity(lp, windVector);
                lp += sunDir * lightStepSize;
            }

            // Beer-Lambert Law + Sugar Powder Effect (1.0 - exp(-density * 2.0))
            float opticalDepthLight = lightDensity * 0.42;
            float shadowTransmittance = exp(-opticalDepthLight);
            float powder = 1.0 - exp(-density * 2.5);

            vec3 lightScatter = sunLightColor * (shadowTransmittance * powder * phase * 4.0);
            vec3 stepAmbient  = ambientLight * (1.0 - density * 0.5);

            vec3 stepLighting = lightScatter + stepAmbient;

            // Extinction ng ray papunta sa camera
            float sampleExtinction = density * 0.15;
            float sampleTransmittance = exp(-sampleExtinction * stepSize);

            cloudEnergy += transmittance * (stepLighting * (1.0 - sampleTransmittance));
            transmittance *= sampleTransmittance;

            if (transmittance < 0.02) break; // Early Exit Optimization
        }

        p += rayDir * stepSize;
    }

    // Distance haze blend papunta sa horizon
    float horizonFade = smoothstep(0.02, 0.18, rayDir.y);
    float alpha = (1.0 - transmittance) * horizonFade;

    return vec4(cloudEnergy, alpha);
}

// -----------------------------------------------------------------------------
// ACES Filmic Tone Mapping (Academy Standard)
// -----------------------------------------------------------------------------
vec3 toneMapACES(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 ray = normalize(vRayDir);
    vec3 sun = normalize(skyData.sunDir.xyz);
    float time = skyData.cameraPos.w;
    float cosTheta = dot(ray, sun);

    // 1. PHYSICAL ATMOSPHERIC SCATTERING
    vec3 skyColor = computeAtmosphere(ray, sun, cosTheta);

    // 2. VOLUMETRIC CLOUD RAYMARCHING
    vec4 clouds = raymarchClouds(vCamPos, ray, sun, time, skyColor);
    vec3 finalLinear = mix(skyColor, clouds.rgb, clouds.a);

    // 3. EXPOSURE + TONEMAPPING + GAMMA CORRECTION
    float exposure = skyData.sunDir.w; // Galing sa Settings Slider
    vec3 mapped = toneMapACES(finalLinear * exposure);
    vec3 finalColor = pow(max(mapped, vec3(0.0)), vec3(1.0 / 2.2));

    outColor = vec4(finalColor, 1.0);
}
