#version 450
layout(location=0) in vec2 uv;
layout(location=0) out vec4 outColor;
float hash21(vec2 p) { p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
float noise(vec2 p) { vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f); return mix(mix(hash21(i),hash21(i+vec2(1,0)),f.x),mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)),f.x),f.y); }
void main() {
    vec2 p=uv;
    float h=clamp(p.y,0.0,1.0);
    vec3 horizon=vec3(0.66,0.84,0.98), zenith=vec3(0.08,0.38,0.82);
    vec3 sky=mix(horizon,zenith,smoothstep(0.0,0.94,h));
    vec2 sunPos=vec2(0.76,0.78);
    float d=length((p-sunPos)*vec2(1.0,1.45));
    float disc=1.0-smoothstep(0.035,0.045,d);
    float halo=exp(-d*18.0)*0.34;
    sky=mix(sky,vec3(1.0,0.91,0.63),disc);
    sky+=vec3(1.0,0.68,0.28)*halo;
    float cloudNoise=noise(p*vec2(8.0,5.0))+0.45*noise(p*vec2(17.0,11.0));
    float band=smoothstep(0.34,0.78,p.y)*(1.0-smoothstep(0.82,0.98,p.y));
    float clouds=smoothstep(0.83,1.13,cloudNoise)*band;
    sky=mix(sky,vec3(0.96,0.98,1.0),clouds*0.88);
    outColor=vec4(sky,1.0);
}