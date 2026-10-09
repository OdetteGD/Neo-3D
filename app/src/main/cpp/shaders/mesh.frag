#version 450
layout(location=0) in vec3 normalWorld;
layout(location=1) in vec3 positionWorld;
layout(location=0) out vec4 outColor;
const float PI = 3.14159265359;
float distributionGGX(vec3 N, vec3 H, float r) {
    float a=r*r, a2=a*a, nh=max(dot(N,H),0.0), d=nh*nh*(a2-1.0)+1.0;
    return a2/max(PI*d*d,0.00001);
}
float geometrySchlick(float nv,float r) {
    float k=(r+1.0)*(r+1.0)/8.0;
    return nv/max(nv*(1.0-k)+k,0.00001);
}
vec3 fresnelSchlick(float c,vec3 f0) { return f0+(1.0-f0)*pow(clamp(1.0-c,0.0,1.0),5.0); }
void main() {
    vec3 N=normalize(normalWorld), V=normalize(vec3(0.0,0.0,4.0)-positionWorld);
    vec3 L=normalize(vec3(-0.55,0.85,0.65)), H=normalize(V+L);
    vec3 albedo=vec3(0.16,0.58,0.92); float metallic=0.18, roughness=0.32;
    vec3 F0=mix(vec3(0.04),albedo,metallic), F=fresnelSchlick(max(dot(H,V),0.0),F0);
    float NDF=distributionGGX(N,H,roughness);
    float G=geometrySchlick(max(dot(N,V),0.0),roughness)*geometrySchlick(max(dot(N,L),0.0),roughness);
    vec3 specular=(NDF*G*F)/max(4.0*max(dot(N,V),0.0)*max(dot(N,L),0.0),0.001);
    vec3 kD=(vec3(1.0)-F)*(1.0-metallic);
    vec3 color=(kD*albedo/PI+specular)*vec3(3.4,3.1,2.7)*max(dot(N,L),0.0)+albedo*0.055;
    color=vec3(1.0)-exp(-color*1.15);
    color=pow(max(color,vec3(0.0)),vec3(1.0/2.2));
    outColor=vec4(color,1.0);
}