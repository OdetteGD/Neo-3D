#version 450
layout(location=0) in vec3 worldNormal;
layout(location=1) in vec2 uv;
layout(location=0) out vec4 outColor;
layout(set=1,binding=0) uniform Material { vec4 baseColor; float metallic; float roughness; float exposure; float padding; } material;
const float PI = 3.14159265359;
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness; float a2 = a * a; float nDotH = max(dot(N,H),0.0);
    float d = nDotH*nDotH*(a2-1.0)+1.0; return a2 / max(PI*d*d, 0.0001);
}
float geometrySchlickGGX(float nDotV, float roughness) { float r=roughness+1.0; float k=(r*r)/8.0; return nDotV / max(nDotV*(1.0-k)+k,0.0001); }
vec3 fresnelSchlick(float cosTheta, vec3 F0) { return F0 + (1.0-F0)*pow(clamp(1.0-cosTheta,0.0,1.0),5.0); }
void main() {
    vec3 albedo = material.baseColor.rgb; float metal = clamp(material.metallic,0.0,1.0); float rough = clamp(material.roughness,0.04,1.0);
    vec3 N=normalize(worldNormal); vec3 V=normalize(vec3(0.2,0.4,1.0)); vec3 L=normalize(vec3(-0.4,0.8,0.5)); vec3 H=normalize(V+L);
    vec3 F0=mix(vec3(0.04),albedo,metal); vec3 F=fresnelSchlick(max(dot(H,V),0.0),F0);
    float NDF=distributionGGX(N,H,rough); float G=geometrySchlickGGX(max(dot(N,V),0.0),rough)*geometrySchlickGGX(max(dot(N,L),0.0),rough);
    vec3 specular=(NDF*G*F)/max(4.0*max(dot(N,V),0.0)*max(dot(N,L),0.0),0.001);
    vec3 kD=(vec3(1.0)-F)*(1.0-metal); float nDotL=max(dot(N,L),0.0);
    vec3 radiance=vec3(3.0,2.8,2.5); vec3 color=(kD*albedo/PI+specular)*radiance*nDotL+albedo*0.03;
    color=vec3(1.0)-exp(-color*max(material.exposure,0.001)); color=pow(color,vec3(1.0/2.2)); outColor=vec4(color,material.baseColor.a);
}
