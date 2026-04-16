#ifndef SHADERS_BRDF_GLSL
#define SHADERS_BRDF_GLSL

#include <shaders/utils.glsl>

vec3 get_random_bounce(vec3 normal, float r1, float r2) {
    float phi = 2.0 * 3.14159 * r1;
    float r = sqrt(r2);
    vec3 u = normalize(cross(abs(normal.x) > 0.1 ? vec3(0, 1, 0) : vec3(1, 0, 0), normal));
    vec3 v = cross(normal, u);
    return normalize(u * cos(phi) * r + v * sin(phi) * r + normal * sqrt(1.0 - r2));
}

vec3 get_ggx_microfacet(vec3 normal, float roughness, float r1, float r2) {
    float a = roughness * roughness;
    float phi = 2.0 * 3.14159265 * r1;
    float cosTheta = sqrt((1.0 - r2) / (1.0 + (a*a - 1.0) * r2));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    vec3 h;
    h.x = cos(phi) * sinTheta;
    h.y = sin(phi) * sinTheta;
    h.z = cosTheta;

    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    return tangent * h.x + bitangent * h.y + normal * h.z;
}

vec3 eval_unshadowed_light(
vec3 hit_pos, vec3 hit_normal, vec3 V_view, vec3 hit_albedo, float roughness, float metallic,
emissive_triangle_t light, vec3 light_pos, vec3 light_normal
) {
    vec3 L = light_pos - hit_pos;
    float dist = max(length(L), 0.0001);
    L /= dist;

    float NdotL = max(dot(hit_normal, L), 0.0);
    float cos_light = max(dot(light_normal, -L), 0.0);
    if (NdotL <= 0.0 || cos_light <= 0.0) return vec3(0.0);

    vec3 H = normalize(V_view + L);
    float NdotH = max(dot(hit_normal, H), 0.0);
    float VdotH = max(dot(V_view, H), 0.0);
    float NdotV = max(dot(hit_normal, V_view), 0.001);

    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    float D = a2 / (3.14159 * denom * denom);

    vec3 F0 = mix(vec3(0.04), hit_albedo, metallic);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    vec3 specular_brdf = (D * F) / max(4.0 * NdotV * NdotL, 0.001);
    vec3 diffuse_brdf = hit_albedo * (1.0 - metallic) / 3.14159;
    float geometry = (NdotL * cos_light) / max(dist * dist, 0.0001);

    return light.emission.rgb * (diffuse_brdf + specular_brdf) * geometry;
}

#endif