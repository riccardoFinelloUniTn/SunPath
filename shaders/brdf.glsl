#ifndef SHADERS_BRDF_GLSL
#define SHADERS_BRDF_GLSL

#include <shaders/utils.glsl>

// Branchless orthonormal basis (Duff et al. 2017, "Building an Orthonormal Basis, Revisited")
void build_onb(vec3 n, out vec3 t, out vec3 b) {
    float sign_n = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign_n + n.z);
    float bb = n.x * n.y * a;
    t = vec3(1.0 + sign_n * n.x * n.x * a, sign_n * bb, -sign_n * n.x);
    b = vec3(bb, sign_n + n.y * n.y * a, -n.y);
}

// Smith height-correlated GGX visibility (Heitz 2014). Already includes 1/(4*NdotV*NdotL).
float smith_v_ggx(float NdotV, float NdotL, float alpha) {
    float a2 = alpha * alpha;
    float ggxV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float ggxL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(ggxV + ggxL, 0.0001);
}

// Smith G1 (masking term only), height-correlated denominator
float smith_g1_ggx(float NdotX, float alpha) {
    float a2 = alpha * alpha;
    float denom = NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX);
    return 2.0 * NdotX / max(denom, 0.0001);
}

vec3 get_random_bounce(vec3 normal, float r1, float r2) {
    float phi = 2.0 * 3.14159 * r1;
    float r = sqrt(r2);
    vec3 u, v;
    build_onb(normal, u, v);
    return normalize(u * cos(phi) * r + v * sin(phi) * r + normal * sqrt(1.0 - r2));
}

// VNDF sampling for GGX (Heitz 2018). Returns world-space half-vector.
vec3 sample_ggx_vndf(vec3 normal, vec3 V_world, float roughness, float r1, float r2) {
    vec3 T, B;
    build_onb(normal, T, B);

    // Transform V to local tangent space
    vec3 Vl = vec3(dot(V_world, T), dot(V_world, B), dot(V_world, normal));

    float a = max(roughness * roughness, 0.001);

    // Stretch view
    vec3 Vh = normalize(vec3(a * Vl.x, a * Vl.y, Vl.z));

    // Orthonormal basis in hemisphere of Vh
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);

    // Disk sample + hemisphere warp
    float rr = sqrt(r1);
    float phi = 2.0 * 3.14159265 * r2;
    float t1 = rr * cos(phi);
    float t2 = rr * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    // Unstretch
    vec3 Hl = normalize(vec3(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));

    // Back to world space
    return T * Hl.x + B * Hl.y + normal * Hl.z;
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

    // Height-correlated Smith visibility (includes 1/(4*NdotV*NdotL))
    float V_term = smith_v_ggx(NdotV, NdotL, a);
    vec3 specular_brdf = D * V_term * F;
    vec3 diffuse_brdf = hit_albedo * (1.0 - metallic) * (vec3(1.0) - F) / 3.14159;
    float geometry = (NdotL * cos_light) / max(dist * dist, 0.0001);

    return light.emission.rgb * (diffuse_brdf + specular_brdf) * geometry;
}

#endif