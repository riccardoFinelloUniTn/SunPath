#ifndef SHADERS_UTILS_GLSL
#define SHADERS_UTILS_GLSL

#include <shaders/common.glsl>

// --- TEXTURE & COLOR UTILS ---
const uint null_texture = ~0;

vec4 sample_texture(in uint texture_index, in vec2 tex_coords, in vec4 fallback_color) {
    if(texture_index == null_texture) {
        return fallback_color;
    } else {
        return texture(texture_samplers[texture_index], tex_coords);
    }
}

float remove_srgb_curve(float x) {
    return x < 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

// --- NORMAL PACKING ---
vec2 signNotZero(vec2 v) {
    return vec2((v.x >= 0.0) ? 1.0 : -1.0, (v.y >= 0.0) ? 1.0 : -1.0);
}

uint pack_normal(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 p = (n.z >= 0.0) ? n.xy : (1.0 - abs(n.yx)) * signNotZero(n.xy);
    return packSnorm2x16(p);
}

vec3 unpack_normal(uint p) {
    vec2 v = unpackSnorm2x16(p);
    vec3 n = vec3(v.x, v.y, 1.0 - abs(v.x) - abs(v.y));
    float t = max(-n.z, 0.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

// --- INTERPOLATION ---
#define INTERPOLATE_VERTEX_ATTRIBUTE(attribute, triangle, barycentrics) \
          triangle[0].attribute * barycentrics.x \
        + triangle[1].attribute * barycentrics.y \
        + triangle[2].attribute * barycentrics.z

vertex_attributes_t interpolate_vertex_attributes(in vertex_attributes_t triangle[3], in vec3 barycentrics) {
    vertex_attributes_t ret;
    ret.position = INTERPOLATE_VERTEX_ATTRIBUTE(position, triangle, barycentrics);
    ret.normal = INTERPOLATE_VERTEX_ATTRIBUTE(normal, triangle, barycentrics);
    ret.tangent = INTERPOLATE_VERTEX_ATTRIBUTE(tangent, triangle, barycentrics);
    ret.base_color_tex_coord = INTERPOLATE_VERTEX_ATTRIBUTE(base_color_tex_coord, triangle, barycentrics);
    ret.metallic_roughness_tex_coord = INTERPOLATE_VERTEX_ATTRIBUTE(metallic_roughness_tex_coord, triangle, barycentrics);
    ret.normal_tex_coord = INTERPOLATE_VERTEX_ATTRIBUTE(normal_tex_coord, triangle, barycentrics);
    ret.occlusion_tex_coord = INTERPOLATE_VERTEX_ATTRIBUTE(occlusion_tex_coord, triangle, barycentrics);
    ret.emissive_tex_coord = INTERPOLATE_VERTEX_ATTRIBUTE(emissive_tex_coord, triangle, barycentrics);
    return ret;
}
#undef INTERPOLATE_VERTEX_ATTRIBUTE

// --- RNG STATE & FUNCTIONS ---
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint seed;

float rnd() {
    seed = seed * 747796405u + 2891336453u;
    uint word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    uint result = (word >> 22u) ^ word;
    return float(result) / 4294967295.0;
}

void init_rng(vec2 pixel, uint frame, vec2 launch_size) {
    uint pixel_idx = uint(pixel.y) * uint(launch_size.x) + uint(pixel.x);
    seed = hash(pixel_idx ^ hash(frame));
}

#endif