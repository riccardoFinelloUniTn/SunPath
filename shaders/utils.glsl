#ifndef SHADERS_UTILS_GLSL
#define SHADERS_UTILS_GLSL
// necessary since this file uses the texture_samplers uniform in sample_texture
#include <shaders/common.glsl>

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// a texture index of ~0 == u32(-1) == 0xffffffff may be passed to indicate that no texture should be used, and the provided value should be used as replacement for all texels
const uint null_texture = ~0;

// sample from the provided texture index or, if it is null, return the fallback color
// texture_samplers is not currently passed as a
vec4 sample_texture(in uint texture_index, in vec2 tex_coords, in vec4 fallback_color) {
    if(texture_index == null_texture) {
        return fallback_color;
    } else {
        return texture(texture_samplers[texture_index], tex_coords);
    }
}

// Octahedral Normal Packing (3D -> 2D -> uint)
uint pack_normal(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 v = (n.z >= 0.0) ? n.xy : (1.0 - abs(n.yx)) * sign(n.xy);
    return packSnorm2x16(v * 0.5 + 0.5);
}

vec3 unpack_normal(uint p) {
    vec2 v = unpackSnorm2x16(p) * 2.0 - 1.0;
    vec3 n = vec3(v, 1.0 - abs(v.x) - abs(v.y));
    float t = max(-n.z, 0.0);
    n.xy += mix(vec2(t), vec2(-t), greaterThanEqual(n.xy, vec2(0.0)));
    return normalize(n);
}

//given 3 vertices, a texture coordinate attribute and barycentric coordinates interpolate the texture coordinate attribute
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

// take a value that should be interpreted as linear and return the equivalent that should be interpreted as sRGB.
// this is useful to write to an sRGB image from a compute or raytracing shader.
// source: https://github.com/Microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/ColorSpaceUtility.hlsli
// note: if this is ever a bottleneck (shouldn't be) consider using the fast version, from the same source
float remove_srgb_curve(float x) {
    // Approximately pow(x, 2.2)
    return x < 0.04045 ?  x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

#endif
