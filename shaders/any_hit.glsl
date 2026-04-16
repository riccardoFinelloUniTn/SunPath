#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include <shaders/common.glsl>
#include <shaders/utils.glsl>

hitAttributeEXT vec2 attribs;

void main() {
    uint blas_instance_id = gl_InstanceCustomIndexEXT;
    mesh_info_t mesh_info = meshes_info_uniform_buffer.m[blas_instance_id];
    material_t material   = mesh_info.material;

    // glTF Alpha Modes: 0 = OPAQUE, 1 = MASK, 2 = BLEND
    // If it's fully opaque, we don't need to do any texture sampling
    if (material.alpha_mode == 0) {
        return;
    }

    // Interpolate UVs
    vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    uint index_offset = gl_PrimitiveID * 3;
    uint indices[3] = {
    mesh_info.indices.i[index_offset+0],
    mesh_info.indices.i[index_offset+1],
    mesh_info.indices.i[index_offset+2]
    };

    vec2 uv0 = mesh_info.vertices.v[indices[0]].base_color_tex_coord;
    vec2 uv1 = mesh_info.vertices.v[indices[1]].base_color_tex_coord;
    vec2 uv2 = mesh_info.vertices.v[indices[2]].base_color_tex_coord;
    vec2 uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

    // Sample the base color texture
    vec4 base_color = sample_texture(material.base_color_texture_index, uv, material.base_color_value);

    if (base_color.a < material.alpha_cutoff) {
        ignoreIntersectionEXT;
    }
}