#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include <shaders/common.glsl>
#include <shaders/utils.glsl>

layout(location = 0) rayPayloadInEXT ray_payload_t payload;

hitAttributeEXT vec2 attribs;

void main() {
    // 1. Get Geometry Info
    vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    uint blas_instance_id = gl_InstanceCustomIndexEXT;
    mesh_info_t mesh_info = meshes_info_uniform_buffer.m[blas_instance_id];
    material_t material   = mesh_info.material;

    // 2. Interpolate Vertices
    uint index_offset = gl_PrimitiveID * 3;
    uint indices[3] = {
    mesh_info.indices.i[index_offset+0],
    mesh_info.indices.i[index_offset+1],
    mesh_info.indices.i[index_offset+2]
    };

    vertex_attributes_t v0 = mesh_info.vertices.v[indices[0]];
    vertex_attributes_t v1 = mesh_info.vertices.v[indices[1]];
    vertex_attributes_t v2 = mesh_info.vertices.v[indices[2]];

    // Interpolate using barycentrics
    vec3 pos     = v0.position * barycentrics.x + v1.position * barycentrics.y + v2.position * barycentrics.z;
    vec3 normal  = v0.normal   * barycentrics.x + v1.normal   * barycentrics.y + v2.normal   * barycentrics.z;
    vec4 tangent = v0.tangent  * barycentrics.x + v1.tangent  * barycentrics.y + v2.tangent  * barycentrics.z;
    vec2 uv      = v0.base_color_tex_coord * barycentrics.x + v1.base_color_tex_coord * barycentrics.y + v2.base_color_tex_coord * barycentrics.z;

    // Texture sampling
    vec4 base_color = sample_texture(material.base_color_texture_index, uv, material.base_color_value);
    vec3 emissive_factor_rgb = material.emissive_factor.rgb;
    float emissive_strength  = material.emissive_factor.w;

    vec4 emissive_sample = sample_texture(
    material.emissive_texture_index,
    uv,
    vec4(emissive_factor_rgb, 1.0)
    );

    vec3 final_emission = emissive_sample.rgb * emissive_strength;

    // --- NORMAL MAPPING ---
    // Transform Geometric Normal to World Space
    vec3 world_normal = normalize(vec3(normal * gl_WorldToObjectEXT));
    vec3 final_normal = world_normal;

    // Only apply normal mapping if a tangent actually exists
    if (length(tangent.xyz) > 0.1) {
        // Transform Tangent to World Space
        vec3 world_tangent = normalize(vec3(tangent.xyz * gl_WorldToObjectEXT));

        // Gram-Schmidt Orthogonalization
        world_tangent = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);

        // Calculate Bitangent using the cross product and the tangent.w sign (handedness)
        vec3 world_bitangent = cross(world_normal, world_tangent) * tangent.w;

        // Construct the TBN Matrix
        mat3 TBN = mat3(world_tangent, world_bitangent, world_normal);

        // Sample the Normal Map (defaults to flat normal 0.5, 0.5, 1.0 if null)
        vec3 sampled_normal = sample_texture(material.normal_texture_index, uv, vec4(0.5, 0.5, 1.0, 1.0)).rgb;

        // Unpack from [0, 1] to [-1, 1]
        sampled_normal = sampled_normal * 2.0 - 1.0;

        // Apply TBN matrix to get the final, perturbed world-space normal
        final_normal = normalize(TBN * sampled_normal);
    }

    // payload
    payload.dist = gl_HitTEXT;
    payload.emission = final_emission;
    payload.albedo_packed = packUnorm4x8(vec4(base_color.rgb, 1.0));
    payload.normal_packed = pack_normal(final_normal); // PACK THE NORMAL MAPPED NORMAL
    payload.material_info = packHalf2x16(vec2(material.roughness_factor, material.metallic_factor));
    payload.transmission_ior_packed = packHalf2x16(vec2(material.transmission_factor, material.ior));
}