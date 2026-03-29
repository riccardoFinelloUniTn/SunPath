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

    // 2. Get Vertices
    uint index_offset = gl_PrimitiveID * 3;
    uint indices[3] = {
    mesh_info.indices.i[index_offset+0],
    mesh_info.indices.i[index_offset+1],
    mesh_info.indices.i[index_offset+2]
    };

    vertex_attributes_t v0 = mesh_info.vertices.v[indices[0]];
    vertex_attributes_t v1 = mesh_info.vertices.v[indices[1]];
    vertex_attributes_t v2 = mesh_info.vertices.v[indices[2]];

    // 3. Interpolate Geometry
    vec3 pos = v0.position * barycentrics.x + v1.position * barycentrics.y + v2.position * barycentrics.z;
    vec3 normal = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;

    // Only interpolate the XYZ direction of the tangent
    vec3 tangent_dir = v0.tangent.xyz * barycentrics.x + v1.tangent.xyz * barycentrics.y + v2.tangent.xyz * barycentrics.z;

    // Handedness must be taken strictly from the first vertex to prevent seam corruption
    float handedness = v0.tangent.w >= 0.0 ? 1.0 : -1.0;

    vec2 uv = v0.base_color_tex_coord * barycentrics.x + v1.base_color_tex_coord * barycentrics.y + v2.base_color_tex_coord * barycentrics.z;
    vec2 normal_uv = v0.normal_tex_coord * barycentrics.x + v1.normal_tex_coord * barycentrics.y + v2.normal_tex_coord * barycentrics.z;

    // 4. Texture sampling
    vec4 base_color = sample_texture(material.base_color_texture_index, uv, material.base_color_value);
    vec3 emissive_factor_rgb = material.emissive_factor.rgb;
    float emissive_strength  = material.emissive_factor.w;

    vec4 emissive_sample = sample_texture(material.emissive_texture_index, uv, vec4(emissive_factor_rgb, 1.0));
    vec3 final_emission = emissive_sample.rgb * emissive_strength;

    // 5. Transform geometric normal correctly (Normal * WorldToObject = proper Inverse Transpose)
    vec3 world_normal = normalize(normal * mat3(gl_WorldToObjectEXT));
    vec3 final_normal = world_normal;

    // By default, our output albedo is just the base color
    vec3 out_albedo = base_color.rgb;

    // 6. Normal Mapping Application
    if (length(tangent_dir) > 0.001) {
        // Build perfect TBN matrix
        vec3 world_tangent = normalize(mat3(gl_ObjectToWorldEXT) * tangent_dir);
        world_tangent = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
        vec3 world_bitangent = cross(world_normal, world_tangent) * handedness;
        mat3 TBN = mat3(world_tangent, world_bitangent, world_normal);

        // CHECK: Does this object actually have a normal map assigned?
        if (material.normal_texture_index != null_texture) {

            // Fetch the RAW normal map in the [0, 1] range (the purple image)
            vec3 raw_normal_map = sample_texture(material.normal_texture_index, normal_uv, vec4(0.5, 0.5, 1.0, 1.0)).rgb;

            // VISUALIZER: Overwrite the albedo so we see the purple normal map instead of the base color
            //out_albedo = raw_normal_map;

            // Do the standard math to actually apply the physical bump to the lighting
            vec3 sampled_normal = raw_normal_map * 2.0 - 1.0;
            float normal_intensity = 1.0;
            sampled_normal.xy *= normal_intensity;
            sampled_normal.z = sqrt(clamp(1.0 - dot(sampled_normal.xy, sampled_normal.xy), 0.0, 1.0));
            sampled_normal = normalize(sampled_normal);

            final_normal = normalize(TBN * sampled_normal);
        }
    }

    // 7. Final Payload Construction
    payload.dist = gl_HitTEXT;
    payload.emission = final_emission;
    payload.albedo_packed = packUnorm4x8(vec4(out_albedo, 1.0)); // Uses our overridden albedo
    payload.normal_packed = pack_normal(final_normal);

    float final_roughness = material.roughness_factor;
    float final_metallic = material.metallic_factor;

    // If the model has a metallic/roughness texture, sample it!
    if (material.metallic_roughness_texture_index != null_texture) {
        // Fallback to 1.0 so we don't zero out the factors if something goes wrong
        vec4 mr_sample = sample_texture(material.metallic_roughness_texture_index, uv, vec4(1.0));

        // glTF standard: Green is Roughness, Blue is Metallic.
        // We multiply by the base factors according to the glTF specification.
        final_roughness *= mr_sample.g;
        final_metallic *= mr_sample.b;
    }

    // Pack the evaluated PBR values into the payload
    payload.material_info = packHalf2x16(vec2(final_roughness, final_metallic));
    payload.transmission_ior_packed = packHalf2x16(vec2(material.transmission_factor, material.ior));
}