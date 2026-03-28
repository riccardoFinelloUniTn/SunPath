#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include <shaders/brdf.glsl>
#include <shaders/restir.glsl>

layout(set = 0, binding = 1, r11f_g11f_b10f) uniform image2D raw_color_image;
layout(set = 0, binding = 5, r16f) uniform image2D depth_image;
layout(set = 0, binding = 6, rgba8_snorm) uniform image2D normal_image;
layout(set = 0, binding = 7, r11f_g11f_b10f) uniform image2D diffuse_image;
layout(set = 0, binding = 8, rg16f) uniform image2D motion_vector_image;

layout(location = 0) rayPayloadEXT ray_payload_t prd;

// RIS specific read/write logic
Reservoir read_history_reservoir(ivec2 coord) {
    uint idx = get_pixel_index(coord, gl_LaunchSizeEXT.xy);
    if (frame_count % 2 == 0) return reservoirs_B[idx];
    else return reservoirs_A[idx];
}

void write_current_reservoir(ivec2 coord, Reservoir r) {
    uint idx = get_pixel_index(coord, gl_LaunchSizeEXT.xy);
    if (frame_count % 2 == 0) reservoirs_A[idx] = r;
    else reservoirs_B[idx] = r;
}

void main() {
    init_rng(gl_LaunchIDEXT.xy, frame_count, gl_LaunchSizeEXT.xy);
    ivec2 pixel_coord = ivec2(gl_LaunchIDEXT.xy);

    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = inUV * 2.0 - 1.0;

    vec4 origin    = matrices_uniform_buffer.view_inverse * vec4(0, 0, 0, 1);
    vec4 target    = matrices_uniform_buffer.proj_inverse * vec4(d.x, d.y, 1, 1);
    vec4 direction = matrices_uniform_buffer.view_inverse * vec4(normalize(target.xyz), 0);

    vec3 rayOrigin = origin.xyz;
    vec3 rayDir    = direction.xyz;

    // Phase 1: Shoot primary ray to find surface
    vec3 hitPos;
    vec3 hit_normal;
    vec3 hit_albedo;
    float roughness;
    float metallic;
    vec3 V_view;
    vec2 prev_uv;

    bool found_diffuse_surface = false;
    float virtual_distance = 0.0;

    for (int virtual_bounce = 0; virtual_bounce < 20; virtual_bounce++) {
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, rayOrigin, 0.001, rayDir, 10000.0, 0);

        if (prd.dist < 0.0) {
            break; // Hit the sky
        }

        hitPos = rayOrigin + rayDir * prd.dist;
        hit_normal = unpack_normal(prd.normal_packed);
        hit_albedo = unpackUnorm4x8(prd.albedo_packed).rgb;

        vec2 mat_info = unpackHalf2x16(prd.material_info);
        roughness = max(mat_info.x, 0.01);
        metallic = clamp(mat_info.y, 0.0, 1.0);

        vec2 trans_ior = unpackHalf2x16(prd.transmission_ior_packed);
        float transmission = trans_ior.x;

        V_view = -rayDir;
        virtual_distance += prd.dist;

        // Glass / Refraction Check
        if (transmission > 0.5) {
            float ior = max(trans_ior.y, 1.0);
            bool is_inside = dot(rayDir, hit_normal) > 0.0;
            vec3 N = is_inside ? -hit_normal : hit_normal;
            float eta = is_inside ? (ior / 1.0) : (1.0 / ior);

            float cos_theta = min(dot(-rayDir, N), 1.0);
            float R0 = (1.0 - eta) / (1.0 + eta);
            R0 = R0 * R0;
            float fresnel = R0 + (1.0 - R0) * pow(1.0 - cos_theta, 5.0);

            vec3 refracted = refract(rayDir, N, eta);
            if (length(refracted) < 0.01) fresnel = 1.0;

            if (rnd() < fresnel) {
                rayDir = reflect(rayDir, N);
            } else {
                rayDir = refracted;
            }
            rayOrigin = hitPos + rayDir * 0.001;
        }
        // Mirror Check
        else if (metallic > 0.9 && roughness < 0.1) {
            rayOrigin = hitPos + hit_normal * 0.001;
            rayDir = reflect(rayDir, hit_normal);
        }
        // Diffuse Object Hit
        else {
            vec3 virtual_world_pos = origin.xyz + direction.xyz * virtual_distance;
            vec4 prev_clip = matrices_uniform_buffer.prev_view_proj * vec4(virtual_world_pos, 1.0);
            vec2 prev_ndc = prev_clip.xy / prev_clip.w;

            prev_uv = vec2(prev_ndc.x, prev_ndc.y) * 0.5 + 0.5;
            vec2 motion_vector = inUV - prev_uv;
            vec3 denoiser_albedo = mix(hit_albedo, vec3(1.0), metallic);

            imageStore(depth_image, pixel_coord, vec4(virtual_distance, 0.0, 0.0, 0.0));
            imageStore(normal_image, pixel_coord, vec4(hit_normal, roughness));
            imageStore(diffuse_image, pixel_coord, vec4(denoiser_albedo, 0.0));
            imageStore(motion_vector_image, pixel_coord, vec4(motion_vector, 0.0, 0.0));

            found_diffuse_surface = true;
            break;
        }
    }

    if (!found_diffuse_surface) {
        imageStore(depth_image, pixel_coord, vec4(100000.0, 0.0, 0.0, 0.0));
        imageStore(normal_image, pixel_coord, vec4(0.0));
        imageStore(diffuse_image, pixel_coord, vec4(0.0));
        imageStore(motion_vector_image, pixel_coord, vec4(0.0));
        write_current_reservoir(pixel_coord, Reservoir(0, uint[3](0,0,0), vec3(0), 0.0, vec3(0), 0.0, 0.0, 0.0, uint[2](0,0)));
        return;
    }

    // Phase 2: RIS Initial Audition
    Reservoir current_r = Reservoir(0, uint[3](0,0,0), vec3(0), 0.0, vec3(0), 0.0, 0.0, 0.0, uint[2](0,0));
    uint num_lights = emissive_triangles.length();
    int RIS_CANDIDATES = 8;

    if (num_lights > 0 && roughness > 0.2) {
        for(int i = 0; i < RIS_CANDIDATES; i++) {
            uint cand_idx = min(uint(rnd() * num_lights), num_lights - 1);
            emissive_triangle_t cand_light = emissive_triangles[cand_idx];

            float sqr1 = sqrt(rnd());
            float u = 1.0 - sqr1;
            float v = rnd() * sqr1;
            float w = 1.0 - u - v;

            vec3 cand_pos = cand_light.v0_area.xyz * u + cand_light.v1.xyz * v + cand_light.v2.xyz * w;
            vec3 cand_normal = normalize(cross(cand_light.v1.xyz - cand_light.v0_area.xyz, cand_light.v2.xyz - cand_light.v0_area.xyz));

            vec3 f_y = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, cand_light, cand_pos, cand_normal);
            float p_hat = max(f_y.r, max(f_y.g, f_y.b));
            float cand_area = cand_light.v0_area.w;
            float p_y = 1.0 / max(float(num_lights) * cand_area, 0.0001);

            // Inline update_reservoir
            current_r.w_sum += (p_hat / p_y);
            current_r.M += 1.0;
            if (rnd() < ((p_hat / p_y) / max(current_r.w_sum, 0.0001))) {
                current_r.light_idx = cand_idx;
                current_r.light_pos = cand_pos;
                current_r.light_normal = cand_normal;
            }
        }

        if (current_r.w_sum > 0.0) {
            emissive_triangle_t winner = emissive_triangles[current_r.light_idx];
            vec3 f_y_winner = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, winner, current_r.light_pos, current_r.light_normal);
            float p_hat_winner = max(f_y_winner.r, max(f_y_winner.g, f_y_winner.b));
            current_r.W = current_r.w_sum / max(current_r.M * p_hat_winner, 0.0001);
        }

        // Temporal Reuse
        if (frame_count > 0) {
            ivec2 prev_coord = ivec2(prev_uv * vec2(gl_LaunchSizeEXT.xy));
            if (prev_coord.x >= 0 && prev_coord.y >= 0 && prev_coord.x < gl_LaunchSizeEXT.x && prev_coord.y < gl_LaunchSizeEXT.y) {
                Reservoir history_r = read_history_reservoir(prev_coord);
                history_r.M = min(history_r.M, 10.0);

                if (history_r.W > 0.0) {
                    history_r.light_idx = min(history_r.light_idx, num_lights - 1);
                    emissive_triangle_t hist_light = emissive_triangles[history_r.light_idx];
                    vec3 f_y_hist = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, hist_light, history_r.light_pos, history_r.light_normal);
                    float p_hat_hist = max(f_y_hist.r, max(f_y_hist.g, f_y_hist.b));

                    merge_reservoirs(current_r, history_r, p_hat_hist, rnd());

                    emissive_triangle_t merged_winner = emissive_triangles[current_r.light_idx];
                    vec3 f_y_merged = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, merged_winner, current_r.light_pos, current_r.light_normal);
                    float p_hat_merged = max(f_y_merged.r, max(f_y_merged.g, f_y_merged.b));
                    current_r.W = current_r.w_sum / max(current_r.M * p_hat_merged, 0.0001);
                }
            }
        }
    }

    write_current_reservoir(pixel_coord, current_r);
}