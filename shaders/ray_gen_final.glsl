#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include <shaders/common.glsl>
#include <shaders/utils.glsl>

layout(set = 0, binding = 1, r11f_g11f_b10f) uniform image2D raw_color_image;
layout(set = 0, binding = 5, r16f) uniform image2D depth_image;
layout(set = 0, binding = 6, rgba8_snorm) uniform image2D normal_image;
layout(set = 0, binding = 7, r11f_g11f_b10f) uniform image2D diffuse_image;
layout(set = 0, binding = 10) uniform sampler2D blue_noise_tex;

struct Reservoir {
    uint light_idx;
    uint _pad0[3];
    vec3 light_pos;
    float _pad1;
    vec3 light_normal;
    float w_sum;
    float M;
    float W;
    uint _pad2[2];
};

layout(std430, set = 0, binding = 11) buffer ReservoirBufferA { Reservoir reservoirs_A[]; };
layout(std430, set = 0, binding = 12) buffer ReservoirBufferB { Reservoir reservoirs_B[]; };

layout(location = 0) rayPayloadEXT ray_payload_t prd;

uint seed;
float rnd() {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    seed = (word >> 22u) ^ word;
    return float(seed) / 4294967295.0;
}

void init_rng(vec2 pixel, uint frame) {
    seed = uint(pixel.x) * 1973u + uint(pixel.y) * 9277u + frame * 26699u;
}

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

uint get_pixel_index(ivec2 coord) {
    return coord.y * gl_LaunchSizeEXT.x + coord.x;
}

Reservoir read_current_reservoir(ivec2 coord) {
    uint idx = get_pixel_index(coord);
    if (frame_count % 2 == 0) return reservoirs_A[idx];
    else return reservoirs_B[idx];
}

void merge_reservoirs(inout Reservoir r, Reservoir new_r, float p_hat_new, float random_val) {
    r.M += new_r.M;
    float weight = p_hat_new * new_r.W * new_r.M;
    r.w_sum += weight;
    if (random_val < (weight / max(r.w_sum, 0.0001))) {
        r.light_idx = new_r.light_idx;
        r.light_pos = new_r.light_pos;
        r.light_normal = new_r.light_normal;
    }
}

vec3 eval_unshadowed_light(
vec3 hit_pos, vec3 hit_normal, vec3 V_view, vec3 hit_albedo, float roughness, float metallic,
emissive_triangle_t light, vec3 light_pos, vec3 light_normal
) {
    vec3 L = light_pos - hit_pos;
    float dist = length(L);
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

void main() {
    vec3 total_radiance = vec3(0.0);
    int SAMPLES = 1;
    int BOUNCES = 20;
    int SHADOW_BOUNCES = BOUNCES / 2;

    init_rng(gl_LaunchIDEXT.xy, frame_count);
    ivec2 pixel_coord = ivec2(gl_LaunchIDEXT.xy);

    vec3 primary_albedo = vec3(1.0);
    ivec2 tex_size = textureSize(blue_noise_tex, 0);
    ivec2 pan_offset = ivec2(frame_count * 1619, frame_count * 3137);
    ivec2 noise_coord = ivec2(gl_LaunchIDEXT.xy + pan_offset) % tex_size;
    vec4 blue_noise = texelFetch(blue_noise_tex, noise_coord, 0);

    for(int i = 0; i < SAMPLES; i++){
        const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
        const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
        vec2 d = inUV * 2.0 - 1.0;
        vec4 origin    = matrices_uniform_buffer.view_inverse * vec4(0, 0, 0, 1);
        vec4 target    = matrices_uniform_buffer.proj_inverse * vec4(d.x, d.y, 1, 1);
        vec4 direction = matrices_uniform_buffer.view_inverse * vec4(normalize(target.xyz), 0);
        vec3 rayOrigin = origin.xyz;
        vec3 rayDir    = direction.xyz;

        vec3 throughput = vec3(1.0);
        vec3 radiance   = vec3(0.0);
        bool in_glass = false;

        for (int bounce = 0; bounce < BOUNCES; bounce++) {
            uint ray_flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
            if (in_glass) {
                ray_flags = gl_RayFlagsOpaqueEXT;
            }

            traceRayEXT(tlas, ray_flags, 0xFF, 0, 0, 0, rayOrigin, 0.001, rayDir, 10000.0, 0);
            bool is_sky = (prd.dist < 0.0);

            if (is_sky) {
                radiance += vec3(0.0, 0.0, 0.0) * throughput;
                break;
            }

            vec3 hit_normal = unpack_normal(prd.normal_packed);
            vec3 hit_albedo = unpackUnorm4x8(prd.albedo_packed).rgb;
            vec3 hitPos = rayOrigin + rayDir * prd.dist;
            vec3 V_view = -rayDir;

            vec2 mat_info = unpackHalf2x16(prd.material_info);
            float roughness = max(mat_info.x, 0.01);
            float metallic = clamp(mat_info.y, 0.0, 1.0);

            vec3 denoiser_albedo = mix(hit_albedo, vec3(1.0), metallic);
            vec2 trans_ior = unpackHalf2x16(prd.transmission_ior_packed);
            float transmission = trans_ior.x;
            float ior = max(trans_ior.y, 1.0);

            radiance += prd.emission * throughput;
            float brightness = max(prd.emission.r, max(prd.emission.g, prd.emission.b));
            if (brightness > 1.0) {
                break;
            }

            if (transmission > 0.5) {
                bool is_inside = dot(rayDir, hit_normal) > 0.0;
                vec3 N = is_inside ? -hit_normal : hit_normal;
                float eta = is_inside ? (ior / 1.0) : (1.0 / ior);
                float cos_theta = min(dot(-rayDir, N), 1.0);
                float R0 = (1.0 - eta) / (1.0 + eta);
                R0 = R0 * R0;
                float fresnel = R0 + (1.0 - R0) * pow(1.0 - cos_theta, 5.0);
                vec3 refracted = refract(rayDir, N, eta);

                if (length(refracted) < 0.01) {
                    fresnel = 1.0;
                }

                if (rnd() < fresnel) {
                    rayDir = reflect(rayDir, N);
                } else {
                    rayDir = refracted;
                    in_glass != is_inside;

                    if (is_inside) {
                        vec3 absorption = 1.0 - hit_albedo;
                        throughput *= exp(-absorption * prd.dist * 5.0);
                    } else {
                        throughput *= hit_albedo;
                    }
                }

                rayOrigin = hitPos + rayDir * 0.001;
                continue;
            }

            uint num_lights = emissive_triangles.length();
            if (num_lights > 0 && bounce < SHADOW_BOUNCES) {
                if (bounce == 0) {
                    // --- SPATIAL REUSE ---
                    Reservoir center_r = read_current_reservoir(pixel_coord);
                    Reservoir spatial_r = Reservoir(0, uint[3](0,0,0), vec3(0), 0.0, vec3(0), 0.0, 0.0, 0.0, uint[2](0,0));

                    // 1. Evaluate our own pixel's temporal candidate first
                    if (center_r.W > 0.0) {
                        center_r.light_idx = min(center_r.light_idx, num_lights - 1);
                        emissive_triangle_t center_light = emissive_triangles[center_r.light_idx];
                        vec3 f_y_center = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, center_light, center_r.light_pos, center_r.light_normal);
                        float p_hat_center = max(f_y_center.r, max(f_y_center.g, f_y_center.b));
                        merge_reservoirs(spatial_r, center_r, p_hat_center, rnd());
                    }

                    // 2. Sample 5 neighbors within a 30 pixel radius
                    int SPATIAL_SAMPLES = 5;
                    float SPATIAL_RADIUS = 30.0;

                    for (int s = 0; s < SPATIAL_SAMPLES; s++) {
                        float angle = rnd() * 2.0 * 3.14159;
                        float radius = sqrt(rnd()) * SPATIAL_RADIUS;
                        ivec2 neighbor_coord = pixel_coord + ivec2(cos(angle) * radius, sin(angle) * radius);

                        // Screen bounds check
                        if (neighbor_coord.x < 0 || neighbor_coord.y < 0 || neighbor_coord.x >= gl_LaunchSizeEXT.x || neighbor_coord.y >= gl_LaunchSizeEXT.y) continue;

                        // Geometry check (prevent grabbing light from a different surface)
                        vec3 neighbor_normal = imageLoad(normal_image, neighbor_coord).xyz;
                        float neighbor_depth = imageLoad(depth_image, neighbor_coord).x;

                        if (dot(hit_normal, neighbor_normal) < 0.9) continue;
                        if (abs(prd.dist - neighbor_depth) > 0.1 * prd.dist) continue;

                        Reservoir neighbor_r = read_current_reservoir(neighbor_coord);
                        if (neighbor_r.W > 0.0) {
                            neighbor_r.light_idx = min(neighbor_r.light_idx, num_lights - 1);
                            emissive_triangle_t neighbor_light = emissive_triangles[neighbor_r.light_idx];
                            vec3 f_y_neighbor = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, neighbor_light, neighbor_r.light_pos, neighbor_r.light_normal);
                            float p_hat_neighbor = max(f_y_neighbor.r, max(f_y_neighbor.g, f_y_neighbor.b));

                            merge_reservoirs(spatial_r, neighbor_r, p_hat_neighbor, rnd());
                        }
                    }

                    // 3. Finalize Spatial Winner and Shoot Shadow Ray
                    if (spatial_r.w_sum > 0.0) {
                        emissive_triangle_t winner = emissive_triangles[spatial_r.light_idx];
                        vec3 f_y_winner = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, winner, spatial_r.light_pos, spatial_r.light_normal);
                        float p_hat_winner = max(f_y_winner.r, max(f_y_winner.g, f_y_winner.b));

                        spatial_r.W = spatial_r.w_sum / max(spatial_r.M * p_hat_winner, 0.0001);

                        vec3 shadow_dir = spatial_r.light_pos - hitPos;
                        float shadow_dist = max(length(shadow_dir), 0.0001);
                        shadow_dir /= shadow_dist;

                        if (dot(hit_normal, shadow_dir) > 0.0) {
                            uint shadow_ray_flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
                            prd.dist = 1.0;
                            traceRayEXT(tlas, shadow_ray_flags, 0xFF, 0, 0, 0, hitPos, 0.001, shadow_dir, shadow_dist - 0.001, 0);

                            if (prd.dist < 0.0) {
                                radiance += f_y_winner * throughput * spatial_r.W;
                            }
                        }
                    }
                } else if (roughness > 0.2) {
                    // Standard NEE for Indirect Bounces
                    uint light_idx = min(uint(rnd() * num_lights), num_lights - 1);
                    emissive_triangle_t light = emissive_triangles[light_idx];

                    float r1_nee = rnd();
                    float r2_nee = rnd();
                    float sqr1 = sqrt(r1_nee);
                    float u = 1.0 - sqr1;
                    float v = r2_nee * sqr1;
                    float w = 1.0 - u - v;
                    vec3 light_pos = light.v0_area.xyz * u + light.v1.xyz * v + light.v2.xyz * w;
                    vec3 light_normal = normalize(cross(light.v1.xyz - light.v0_area.xyz, light.v2.xyz - light.v0_area.xyz));

                    vec3 shadow_ray_dir = light_pos - hitPos;
                    float light_dist = length(shadow_ray_dir);
                    shadow_ray_dir /= light_dist;

                    float cos_theta_light = max(dot(light_normal, -shadow_ray_dir), 0.0);
                    float cos_theta_surface = max(dot(hit_normal, shadow_ray_dir), 0.0);
                    if (cos_theta_light > 0.0 && cos_theta_surface > 0.0) {
                        uint shadow_ray_flags = gl_RayFlagsTerminateOnFirstHitEXT |
                        gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
                        prd.dist = 1.0;

                        traceRayEXT(tlas, shadow_ray_flags, 0xFF, 0, 0, 0, hitPos, 0.001, shadow_ray_dir, light_dist - 0.001, 0);
                        if (prd.dist < 0.0) {
                            float light_area = light.v0_area.w;
                            float solid_angle_pdf = (light_dist * light_dist) / (cos_theta_light * light_area * float(num_lights));
                            radiance += (light.emission.rgb * hit_albedo * throughput * cos_theta_surface) / (solid_angle_pdf * 3.14159);
                        }
                    }
                }
            }

            vec3 N = hit_normal;
            vec3 F0 = mix(vec3(0.04), hit_albedo, metallic);
            float cos_theta = max(dot(N, V_view), 0.0);
            vec3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
            float p_specular = clamp(max(F.r, max(F.g, F.b)), 0.05, 1.0);

            float r1, r2;
            if (bounce == 0) {
                r1 = fract(blue_noise.r + float(frame_count % 1024) * 0.618034);
                r2 = fract(blue_noise.g + float(frame_count % 1024) * 0.618034);
            } else {
                r1 = rnd();
                r2 = rnd();
            }

            if (rnd() < p_specular) {
                vec3 H = get_ggx_microfacet(N, roughness, r1, r2);
                rayDir = reflect(-V_view, H);

                if (dot(N, rayDir) <= 0.0) {
                    rayDir = get_random_bounce(N, r1, r2);
                }
                throughput *= (F / p_specular);
            } else {
                rayDir = get_random_bounce(N, r1, r2);
                throughput *= hit_albedo * (1.0 - metallic) * (1.0 - F) / (1.0 - p_specular);
            }

            float p = max(throughput.r, max(throughput.g, throughput.b));
            if (p < 0.001) {
                break;
            }

            if (bounce > 2) {
                if (rnd() > p) break;
                throughput /= p;
            }

            rayOrigin = hitPos + hit_normal * 0.001;
        }

        total_radiance += radiance;
        total_radiance = min(total_radiance, 10.0);
    }

    vec3 current_frame_color = total_radiance / float(SAMPLES);
    vec3 raw_lighting = current_frame_color;

    imageStore(raw_color_image, ivec2(gl_LaunchIDEXT.xy), vec4(raw_lighting, 1.0));
}