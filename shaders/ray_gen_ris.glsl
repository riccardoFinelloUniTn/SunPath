#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include <shaders/common.glsl>
#include <shaders/utils.glsl>

layout(set = 0, binding = 1, r11f_g11f_b10f) uniform image2D raw_color_image;
layout(set = 0, binding = 5, r16f) uniform image2D depth_image;
layout(set = 0, binding = 6, rgba8_snorm) uniform image2D normal_image;
layout(set = 0, binding = 7, r11f_g11f_b10f) uniform image2D diffuse_image;
layout(set = 0, binding = 8, rg16f) uniform image2D motion_vector_image;
layout(set = 0, binding = 10) uniform sampler2D blue_noise_tex;


// ReSTIR Structure mathematically aligned to std430
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

// ReSTIR Ping-Pong Buffers
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



uint get_pixel_index(ivec2 coord) {
    return coord.y * gl_LaunchSizeEXT.x + coord.x;
}

Reservoir read_history_reservoir(ivec2 coord) {
    uint idx = get_pixel_index(coord);
    if (frame_count % 2 == 0) return reservoirs_B[idx];
    else return reservoirs_A[idx];
}

void write_current_reservoir(ivec2 coord, Reservoir r) {
    uint idx = get_pixel_index(coord);
    if (frame_count % 2 == 0) reservoirs_A[idx] = r;
    else reservoirs_B[idx] = r;
}

void update_reservoir(inout Reservoir r, uint cand_idx, vec3 cand_pos, vec3 cand_normal, float weight, float random_val) {
    r.w_sum += weight;
    r.M += 1.0;
    if (random_val < (weight / max(r.w_sum, 0.0001))) {
        r.light_idx = cand_idx;
        r.light_pos = cand_pos;
        r.light_normal = cand_normal;
    }
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

void main() {
    init_rng(gl_LaunchIDEXT.xy, frame_count);
    ivec2 pixel_coord = ivec2(gl_LaunchIDEXT.xy);

    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = inUV * 2.0 - 1.0;
    d.y = -d.y;

    vec4 origin    = matrices_uniform_buffer.view_inverse * vec4(0, 0, 0, 1);
    vec4 target    = matrices_uniform_buffer.proj_inverse * vec4(d.x, d.y, 1, 1);
    vec4 direction = matrices_uniform_buffer.view_inverse * vec4(normalize(target.xyz), 0);

    vec3 rayOrigin = origin.xyz;
    vec3 rayDir    = direction.xyz;

    // Phase 1: Shoot primary ray to find surface
    traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, rayOrigin, 0.001, rayDir, 10000.0, 0);

    if (prd.dist < 0.0) {
        write_current_reservoir(pixel_coord, Reservoir(0, uint[3](0,0,0), vec3(0), 0.0, vec3(0), 0.0, 0.0, 0.0, uint[2](0,0)));
        return;
    }

    vec3 hitPos = rayOrigin + rayDir * prd.dist;
    uint hx = hash(floatBitsToUint(hitPos.x));
    uint hy = hash(floatBitsToUint(hitPos.y));
    uint hz = hash(floatBitsToUint(hitPos.z));
    seed = hash(hx ^ hy ^ hz ^ frame_count);
    vec3 hit_normal = unpack_normal(prd.normal_packed);
    vec3 hit_albedo = unpackUnorm4x8(prd.albedo_packed).rgb;
    vec2 mat_info = unpackHalf2x16(prd.material_info);
    float roughness = max(mat_info.x, 0.01);
    float metallic = clamp(mat_info.y, 0.0, 1.0);
    vec3 V_view = -rayDir;

    // Phase 2: RIS Initial Audition
    Reservoir current_r = Reservoir(0, uint[3](0,0,0), vec3(0), 0.0, vec3(0), 0.0, 0.0, 0.0, uint[2](0,0));
    uint num_lights = emissive_triangles.length();
    int RIS_CANDIDATES = 8;

    if (num_lights > 0) {
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

            update_reservoir(current_r, cand_idx, cand_pos, cand_normal, p_hat / p_y, rnd());
        }

        if (current_r.w_sum > 0.0) {
            emissive_triangle_t winner = emissive_triangles[current_r.light_idx];
            vec3 f_y_winner = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, winner, current_r.light_pos, current_r.light_normal);
            float p_hat_winner = max(f_y_winner.r, max(f_y_winner.g, f_y_winner.b));
            current_r.W = current_r.w_sum / max(current_r.M * p_hat_winner, 0.0001);
        }

        // Phase 3: Temporal Reuse

        /*
        if (frame_count > 0) {
            vec4 prev_clip = matrices_uniform_buffer.prev_view_proj * vec4(hitPos, 1.0);
            vec2 prev_ndc = prev_clip.xy / prev_clip.w;
            vec2 prev_uv = vec2(prev_ndc.x, -prev_ndc.y) * 0.5 + 0.5;
            ivec2 prev_coord = ivec2(prev_uv * vec2(gl_LaunchSizeEXT.xy));

            if (prev_coord.x >= 0 && prev_coord.y >= 0 && prev_coord.x < gl_LaunchSizeEXT.x && prev_coord.y < gl_LaunchSizeEXT.y) {
                Reservoir history_r = read_history_reservoir(prev_coord);

                // Limit temporal history to maintain reactivity
                history_r.M = min(history_r.M, 20.0);

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

        */
    }


    write_current_reservoir(pixel_coord, current_r);
}