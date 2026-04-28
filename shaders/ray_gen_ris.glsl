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

// Parities are uniform across the dispatch: cache them once.
uint g_current_buf_idx;
uint g_history_buf_idx;

Reservoir read_history_reservoir(ivec2 coord) {
    uint idx = get_pixel_index(coord, gl_LaunchSizeEXT.xy);
    return reservoirs[g_history_buf_idx].r[idx];
}

void write_current_reservoir(ivec2 coord, Reservoir r) {
    uint idx = get_pixel_index(coord, gl_LaunchSizeEXT.xy);
    reservoirs[g_current_buf_idx].r[idx] = r;
}

void write_current_gi_reservoir(ivec2 coord, ReservoirGI r) {
    uint idx = get_pixel_index(coord, gl_LaunchSizeEXT.xy);
    reservoirs_gi[g_current_buf_idx].r[idx] = r;
}

ReservoirGI read_history_gi_reservoir(ivec2 coord) {
    uint idx = get_pixel_index(coord, gl_LaunchSizeEXT.xy);
    return reservoirs_gi[g_history_buf_idx].r[idx];
}

void main() {
    init_rng(gl_LaunchIDEXT.xy, frame_count, gl_LaunchSizeEXT.xy);
    ivec2 pixel_coord = ivec2(gl_LaunchIDEXT.xy);

    g_current_buf_idx = current_reservoir_idx();
    g_history_buf_idx = history_reservoir_idx();

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
    vec2 prev_uv = vec2(-1.0);
    bool prev_valid = false; // true iff virtual_world_pos lies in front of the prev-frame camera
                             // (protects temporal reuse from w<=0 reprojection flipping prev_uv
                             //  into a random in-bounds pixel).

    bool found_diffuse_surface = false;
    float virtual_distance = 0.0;

    for (int virtual_bounce = 0; virtual_bounce < 20; virtual_bounce++) {
        traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xFF, 0, 0, 0, rayOrigin, 0.001, rayDir, 10000.0, 0);

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

            const float MIN_PREV_W = 0.01;
            prev_valid = prev_clip.w > MIN_PREV_W;
            if (prev_valid) {
                vec2 prev_ndc = prev_clip.xy / prev_clip.w;
                prev_uv = vec2(prev_ndc.x, prev_ndc.y) * 0.5 + 0.5;
                prev_valid = all(greaterThanEqual(prev_uv, vec2(0.0))) && all(lessThan(prev_uv, vec2(1.0)));
            }

            vec2 motion_vector = prev_valid ? (inUV - prev_uv) : (inUV + vec2(2.0));
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
        vec2 sky_motion = inUV + vec2(2.0); // default: guaranteed out-of-range, rejected by TAA
        vec4 sky_prev_clip = matrices_uniform_buffer.prev_view_proj * vec4(rayDir, 0.0);

        const float MIN_SKY_W = 0.01;
        if (sky_prev_clip.w > MIN_SKY_W) {
            vec2 sky_prev_ndc = sky_prev_clip.xy / sky_prev_clip.w;
            vec2 sky_prev_uv  = sky_prev_ndc * 0.5 + 0.5;
            if (all(greaterThanEqual(sky_prev_uv, vec2(0.0))) && all(lessThan(sky_prev_uv, vec2(1.0)))) {
                sky_motion = inUV - sky_prev_uv;
            }
        }

        imageStore(depth_image, pixel_coord, vec4(100000.0, 0.0, 0.0, 0.0));
        imageStore(normal_image, pixel_coord, vec4(0.0));
        imageStore(diffuse_image, pixel_coord, vec4(0.0));
        imageStore(motion_vector_image, pixel_coord, vec4(sky_motion, 0.0, 0.0));
        write_current_reservoir(pixel_coord, Reservoir(vec3(0), 0.0, vec3(0), 0.0, 0u, 0.0, 0u, 0.0));
        write_current_gi_reservoir(pixel_coord, ReservoirGI(vec3(0), 0.0, vec3(0), 0.0, 0u, 0.0, 0u, 0.0));
        return;
    }

    // Phase 2: RIS Initial Audition
    Reservoir current_r = Reservoir(vec3(0), 0.0, vec3(0), 0.0, 0u, 0.0, 0u, 0.0);
    uint num_lights = emissive_triangles.length();
    int RIS_CANDIDATES = 16;

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
        if (frame_count > 0 && prev_valid) {

            vec2 prev_pixel_f = prev_uv * vec2(gl_LaunchSizeEXT.xy);
            vec2 di_jitter    = vec2(rnd(), rnd()) - 0.5;
            ivec2 prev_coord  = ivec2(prev_pixel_f + di_jitter);
            if (prev_coord.x >= 0 && prev_coord.y >= 0 && prev_coord.x < gl_LaunchSizeEXT.x && prev_coord.y < gl_LaunchSizeEXT.y) {
                Reservoir history_r = read_history_reservoir(prev_coord);
                history_r.M = min(history_r.M, 10.0);
                history_r.W = min(history_r.W, 20.0);

                vec3 hist_normal = unpack_normal(history_r.hit_normal_packed);
                float normal_conf_di = smoothstep(0.9, 0.99, dot(hit_normal, hist_normal));
                float depth_diff_di  = abs(virtual_distance - history_r.depth) / max(virtual_distance, 1e-4);
                float depth_conf_di  = 1.0 - smoothstep(0.05, 0.20, depth_diff_di);
                float conf_di        = normal_conf_di * depth_conf_di;

                // Confidence-weight history M. conf=1 -> full memory (capped at 10),
                // conf=0 -> zero contribution, equivalent to the old hard reject.
                history_r.M *= conf_di;

                // Light index validity (light count may have changed)
                bool idx_ok = history_r.light_idx < num_lights;


                if (history_r.W > 0.0 && history_r.M > 0.0 && idx_ok) {
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

    // Store geometry for next frame's temporal reuse
    current_r.hit_normal_packed = pack_normal(hit_normal);
    current_r.depth = virtual_distance;

    write_current_reservoir(pixel_coord, current_r);

    // Phase 3: ReSTIR GI — Initial Sample (Ouyang 2021, Sec. 5.1)
    // One cosine-weighted indirect bounce from the shading point x1 to a surface sample x2.
    // No temporal/spatial reuse yet; M starts at 1 and W = 1/pdf when p_hat > 0.
    ReservoirGI current_gi_r = ReservoirGI(vec3(0), 0.0, vec3(0), 0.0, 0u, 0.0, 0u, 0.0);

    vec3 gi_dir = get_random_bounce(hit_normal, rnd(), rnd());
    float gi_NdotL = max(dot(hit_normal, gi_dir), 0.0);

    if (gi_NdotL > 0.0) {
        // Trace x1 -> x2 (reuses prd; DI work above is already done).
        vec3 gi_origin = hitPos + hit_normal * 0.001;
        traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xFF, 0, 0, 0, gi_origin, 0.001, gi_dir, 10000.0, 0);

        vec3 sample_pos = vec3(0.0);
        vec3 sample_normal = vec3(0.0);
        vec3 sample_radiance = vec3(0.0);

        if (prd.dist > 0.0) {
            sample_pos = gi_origin + gi_dir * prd.dist;
            sample_normal = unpack_normal(prd.normal_packed);
            vec3 x2_albedo = unpackUnorm4x8(prd.albedo_packed).rgb;

            // Direct emission visible at x2 along -gi_dir (BSDF-sampled contribution).
            sample_radiance = prd.emission;

            // 1-sample NEE at x2: pick a random emissive triangle, sample a point on it,
            // shadow-test and add the diffuse reflection toward x1. This is what turns the
            // GI reservoir from "only emissive-visibility" into real 1-bounce diffuse indirect
            // (Ouyang 2021, Sec. 5.1 — NEE at the reconnection vertex).
            uint nee_num_lights = emissive_triangles.length();
            if (nee_num_lights > 0) {
                uint nee_idx = min(uint(rnd() * nee_num_lights), nee_num_lights - 1);
                emissive_triangle_t nee_light = emissive_triangles[nee_idx];

                float sq = sqrt(rnd());
                float nu = 1.0 - sq;
                float nv = rnd() * sq;
                float nw = 1.0 - nu - nv;

                vec3 nee_pos    = nee_light.v0_area.xyz * nu + nee_light.v1.xyz * nv + nee_light.v2.xyz * nw;
                vec3 nee_normal = normalize(cross(nee_light.v1.xyz - nee_light.v0_area.xyz, nee_light.v2.xyz - nee_light.v0_area.xyz));

                vec3 to_light = nee_pos - sample_pos;
                float nee_dist = max(length(to_light), 0.0001);
                to_light /= nee_dist;

                float nee_cos_surf  = max(dot(sample_normal, to_light), 0.0);
                float nee_cos_light = max(dot(nee_normal, -to_light), 0.0);

                if (nee_cos_surf > 0.0 && nee_cos_light > 0.0) {
                    // No gl_RayFlagsOpaqueEXT: the any-hit shader must run so that alpha-cutout
                    // geometry can ignoreIntersectionEXT below its alpha_cutoff. Forcing opaque
                    // here was the root cause of the "alpha-masked leaves cast real shadows
                    // through GI" artifact — this NEE at x2 is exactly the path that feeds
                    // indirect-bounce-then-light shadowing into the GI reservoir.
                    uint nee_flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
                    prd.dist = 1.0;
                    traceRayEXT(tlas, nee_flags, 0xFF, 0, 0, 0, sample_pos + sample_normal * 0.001, 0.001, to_light, nee_dist - 0.001, 0);

                    if (prd.dist < 0.0) {
                        // Same diffuse-only NEE convention as the existing random-walk NEE:
                        // f_r = albedo / π, no (1-metallic) factor to match the final-pass formula.
                        float nee_area   = nee_light.v0_area.w;
                        float nee_pdf_sa = (nee_dist * nee_dist) / max(nee_cos_light * nee_area * float(nee_num_lights), 0.0001);
                        sample_radiance += (nee_light.emission.rgb * x2_albedo * nee_cos_surf) / (nee_pdf_sa * 3.14159);
                    }
                }
            }
        }

        // Firefly suppression: clamp per-channel radiance at x2 before it enters the reservoir.
        // The two spike sources for the "uniform surface + distant light" case are:
        //   (a) the cosine-weighted GI bounce happens to hit the light directly
        //       (prd.emission at x2 = huge), with probability ~ solid_angle_of_light / pi.
        //   (b) NEE at x2 grazes the light (nee_cos_light ~ 0 -> nee_pdf_sa small ->
        //       division explodes, even though the underlying radiance is modest).
        // Without a clamp, a single sample like that gets written to the GI reservoir and
        // kept alive by temporal reuse for up to M=20 frames, which reads as a persistent
        // bright pixel on an otherwise clean surface. Clamping at ~5 (well above direct
        // diffuse albedo*cosθ/π for any physically-plausible unoccluded light) kills the
        // spikes while leaving legitimate bright indirect (metal bounces, caustics) alone.
        const float GI_RADIANCE_CLAMP = 5.0;
        sample_radiance = min(sample_radiance, vec3(GI_RADIANCE_CLAMP));

        // Target function p_hat = luminance of the indirect contribution (diffuse lobe only at x1,
        // since specular indirect is handled by the final pass's BRDF-sampled bounce).
        float p_hat = gi_target_pdf(hitPos, hit_normal, hit_albedo, metallic, sample_pos, sample_radiance);
        float pdf   = gi_NdotL / 3.14159; // cosine-weighted sample pdf

        current_gi_r.M                    = 1.0;
        current_gi_r.w_sum                = (pdf > 0.0) ? (p_hat / pdf) : 0.0;
        current_gi_r.W                    = (p_hat > 0.0) ? (current_gi_r.w_sum / (current_gi_r.M * p_hat)) : 0.0;
        current_gi_r.sample_pos           = sample_pos;
        current_gi_r.sample_normal_packed = pack_normal(sample_normal);
        current_gi_r.sample_radiance      = sample_radiance;
    }

    // Reproject with the motion vector, validate normal+depth, cap M, merge with jacobian=1
    // (shading point is approximately unchanged across the temporal shift).


    if (frame_count > 0 && prev_valid) {
        // Same stochastic nearest-neighbor jitter as the DI path above — see comment there.
        // Using a fresh pair of rnd() draws so the GI jitter is uncorrelated with the DI
        // jitter; that's important because GI temporal aliasing manifests most visibly on
        // the same dimly-lit uniform surfaces as DI and correlated jitter would just shift
        // the banding rather than break it.
        vec2 prev_pixel_f_gi = prev_uv * vec2(gl_LaunchSizeEXT.xy);
        vec2 gi_jitter       = vec2(rnd(), rnd()) - 0.5;
        ivec2 prev_coord_gi  = ivec2(prev_pixel_f_gi + gi_jitter);
        if (prev_coord_gi.x >= 0 && prev_coord_gi.y >= 0 && prev_coord_gi.x < gl_LaunchSizeEXT.x && prev_coord_gi.y < gl_LaunchSizeEXT.y) {
            ReservoirGI history_gi = read_history_gi_reservoir(prev_coord_gi);

            // Soft geometry confidence instead of a hard pass/fail gate. The hard gate flips
            // abruptly at an iso-depth / iso-normal contour, which on flat surfaces is nearly
            // horizontal in screen space — the result is a visible horizontal seam where
            // temporally-accumulated pixels meet single-sample pixels. Replacing the boolean
            // with a smoothstep over both the normal and depth similarity, and using the
            // product as a multiplier on history M, makes the W-jump across that boundary
            // continuous so the seam becomes a smooth gradient instead of a bar.
            vec3 gi_hist_normal = unpack_normal(history_gi.hit_normal_packed);
            float normal_conf = smoothstep(0.8, 0.95, dot(hit_normal, gi_hist_normal));
            float depth_diff  = abs(virtual_distance - history_gi.depth) / max(virtual_distance, 1e-4);
            // Depth tolerance widened from (0.05, 0.12) to (0.05, 0.20) specifically for
            // forward/backward (dolly) motion: the old upper bound of 12% relative depth
            // change was tripped by modest forward steps on floors at glancing angles and
            // by any appreciable camera motion when the shading point is a reflection
            // (virtual-image depth changes ~2x the camera translation). The new 20% upper
            // bound keeps GI reservoirs alive through normal navigation speeds while the
            // 5% lower bound preserves responsiveness for actual geometry changes.
            float depth_conf  = 1.0 - smoothstep(0.05, 0.20, depth_diff);
            float conf        = normal_conf * depth_conf;

            // Cap + confidence-weight history M. Reduced from 20 to 12 specifically to break
            // the motion-time firefly comet trail: at 20, a spike reservoir persists across ~20
            // reprojections, and because reprojection uses stochastic nearest-neighbor jitter,
            // each frame it lands on a slightly different pixel → visible smear in the motion
            // direction. 12 is still enough memory for noise suppression on static pixels but
            // short enough that a firefly fades within a few frames of motion.
            history_gi.M = min(history_gi.M, 12.0) * conf;
            // Clamp history W on read, same rationale as the DI path above: bound per-hop
            // contribution of a potentially-spiky history reservoir so it can't amplify through
            // the merge weight = p_hat * W * M * jacobian and dominate the output.
            history_gi.W = min(history_gi.W, 10.0);

            if (history_gi.W > 0.0 && history_gi.M > 0.0) {
                float p_hat_hist = gi_target_pdf(hitPos, hit_normal, hit_albedo, metallic, history_gi.sample_pos, history_gi.sample_radiance);

                merge_reservoirs_gi(current_gi_r, history_gi, p_hat_hist, 1.0, rnd());

                // Floor on p_hat_merged prevents a degenerate reconnection (tiny NdotL or
                // near-black radiance) from turning W into a screen-space spike, which also
                // reads as a band when it lines up with the confidence boundary.
                float p_hat_merged = gi_target_pdf(hitPos, hit_normal, hit_albedo, metallic, current_gi_r.sample_pos, current_gi_r.sample_radiance);
                current_gi_r.W = (p_hat_merged > 1e-6) ? (current_gi_r.w_sum / (current_gi_r.M * p_hat_merged)) : 0.0;
            }
        }
    }


    // Store geometry for next frame's temporal validation (same scheme as DI reservoir).
    current_gi_r.hit_normal_packed = pack_normal(hit_normal);
    current_gi_r.depth             = virtual_distance;

    write_current_gi_reservoir(pixel_coord, current_gi_r);
}