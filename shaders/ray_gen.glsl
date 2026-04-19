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
layout(location = 0) rayPayloadEXT ray_payload_t prd;
//layout(location = 1) rayPayloadEXT bool is_shadow_miss;

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

void main() {

    vec3 total_radiance = vec3(0.0);
    int SAMPLES = 1;
    int BOUNCES = 10;
    int SHADOW_BOUNCES = BOUNCES / 2;
    int ROUGH_BOUNCES = (BOUNCES/ 3) + 1;
    init_rng(gl_LaunchIDEXT.xy, frame_count);

    vec3 primary_albedo = vec3(1.0);

    ivec2 tex_size = textureSize(blue_noise_tex, 0);
    ivec2 noise_coord = ivec2(gl_LaunchIDEXT.xy) % tex_size;
    vec4 blue_noise = texelFetch(blue_noise_tex, noise_coord, 0);

    for(int i = 0; i < SAMPLES; i++){
        const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
        const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
        vec2 d = inUV * 2.0 - 1.0;
        d.y = -d.y;

        vec4 origin    = matrices_uniform_buffer.view_inverse * vec4(0, 0, 0, 1);
        vec4 target    = matrices_uniform_buffer.proj_inverse * vec4(d.x, d.y, 1, 1);
        vec4 direction = matrices_uniform_buffer.view_inverse * vec4(normalize(target.xyz), 0);

        vec3 rayOrigin = origin.xyz;
        vec3 rayDir    = direction.xyz;

        vec3 throughput = vec3(1.0);
        vec3 radiance   = vec3(0.0);

        float virtual_dist = 0.0;
        bool gbuffer_written = false;

        bool in_glass = false;

        for (int bounce = 0; bounce < BOUNCES; bounce++) {

            uint ray_flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
            if (in_glass) {
                ray_flags = gl_RayFlagsOpaqueEXT; // Drop the cull flag
            }

            traceRayEXT(tlas, ray_flags, 0xFF, 0, 0, 0, rayOrigin, 0.001, rayDir, 10000.0, 0);

            bool is_sky = (prd.dist < 0.0);

            if (is_sky) {
                if (bounce == 0) {
                    imageStore(depth_image, ivec2(gl_LaunchIDEXT.xy), vec4(100000.0, 0.0, 0.0, 0.0));
                    imageStore(normal_image, ivec2(gl_LaunchIDEXT.xy), vec4(0.0));
                    imageStore(motion_vector_image, ivec2(gl_LaunchIDEXT.xy), vec4(0.0));
                }
                radiance += vec3(0.0, 0.0, 0.0) * throughput;
                break;
            }

            vec3 hit_normal = unpack_normal(prd.normal_packed);
            vec3 hit_albedo = unpackUnorm4x8(prd.albedo_packed).rgb;

            vec3 hitPos = rayOrigin + rayDir * prd.dist;

            vec2 mat_info = unpackHalf2x16(prd.material_info);
            float roughness = max(mat_info.x, 0.01);
            float metallic = clamp(mat_info.y, 0.0, 1.0);

            vec3 denoiser_albedo = mix(hit_albedo, vec3(1.0), metallic);

            //TODO Check utility of this
            if (bounce > 0) {
                float bias_strength = 0.5;
                float bias_weight = smoothstep(0.0, 0.5, roughness) * bias_strength;
                roughness = mix(roughness, 1.0, bias_weight);
            }

            vec2 trans_ior = unpackHalf2x16(prd.transmission_ior_packed);
            float transmission = trans_ior.x;
            float ior = max(trans_ior.y, 1.0);

            virtual_dist += prd.dist;

            if (bounce >= ROUGH_BOUNCES && roughness > 0.4 && !in_glass) {
                break;
            }

            if (bounce == 0) {
                imageStore(depth_image, ivec2(gl_LaunchIDEXT.xy), vec4(prd.dist, 0.0, 0.0, 0.0));
                imageStore(normal_image, ivec2(gl_LaunchIDEXT.xy), vec4(hit_normal, roughness));
                imageStore(diffuse_image, ivec2(gl_LaunchIDEXT.xy), vec4(denoiser_albedo, 0.0));

                vec3 world_pos = rayOrigin + rayDir * prd.dist;
                vec4 prev_clip = matrices_uniform_buffer.prev_view_proj * vec4(world_pos, 1.0);
                vec2 prev_ndc = prev_clip.xy / prev_clip.w;
                vec2 prev_uv = vec2(prev_ndc.x, -prev_ndc.y) * 0.5 + 0.5;

                imageStore(motion_vector_image, ivec2(gl_LaunchIDEXT.xy), vec4(inUV - prev_uv, 0.0, 0.0));
            }

            //Virtual gbuffer handling for very reflecting surfaces
            if (!gbuffer_written) {
                if (roughness > 0.1 || bounce == BOUNCES - 1) {


                    primary_albedo = denoiser_albedo;

                    // Write the Reflected Depth, Normal, and Diffuse
                    imageStore(depth_image, ivec2(gl_LaunchIDEXT.xy), vec4(virtual_dist, 0.0, 0.0, 0.0));
                    imageStore(normal_image, ivec2(gl_LaunchIDEXT.xy), vec4(hit_normal, roughness));
                    imageStore(diffuse_image, ivec2(gl_LaunchIDEXT.xy), vec4(denoiser_albedo, 0.0));

                    vec3 virtual_pos = origin.xyz + direction.xyz * virtual_dist;
                    vec4 prev_clip = matrices_uniform_buffer.prev_view_proj * vec4(virtual_pos, 1.0);
                    vec2 prev_ndc = prev_clip.xy / prev_clip.w;
                    vec2 prev_uv = vec2(prev_ndc.x, -prev_ndc.y) * 0.5 + 0.5;

                    imageStore(motion_vector_image, ivec2(gl_LaunchIDEXT.xy), vec4(inUV - prev_uv, 0.0, 0.0));

                    gbuffer_written = true;
                }
            }
            radiance += prd.emission * throughput;
            float brightness = max(prd.emission.r, max(prd.emission.g, prd.emission.b));
            if (brightness > 1.0) {
                break;
            }

            if (transmission > 0.5) {
                // A. Detect if we are entering or leaving the glass
                bool is_inside = dot(rayDir, hit_normal) > 0.0;

                //Flip the normal
                vec3 N = is_inside ? -hit_normal : hit_normal;

                // Calculate the IOR ratio (Assuming outside air is 1.0)
                float eta = is_inside ? (ior / 1.0) : (1.0 / ior);

                // D. Schlick's Approximation for Fresnel
                float cos_theta = min(dot(-rayDir, N), 1.0);
                float R0 = (1.0 - eta) / (1.0 + eta);
                R0 = R0 * R0;
                float fresnel = R0 + (1.0 - R0) * pow(1.0 - cos_theta, 5.0);

                // E. Snell's Law (Bend the ray)
                vec3 refracted = refract(rayDir, N, eta);

                //Total Internal Reflection (TIR)
                if (length(refracted) < 0.01) {
                    fresnel = 1.0; //Angle too steep, force a perfect mirror bounce inside
                }

                // Stochastic Branching: Reflect or Refract?
                if (rnd() < fresnel) {
                    // Reflection
                    rayDir = reflect(rayDir, N);
                } else {
                    // Refraction
                    rayDir = refracted;
                    in_glass != is_inside;

                    // Beer-Lambert Law (Volumetric Absorption)
                    if (is_inside) {
                        // The deeper the ray travels through tinted glass, the exponentially darker it gets
                        vec3 absorption = 1.0 - hit_albedo;
                        throughput *= exp(-absorption * prd.dist * 5.0); //5.0 is an arbitrary density multiplier
                    } else {
                        // Just entering the glass, multiply by the surface tint
                        throughput *= hit_albedo;
                    }
                }

                rayOrigin = hitPos + rayDir * 0.001;
                continue;
            }


            uint num_lights = emissive_indirection.length();
            if (num_lights > 0 && bounce < SHADOW_BOUNCES  && roughness > 0.2) {
                // Pick a random entry from the dense indirection buffer
                uint light_idx = min(uint(rnd() * num_lights), num_lights - 1);
                emissive_indirection_entry_t entry = emissive_indirection[light_idx];

                // Fetch local-space triangle and entity transform
                emissive_triangle_t light = emissive_triangles[entry.blas_tri_index];
                entity_transform_t xform = entity_transforms[entry.entity_id];

                // Transform vertices from local space to world space
                vec3 wv0 = vec3(dot(xform.rows[0], vec4(light.v0.xyz, 1.0)),
                                dot(xform.rows[1], vec4(light.v0.xyz, 1.0)),
                                dot(xform.rows[2], vec4(light.v0.xyz, 1.0)));
                vec3 wv1 = vec3(dot(xform.rows[0], vec4(light.v1.xyz, 1.0)),
                                dot(xform.rows[1], vec4(light.v1.xyz, 1.0)),
                                dot(xform.rows[2], vec4(light.v1.xyz, 1.0)));
                vec3 wv2 = vec3(dot(xform.rows[0], vec4(light.v2.xyz, 1.0)),
                                dot(xform.rows[1], vec4(light.v2.xyz, 1.0)),
                                dot(xform.rows[2], vec4(light.v2.xyz, 1.0)));

                // Compute world-space area from cross product
                vec3 edge1 = wv1 - wv0;
                vec3 edge2 = wv2 - wv0;
                float light_area = 0.5 * length(cross(edge1, edge2));

                // Pick a random point on the triangle using barycentrics
                float r1 = rnd();
                float r2 = rnd();
                float sqr1 = sqrt(r1);
                float u = 1.0 - sqr1;
                float v = r2 * sqr1;
                float w = 1.0 - u - v;

                vec3 light_pos = light.v0_area.xyz * u + light.v1.xyz * v + light.v2.xyz * w;
                vec3 light_normal = normalize(cross(light.v1.xyz - light.v0_area.xyz, light.v2.xyz - light.v0_area.xyz));

                // Setup the shadow ray
                vec3 shadow_ray_dir = light_pos - hitPos;
                float light_dist = length(shadow_ray_dir);
                shadow_ray_dir /= light_dist;

                // Check if the light is facing us, and we are facing the light
                float cos_theta_light = max(dot(light_normal, -shadow_ray_dir), 0.0);
                float cos_theta_surface = max(dot(hit_normal, shadow_ray_dir), 0.0);

                if (cos_theta_light > 0.0 && cos_theta_surface > 0.0) {
                    uint ray_flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
                    prd.dist = 1.0;

                    traceRayEXT(tlas, ray_flags, 0xFF, 0, 0, 0, hitPos, 0.001, shadow_ray_dir, light_dist - 0.001, 0);

                    // If it is < 0.0, ray_miss.glsl ran, meaning the light is visible
                    if (prd.dist < 0.0) {
                        float light_area = light.v0_area.w;
                        float solid_angle_pdf = (light_dist * light_dist) / (cos_theta_light * light_area * float(num_lights));
                        radiance += (light.emission.rgb * hit_albedo * throughput * cos_theta_surface) / (solid_angle_pdf * 3.14159);
                    }
                }
            }




            vec3 V = -rayDir;
            vec3 N = hit_normal;

            vec3 F0 = mix(vec3(0.04), hit_albedo, metallic);

            float cos_theta = max(dot(N, V), 0.0);
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
                rayDir = reflect(-V, H);

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
    vec3 raw_lighting = current_frame_color / max(primary_albedo, vec3(0.001));

    imageStore(raw_color_image, ivec2(gl_LaunchIDEXT.xy), vec4(raw_lighting, 1.0));
}