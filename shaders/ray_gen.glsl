#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include <shaders/common.glsl>
#include <shaders/utils.glsl>

layout(set = 0, binding = 1, r11f_g11f_b10f) uniform image2D raw_color_image; // This used to be your final output

layout(set = 0, binding = 5, r16f) uniform image2D depth_image;
layout(set = 0, binding = 6, rgba8_snorm) uniform image2D normal_image;
layout(set = 0, binding = 7, r11f_g11f_b10f) uniform image2D diffuse_image;
layout(set = 0, binding = 8, rg16f) uniform image2D motion_vector_image;

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

vec3 get_random_bounce(vec3 normal) {
    float r1 = rnd();
    float r2 = rnd();
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

    // Create an orthogonal basis around the normal
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    return tangent * h.x + bitangent * h.y + normal * h.z;
}

void main() {

    vec3 total_radiance = vec3(0.0);
    int SAMPLES = 1;
    init_rng(gl_LaunchIDEXT.xy, frame_count);
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

        for (int bounce = 0; bounce < 8; bounce++) {
            traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, rayOrigin, 0.001, rayDir, 10000.0, 0);

            bool is_sky = (prd.dist < 0.0);

            // SKY HANDLING
            if (is_sky) {
                if (bounce == 0) {
                    imageStore(depth_image, ivec2(gl_LaunchIDEXT.xy), vec4(100000.0, 0.0, 0.0, 0.0));
                    imageStore(normal_image, ivec2(gl_LaunchIDEXT.xy), vec4(0.0));
                    imageStore(motion_vector_image, ivec2(gl_LaunchIDEXT.xy), vec4(0.0));
                }
                radiance += vec3(0.05, 0.05, 0.1) * throughput; // Ambient Sky Color
                break;
            }

            // UNPACK ATTRIBUTES
            vec3 hit_normal = unpack_normal(prd.normal_packed);
            vec3 hit_albedo = unpackUnorm4x8(prd.albedo_packed).rgb;


            // G-BUFFER CALCULATION
            if (bounce == 0) {
                imageStore(depth_image, ivec2(gl_LaunchIDEXT.xy), vec4(prd.dist, 0.0, 0.0, 0.0));
                imageStore(normal_image, ivec2(gl_LaunchIDEXT.xy), vec4(hit_normal, 0.0));
                imageStore(diffuse_image, ivec2(gl_LaunchIDEXT.xy), vec4(hit_albedo, 0.0));

                vec3 world_pos = rayOrigin + rayDir * prd.dist;
                vec4 prev_clip = matrices_uniform_buffer.prev_view_proj * vec4(world_pos, 1.0);
                vec2 prev_ndc = prev_clip.xy / prev_clip.w;
                vec2 prev_uv = vec2(prev_ndc.x, -prev_ndc.y) * 0.5 + 0.5;

                imageStore(motion_vector_image, ivec2(gl_LaunchIDEXT.xy), vec4(inUV - prev_uv, 0.0, 0.0));
            }

            // 4. EMISSIVE BREAK
            radiance += prd.emission * throughput;
            float brightness = max(prd.emission.r, max(prd.emission.g, prd.emission.b));
            if (brightness > 1.0) {
                break;
            }

            vec3 hitPos = rayOrigin + rayDir * prd.dist;

            // 6. UNPACK MATERIAL
            vec2 mat_info = unpackHalf2x16(prd.material_info);
            float roughness = max(mat_info.x, 0.01); // Prevent pure-mirror div-by-zero
            float metallic = clamp(mat_info.y, 0.0, 1.0);

            vec3 V = -rayDir;
            vec3 N = hit_normal;

            // Calculate Base Reflectivity (F0)
            vec3 F0 = mix(vec3(0.04), hit_albedo, metallic);

            float cos_theta = max(dot(N, V), 0.0);
            vec3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);

            // Statistical Path Selection Probability
            float p_specular = clamp(max(F.r, max(F.g, F.b)), 0.05, 0.95);

            // 7. PBR BOUNCE & THROUGHPUT
            if (rnd() < p_specular) {
                // --- SPECULAR BOUNCE ---
                vec3 H = get_ggx_microfacet(N, roughness, rnd(), rnd());
                rayDir = reflect(-V, H);

                if (dot(N, rayDir) <= 0.0) {
                    rayDir = get_random_bounce(N);
                }

                throughput *= (F / p_specular);
            } else {
                // --- DIFFUSE BOUNCE ---
                rayDir = get_random_bounce(N);

                // Only tint with albedo if it's a diffuse bounce!
                throughput *= hit_albedo * (1.0 - metallic) * (1.0 - F) / (1.0 - p_specular);
            }

            // 8. RUSSIAN ROULETTE & EARLY TERMINATION
            // Calculate using the NEW throughput after PBR logic applied it
            float p = max(throughput.r, max(throughput.g, throughput.b));

            if (p < 0.001) {
                break; // Material absorbed almost everything, kill ray
            }

            if (bounce > 2) {
                if (rnd() > p) break;
                throughput /= p;
            }

            // 9. SETUP NEXT ORIGIN
            // (rayDir was already correctly set by the PBR block)
            rayOrigin = hitPos + hit_normal * 0.001;
        }

        total_radiance += radiance;
        total_radiance = min(total_radiance, 10.0);
    }

    vec3 current_frame_color = total_radiance / float(SAMPLES);

    imageStore(raw_color_image, ivec2(gl_LaunchIDEXT.xy), vec4(current_frame_color, 1.0));
}