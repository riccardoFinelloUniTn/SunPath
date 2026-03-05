#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include <shaders/common.glsl>
#include <shaders/utils.glsl>

layout(set = 0, binding = 1, r11f_g11f_b10f) uniform image2D raw_color_image; // This used to be your final output

layout(set = 0, binding = 5, r32f) uniform image2D depth_image;
layout(set = 0, binding = 6, rgba8_snorm) uniform image2D normal_image;
layout(set = 0, binding = 7, rg16f) uniform image2D motion_vector_image;

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

        for (int bounce = 0; bounce < 5; bounce++) {
            traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, rayOrigin, 0.001, rayDir, 10000.0, 0);

            // one time calculation of g-buffer values (depth, normal)
            if (bounce == 0) {

                float depth_value;
                vec3 normal_value;
                vec2 motion_value;
                vec2 test;

                if (prd.type == 1) { // Hit sky
                    depth_value = 100000.0; // Infinite distance
                    normal_value = vec3(0.0,0.0,0.0);
                    motion_value = vec2(0.0, 0.0);
                } else {
                    depth_value = prd.dist;
                    normal_value = prd.normal;
                    vec3 world_pos = rayOrigin + rayDir * prd.dist;
                    vec4 prev_clip = prev_view_proj * vec4(world_pos, 1.0);

                    vec2 prev_ndc = prev_clip.xy / prev_clip.w;
                    vec2 prev_uv = vec2(prev_ndc.x, -prev_ndc.y) * 0.5 + 0.5;

                    motion_value = inUV - prev_uv;
                    test = prev_uv;
                }
                imageStore(normal_image, ivec2(gl_LaunchIDEXT.xy), vec4(normal_value, 0.0));
                imageStore(depth_image, ivec2(gl_LaunchIDEXT.xy), vec4(depth_value, 0.0, 0.0, 0.0));
                imageStore(motion_vector_image, ivec2(gl_LaunchIDEXT.xy), vec4(motion_value, 0.0, 0.0));
            }

            //Hit sky
            if (prd.type == 1) {
                radiance += vec3(0.05, 0.05, 0.1) * throughput; // Ambient Sky Color
                break;
            }

            //Hit Object
            vec3 hitPos = rayOrigin + rayDir * prd.dist;

            float brightness = max(prd.emission.r, max(prd.emission.g, prd.emission.b));
            radiance += prd.emission * throughput;

            if (brightness > 1.0) {
                break;
            }

            throughput *= prd.albedo;

            if (bounce > 2) {
                float p = max(throughput.r, max(throughput.g, throughput.b));

                if (rnd() > p) break;

                if (p > 0.001) {
                    throughput /= p;
                } else {
                    break;
                }
            }

            //bounce
            rayDir    = get_random_bounce(prd.normal);
            rayOrigin = hitPos + prd.normal * 0.001;
        }

        total_radiance += radiance;
        total_radiance = min(total_radiance, 10.0);
    }

    vec3 current_frame_color = total_radiance / float(SAMPLES);

    imageStore(raw_color_image, ivec2(gl_LaunchIDEXT.xy), vec4(current_frame_color, 1.0));
}