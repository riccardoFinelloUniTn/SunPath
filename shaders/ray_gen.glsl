#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include <shaders/common.glsl>
#include <shaders/utils.glsl>

#include <shaders/temporal_accumulation.glsl>

layout(location = 0) rayPayloadEXT ray_payload_t prd;



layout(set = 0, binding = 5) uniform sampler2D historyTextures[2];
layout(set = 0, binding = 6, rgba32f) uniform image2D accumulationImages[2];

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
    init_rng(gl_LaunchIDEXT.xy, 1);
    for(int i = 0; i < SAMPLES; i++){
        const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
        const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
        vec2 d = inUV * 2.0 - 1.0;
        d.y = -d.y;

        // TODO: Pass actual frame count from Rust push constants later.


        vec4 origin    = matrices_uniform_buffer.view_inverse * vec4(0, 0, 0, 1);
        vec4 target    = matrices_uniform_buffer.proj_inverse * vec4(d.x, d.y, 1, 1);
        vec4 direction = matrices_uniform_buffer.view_inverse * vec4(normalize(target.xyz), 0);

        vec3 rayOrigin = origin.xyz;
        vec3 rayDir    = direction.xyz;

        vec3 throughput = vec3(1.0);
        vec3 radiance   = vec3(0.0);

        for (int bounce = 0; bounce < 5; bounce++) {
            traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, rayOrigin, 0.001, rayDir, 10000.0, 0);

            //Hit sky
            if (prd.type == 1) {
                radiance += vec3(0.05, 0.05, 0.1) * throughput; // Ambient Sky Color
                break;
            }

            //Hit Object
            vec3 hitPos = rayOrigin + rayDir * prd.dist;

            radiance += prd.emission * throughput;

            throughput *= prd.albedo;

            if (bounce > 2) {
                float p = max(throughput.r, max(throughput.g, throughput.b));
                if (rnd() > p) break;
                throughput /= p;
            }

            //bounce
            rayDir    = get_random_bounce(prd.normal);
            rayOrigin = hitPos + prd.normal * 0.001;
        }

        total_radiance += radiance;

    }

    vec3 average_radiance = total_radiance / float(SAMPLES);

    vec3 current_frame_color = total_radiance / float(SAMPLES);


    uint history_idx = frame_count % 2;
    uint accum_idx = (frame_count + 1) % 2;
    // ----------------------

    // temporal accumulation logic
    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 uv = pixelCenter / vec2(gl_LaunchSizeEXT.xy);

    // DEBUG: History Read Test
    // Now 'historyTextures' and 'history_idx' are defined, so this works:
    vec3 history_debug = texture(historyTextures[history_idx], uv).rgb;

    vec3 history_color;

    if (frame_count == 0) {
        // Frame 0: No history exists yet. Use current color or black.
        history_color = current_frame_color;
    } else {
        // Frame > 0: Safe to read history
        history_color = texture(historyTextures[history_idx], uv).rgb;
    }

    // Debug Output
    imageStore(accumulationImages[accum_idx], ivec2(gl_LaunchIDEXT.xy), vec4(history_color, 1.0));


    // ...

    //imageStore(accumulationImage, ivec2(gl_LaunchIDEXT.xy), vec4(accumulated_color, 1.0));



    // simple tone mapping (Reinhard) and Gamma Correction for the screen
    //vec3 display_color = accumulated_color / (accumulated_color + vec3(1.0));
    //display_color = pow(display_color, vec3(1.0/2.2));
    //output

    //imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(average_radiance, 1.0));
}