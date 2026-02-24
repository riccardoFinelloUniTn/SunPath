#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Push Constants so we know what frame we are on
layout(push_constant) uniform PushConstants {
    uint frame_count;
} pc;

// Raw RT Output and Final Screen Output
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D raw_rt_color;
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D output_image;


layout(set = 0, binding = 2, r32f) uniform readonly image2D depth_image;
layout(set = 0, binding = 3, rgba16f) uniform readonly image2D normal_image;
layout(set = 0, binding = 4, rg16f) uniform readonly image2D motion_vector_image;

layout(set = 0, binding = 5) uniform sampler2D history_samplers[2];
layout(set = 0, binding = 6, rgba32f) uniform image2D accumulation_images[2];


void main() {
    ivec2 size = imageSize(output_image);
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    if (uv.x >= size.x || uv.y >= size.y) return;

    vec3 centerColor = imageLoad(raw_rt_color, uv).rgb;

    vec3 minColor = vec3(10000.0);
    vec3 maxColor = vec3(-10000.0);
    vec3 avgColor = vec3(0.0);

    int count = 0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            // Skip the center pixel for the stats
            if (x == 0 && y == 0) continue;

            ivec2 neighborPos = uv + ivec2(x, y);

            // Bounds check
            if (neighborPos.x < 0 || neighborPos.y < 0 ||
            neighborPos.x >= size.x || neighborPos.y >= size.y) continue;

            vec3 c = imageLoad(raw_rt_color, neighborPos).rgb;

            minColor = min(minColor, c);
            maxColor = max(maxColor, c);
            avgColor += c;
            count++;
        }
    }

    avgColor /= float(count);


    vec3 clampedColor = clamp(centerColor, minColor, maxColor);

    float mixFactor = 0.2;
    vec3 finalColor = mix(clampedColor, avgColor, mixFactor);

    //temporarly disabled. Use finalcolor instead of centercolor

    imageStore(output_image, uv, vec4(centerColor, 1.0));
}