#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint frame_count;
    int step_width;
} pc;

layout(set = 0, binding = 0, r11f_g11f_b10f) uniform readonly image2D temporal_result; // Input
layout(set = 0, binding = 1) uniform sampler2D depth_image;
layout(set = 0, binding = 2) uniform sampler2D normal_image;
layout(set = 0, binding = 3, r11f_g11f_b10f) uniform writeonly image2D spatial_output;

float get_luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722)); // [cite: 62]
}

// Fixed B-Spline weights for 5x5 A-Trous
const float kernel[5] = { 1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0 };

void main() {
    ivec2 size = imageSize(spatial_output);
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) return;

    vec3 center_color = imageLoad(temporal_result, pixel_coords).rgb;
    float center_depth = texelFetch(depth_image, pixel_coords, 0).r;
    vec3 center_normal = texelFetch(normal_image, pixel_coords, 0).rgb;

    //imageStore(spatial_output, pixel_coords, vec4(center_color, 1.0));
    //return;
    // Early bypass for sky - bypass tonemapping as it's now a separate pass
    if (center_depth >= 10000.0) {
        imageStore(spatial_output, pixel_coords, vec4(center_color, 1.0));
        return;
    }

    vec3 sum_color = vec3(0.0);
    float sum_weight = 0.0; // [cite: 69]

    float DEPTH_SENSITIVITY = 1;
    float NORMAL_SENSITIVITY = 80.0; // Tighter angle rejection

    float center_luma = get_luminance(center_color);

    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            ivec2 sample_offset = ivec2(x, y) * pc.step_width;
            ivec2 sample_coord = clamp(pixel_coords + sample_offset, ivec2(0), size - 1);

            vec3 sample_color = imageLoad(temporal_result, sample_coord).rgb;
            float sample_depth = texelFetch(depth_image, sample_coord, 0).r;
            vec3 sample_normal = texelFetch(normal_image, sample_coord, 0).rgb;

            float sample_luma = get_luminance(sample_color);

            // Edge-stopping weights
            float w_depth = exp(-abs(center_depth - sample_depth) * DEPTH_SENSITIVITY);
            float w_normal = exp((dot(center_normal, sample_normal) - 1.0) * NORMAL_SENSITIVITY);

            float luma_diff = abs(center_luma - sample_luma);
            float luma_sigma = max(center_luma, sample_luma) * 0.4 + 0.01;
            float w_luma = exp(-luma_diff / luma_sigma);

            // Combine
            float weight = w_depth * w_normal * w_luma * kernel[x + 2] * kernel[y + 2];

            sum_color += sample_color * weight;
            sum_weight += weight;
        }
    }

    vec3 spatially_denoised_color = sum_color / max(sum_weight, 0.0001);

    imageStore(spatial_output, pixel_coords, vec4(spatially_denoised_color, 1.0));
}