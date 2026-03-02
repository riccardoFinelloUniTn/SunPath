#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint frame_count;
} pc;

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D temporal_result;
layout(set = 0, binding = 1, r32f)    uniform readonly image2D depth_image;
layout(set = 0, binding = 2, rgba16f) uniform readonly image2D normal_image;
layout(set = 0, binding = 3, rgba32f) uniform writeonly image2D spatial_output;

float get_luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 ACESFilm(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
    ivec2 size = imageSize(spatial_output);
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) return;

    // Read from the Temporally Accumulated image, not the raw raytracer
    vec3 center_color = imageLoad(temporal_result, pixel_coords).rgb;
    float center_depth = imageLoad(depth_image, pixel_coords).r;
    vec3 center_normal = imageLoad(normal_image, pixel_coords).rgb;

    //imageStore(spatial_output, pixel_coords, vec4(center_color, 1.0));
    //return;

    // Early bypass for sky or extremely bright pixels
    if (center_depth >= 10000.0 || get_luminance(center_color) > 100.0) {
        float EXPOSURE = 3.0;
        vec3 final_color = ACESFilm(center_color * EXPOSURE);
        imageStore(spatial_output, pixel_coords, vec4(final_color, 1.0));
        return;
    }

    vec3 sum_color = vec3(0.0);
    float sum_weight = 0.0;
    int radius = 2;
    float DEPTH_SENSITIVITY = 1.0;
    float NORMAL_SENSITIVITY = 64.0;
    float LUMA_SENSITIVITY = 2.0;
    float MAX_LUMINANCE = 10.0;

    float center_luma = get_luminance(center_color);

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            ivec2 sample_coord = clamp(pixel_coords + ivec2(x, y), ivec2(0), size - 1);

            vec3 sample_color = imageLoad(temporal_result, sample_coord).rgb;
            float sample_depth = imageLoad(depth_image, sample_coord).r;
            vec3 sample_normal = imageLoad(normal_image, sample_coord).rgb;

            float sample_luma = get_luminance(sample_color);
            if (sample_luma > MAX_LUMINANCE) {
                sample_color *= (MAX_LUMINANCE / sample_luma);
                sample_luma = MAX_LUMINANCE;
            }

            float w_depth = exp(-abs(center_depth - sample_depth) * DEPTH_SENSITIVITY);
            float w_normal = pow(max(0.0, dot(center_normal, sample_normal)), NORMAL_SENSITIVITY);
            float w_luma = exp(-abs(center_luma - sample_luma) * LUMA_SENSITIVITY);

            float weight = w_depth * w_normal * w_luma;
            sum_color += sample_color * weight;
            sum_weight += weight;
        }
    }

    vec3 spatially_denoised_color = sum_color / max(sum_weight, 0.0001);

    // Apply Tonemapping here at the very end
    float EXPOSURE = 3.0;
    vec3 final_color = ACESFilm(spatially_denoised_color * EXPOSURE);

    imageStore(spatial_output, pixel_coords, vec4(final_color, 1.0));
}