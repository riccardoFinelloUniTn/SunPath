#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint frame_count;
} pc;

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D raw_rt_color;
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D output_image;

layout(set = 0, binding = 2, r32f) uniform readonly image2D depth_image;
layout(set = 0, binding = 3, rgba16f) uniform readonly image2D normal_image;
layout(set = 0, binding = 4, rg16f) uniform readonly image2D motion_vector_image;

layout(set = 0, binding = 5) uniform sampler2D history_samplers[2];
layout(set = 0, binding = 6, rgba32f) uniform image2D accumulation_images[2];

const float DISTANCE_FACTOR = 80;   //The factor of division to calculate the distance. Clamping creates artifacts and is expensive, so it's divided by a constant value.
const float ACCUMULATION_FACTOR = 1.0;      //factor to decide how to mix the current frame with the accumulated ones: 1.0 current frame, 0.0 previous frames

vec3 get_historical_color(uint history_idx, vec2 uv, vec3 current_color) {
    // If it's the first frame, we have no history, so return current color
    if (pc.frame_count == 0) {
        return current_color;
    }

    // Sample the accumulation buffer from the previous frame
    return texture(history_samplers[history_idx], uv).rgb;
}

// values based on human eye

float get_luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// ACES filmic tone mapping curve
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}


void main() {
    ivec2 size = imageSize(output_image);
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) return;

    // 1. Read the center pixel's G-Buffer data
    vec3 center_color = imageLoad(raw_rt_color, pixel_coords).rgb;
    float center_depth = imageLoad(depth_image, pixel_coords).r;
    vec3 center_normal = imageLoad(normal_image, pixel_coords).rgb;

    // If we hit the sky, bypass spatial filtering completely
    if (center_depth >= 10000.0 || get_luminance(center_color) > 100.0) {
        // Just pass the raw light color directly to temporal accumulation
        vec2 uv = (vec2(pixel_coords) + 0.5) / vec2(size);
        uint history_idx = pc.frame_count % 2;
        uint accum_idx   = (pc.frame_count + 1) % 2;

        vec3 history_color = get_historical_color(history_idx, uv, center_color);
        vec3 accumulated_color = mix(history_color, center_color, ACCUMULATION_FACTOR);

        imageStore(accumulation_images[accum_idx], pixel_coords, vec4(accumulated_color, 1.0));
        imageStore(output_image, pixel_coords, vec4(accumulated_color, 1.0));
        return;
    }

    // 2. Spatial Filtering Variables
    vec3 sum_color = vec3(0.0);
    float sum_weight = 0.0;

    int radius = 2; // 5x5 kernel

    float DEPTH_SENSITIVITY = 1.0;
    float NORMAL_SENSITIVITY = 64.0;
    float LUMA_SENSITIVITY = 2.0;     // NEW: How strictly to separate light/dark edges
    float MAX_LUMINANCE = 10.0;       // NEW: The absolute maximum energy allowed in the blur

    float center_luma = get_luminance(center_color);
    float sample_depth;
    vec3 sample_normal;

    // 3. Loop over the neighboring pixels
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            ivec2 offset = ivec2(x, y);
            ivec2 sample_coord = clamp(pixel_coords + offset, ivec2(0), size - 1);

            vec3 sample_color = imageLoad(raw_rt_color, sample_coord).rgb;
            sample_depth = imageLoad(depth_image, sample_coord).r;
            sample_normal = imageLoad(normal_image, sample_coord).rgb;

            float sample_luma = get_luminance(sample_color);

            // If the sample is absurdly bright, clamp its energy down so it doesn't nuke the average
            if (sample_luma > MAX_LUMINANCE) {
                sample_color *= (MAX_LUMINANCE / sample_luma);
                sample_luma = MAX_LUMINANCE;
            }

            //Calculate Depth & Normal Weights
            float depth_diff = abs(center_depth - sample_depth);
            float w_depth = exp(-depth_diff * DEPTH_SENSITIVITY);

            float normal_dot = max(0.0, dot(center_normal, sample_normal));
            float w_normal = pow(normal_dot, NORMAL_SENSITIVITY);

            // Calculate Luminance Weight
            // If the neighbor is much brighter/darker, drop its weight
            float luma_diff = abs(center_luma - sample_luma);
            float w_luma = exp(-luma_diff * LUMA_SENSITIVITY);

            // -- Combine Weights --
            float weight = w_depth * w_normal * w_luma;

            sum_color += sample_color * weight;
            sum_weight += weight;
        }
    }

    // 4. Final Spatially Denoised Color
    // Divide by sum_weight to normalize, use max() to prevent division by zero
    vec3 spatially_denoised_color = sum_color / max(sum_weight, 0.0001);

    // 5. Temporal Accumulation (Your existing logic)
    vec2 uv = (vec2(pixel_coords) + 0.5) / vec2(size);
    uint history_idx = pc.frame_count % 2;
    uint accum_idx   = (pc.frame_count + 1) % 2;

    vec3 history_color = get_historical_color(history_idx, uv, spatially_denoised_color);

    // Mix the history with our NEW filtered color, not the raw noisy one
    vec3 accumulated_color = mix(history_color, spatially_denoised_color, ACCUMULATION_FACTOR);

    float EXPOSURE = 3.0;
    vec3 exposed_color = accumulated_color * EXPOSURE;
    
    vec3 final_color = ACESFilm(exposed_color);

    //imageStore(accumulation_images[accum_idx], pixel_coords, vec4(accumulated_color, 1.0));
    //float d = sample_depth / DISTANCE_FACTOR;
    imageStore(output_image, pixel_coords, vec4(final_color, 1.0));
}