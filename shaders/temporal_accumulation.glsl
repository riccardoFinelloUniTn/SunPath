#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint frame_count;
} pc;

layout(set = 0, binding = 0) uniform sampler2D raw_rt_color;
layout(set = 0, binding = 1, rg16f)   uniform readonly image2D motion_vector_image;
layout(set = 0, binding = 2, r11f_g11f_b10f) uniform image2D accumulation_images[2];
layout(set = 0, binding = 3)          uniform sampler2D history_samplers[2];

const float ACCUMULATION_FACTOR = 0.25;
const float COLOR_THRESHOLD = 0.1;

vec3 get_historical_color(uint history_idx, vec2 uv, vec3 current_color) {
    if (pc.frame_count == 0) return current_color;
    return texture(history_samplers[history_idx], uv).rgb;
}

float get_luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 perform_temporal_accumulation(vec3 current_color, sampler2D history_sampler, vec2 uv, vec2 motion_vector, uint frame_count) {
    vec2 prev_uv = uv - motion_vector;
    bool is_off_screen = any(lessThan(prev_uv, vec2(0.0))) || any(greaterThan(prev_uv, vec2(1.0)));

    if (is_off_screen) return current_color;


    vec3 history_color = texture(history_sampler, prev_uv).rgb;


    vec3 diff = abs(history_color.rgb - current_color.rgb);
    float max_diff = max(diff.r, max(diff.g, diff.b));

    if (max_diff > COLOR_THRESHOLD) {
        return current_color;
    }

    return mix(history_color, current_color, ACCUMULATION_FACTOR);
}

void main() {
    ivec2 size = imageSize(accumulation_images[0]);
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) return;

    vec3 current_color = texelFetch(raw_rt_color, pixel_coords, 0).rgb;
    vec2 uv = (vec2(pixel_coords) + 0.5) / vec2(size);

    // We look at the 3x3 area around the current pixel to see what colors are "legal"
    vec3 min_color = current_color;
    vec3 max_color = current_color;
    float center_luma = get_luminance(current_color);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            ivec2 neighbor_coords = clamp(pixel_coords + ivec2(x, y), ivec2(0), size - 1);
            vec3 neighbor_color = texelFetch(raw_rt_color, neighbor_coords, 0).rgb;

            float neighbor_luma = get_luminance(neighbor_color);
            if (abs(neighbor_luma - center_luma) < max(center_luma * 5.0, 0.5)) {
                min_color = min(min_color, neighbor_color);
                max_color = max(max_color, neighbor_color);
            }
        }
    }
    vec2 motion_vector = imageLoad(motion_vector_image, pixel_coords).rg;


    vec2 prev_uv = uv - motion_vector;

    uint history_idx = pc.frame_count % 2;
    uint accum_idx   = (pc.frame_count + 1) % 2;

    // Default to current if off-screen or first frames
    vec3 accumulated_color = current_color;

    bool is_off_screen = any(lessThan(prev_uv, vec2(0.0))) || any(greaterThan(prev_uv, vec2(1.0)));

    if (!is_off_screen && pc.frame_count > 2) {
        //Sub-pixel Fetch (Bilinear)
        vec3 history_color = texture(history_samplers[history_idx], prev_uv).rgb;

        // If it's a "ghost" (color from an old object), it gets crushed to the new color.
        vec3 clamped_history = clamp(history_color, min_color, max_color);

        accumulated_color = mix(clamped_history, current_color, ACCUMULATION_FACTOR);
    }

    //imageStore(accumulation_images[accum_idx], pixel_coords, vec4(motion_vector * 100000.0, 0.0, 1.0));
    //return;
    imageStore(accumulation_images[accum_idx], pixel_coords, vec4(accumulated_color, 1.0));
    //imageStore(accumulation_images[accum_idx], pixel_coords, vec4(mix(motion_vector.r, accumulated_color.r, 0.5),mix(motion_vector.g, accumulated_color.g, 0.5),accumulated_color.b, 1.0));


}