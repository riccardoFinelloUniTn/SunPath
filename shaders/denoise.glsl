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

// --- HELPER FUNCTIONS ---

// Determines which image index we read from (past) and which we write to (present)
void get_ping_pong_indices(out uint history_idx, out uint accum_idx) {
    history_idx = pc.frame_count % 2;
    accum_idx   = (pc.frame_count + 1) % 2;
}

vec3 get_historical_color(uint history_idx, vec2 uv, vec3 current_color) {
    // If it's the first frame, we have no history, so return current color
    if (pc.frame_count == 0) {
        return current_color;
    }

    // Sample the accumulation buffer from the previous frame
    return texture(history_samplers[history_idx], uv).rgb;
}

// --- MAIN ---

void main() {
    ivec2 size = imageSize(output_image);
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) return;

    vec2 uv = (vec2(pixel_coords) + 0.5) / vec2(size);

    vec3 current_color = imageLoad(raw_rt_color, pixel_coords).rgb;

    uint history_idx, accum_idx;
    get_ping_pong_indices(history_idx, accum_idx);

    vec3 history_color = get_historical_color(history_idx, uv, current_color);

    //float blend_factor = 1.0 / (float(pc.frame_count) + 1.0);
    vec3 accumulated_color = mix(history_color, current_color, 0.01);

    // 6. Persistence: Store the result for the NEXT frame to use as history
    imageStore(accumulation_images[accum_idx], pixel_coords, vec4(accumulated_color, 1.0));

    vec4 red = vec4(1.0,0.0,0.0, 1.0);

    // 7. Output: Show the result on the screen
    imageStore(output_image, pixel_coords, vec4(accumulated_color, 1.0));
}