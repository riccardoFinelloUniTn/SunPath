
// Parameters:
// current_color: The noisy color calculated for this frame.
// history_sampler: The texture sampler for the previous frame.
// uv: The UV coordinates of the current pixel (0.0 to 1.0).
// motion_vector: (Future use) How much the pixel moved since last frame.
// frame_count: The total number of frames rendered so far.

vec3 perform_temporal_accumulation(
vec3 current_color,
sampler2D history_sampler,
vec2 uv,
vec2 motion_vector,
uint frame_count
) {

    vec2 prev_uv = uv - motion_vector;

    bool is_off_screen = any(lessThan(prev_uv, vec2(0.0))) || any(greaterThan(prev_uv, vec2(1.0)));

    if (is_off_screen) {
        return current_color; // No history available, return new color
    }

    vec3 history_color = texture(history_sampler, prev_uv).rgb;


    // Since you don't have motion vectors yet, simple accumulation (1/N) is best
    // to see the noise disappear perfectly.
    float blend_factor = (frame_count == 0) ? 1.0 : (1.0 / float(frame_count + 1));

    blend_factor = max(blend_factor, 0.01);

    return mix(history_color, current_color, blend_factor);
}