#ifndef SHADERS_RESTIR_GLSL
#define SHADERS_RESTIR_GLSL

#include <shaders/utils.glsl>

// 48 bytes, std430-packed (vec3+float tightly packed, no explicit padding required)
struct Reservoir {
    vec3 light_pos;         // offset 0
    float w_sum;            // offset 12
    vec3 light_normal;      // offset 16
    float M;                // offset 28
    uint light_idx;         // offset 32
    float W;                // offset 36
    uint hit_normal_packed; // offset 40
    float depth;            // offset 44
};

// Descriptor array: reservoirs[0] = buffer A, reservoirs[1] = buffer B.
// Index is uniform across the dispatch (derived from push-constant frame_count parity),
// so no descriptor indexing extension is required.
layout(std430, set = 0, binding = 11) buffer ReservoirBuffer { Reservoir r[]; } reservoirs[2];

uint get_pixel_index(ivec2 coord, vec2 launch_size) {
    return coord.y * uint(launch_size.x) + coord.x;
}

// Parity selectors: index of the buffer to read history from / write current into.
uint current_reservoir_idx() { return frame_count & 1u; }
uint history_reservoir_idx() { return (frame_count & 1u) ^ 1u; }

void merge_reservoirs(inout Reservoir r, Reservoir new_r, float p_hat_new, float random_val) {
    r.M += new_r.M;
    float weight = p_hat_new * new_r.W * new_r.M;
    r.w_sum += weight;

    if (random_val < (weight / max(r.w_sum, 0.0001))) {
        r.light_idx = new_r.light_idx;
        r.light_pos = new_r.light_pos;
        r.light_normal = new_r.light_normal;
    }
}

#endif