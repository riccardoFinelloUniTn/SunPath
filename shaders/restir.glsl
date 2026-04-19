#ifndef SHADERS_RESTIR_GLSL
#define SHADERS_RESTIR_GLSL

#include <shaders/utils.glsl>

struct Reservoir {
    uint light_idx;
    uint _pad0[3];
    vec3 light_pos;
    float _pad1;
    vec3 light_normal;
    float w_sum;
    float M;
    float W;
    uint _pad2[2];
};

layout(std430, set = 0, binding = 11) buffer ReservoirBufferA { Reservoir reservoirs_A[]; };
layout(std430, set = 0, binding = 12) buffer ReservoirBufferB { Reservoir reservoirs_B[]; };

uint get_pixel_index(ivec2 coord, vec2 launch_size) {
    return coord.y * uint(launch_size.x) + coord.x;
}

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