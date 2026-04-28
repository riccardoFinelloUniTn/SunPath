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
layout(std430, set = 0, binding = 13) buffer ReservoirBuffer { Reservoir r[]; } reservoirs[2];

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

// 48 bytes, std430-packed. Stores a surface sample (x2) used for indirect illumination.
// vec3+float pairs filling 16-byte slots, and an
// octahedral-packed sample_normal to keep the struct to 48 bytes (same as the DI reservoir).
struct ReservoirGI {
    vec3 sample_pos;            // offset 0  - world-space position of the indirect hit point (x2)
    float w_sum;                // offset 12 - sum of all candidate weights so far

    vec3 sample_radiance;       // offset 16 - incoming radiance from x2 toward x1 (emission + NEE at x2)
    float M;                    // offset 28 - number of candidate samples processed

    uint sample_normal_packed;  // offset 32 - octahedral-packed normal at x2 (for Jacobian / reconnection)
    float W;                    // offset 36 - final unbiased probabilistic weight, used to scale the indirect contribution
    uint hit_normal_packed;     // offset 40 - octahedral-packed normal at x1 (for temporal validation)
    float depth;                // offset 44 - virtual ray distance to x1 (for temporal validation)
};

// Descriptor array: reservoirs_gi[0] = buffer A, reservoirs_gi[1] = buffer B.
// Same ping-pong scheme as the DI reservoirs; index is uniform across the dispatch.
layout(std430, set = 0, binding = 13) buffer ReservoirGiBuffer { ReservoirGI r[]; } reservoirs_gi[2];

// Target pdf for the GI reservoir (luminance of the diffuse indirect contribution from the
// reconnection vertex x2 back to the shading point at (shade_pos, shade_normal)). Used
// symmetrically at reservoir init, temporal/spatial merge, and final W recomputation so all
// three see the same p_hat definition.
float gi_target_pdf(vec3 shade_pos, vec3 shade_normal, vec3 albedo, float metallic, vec3 sample_pos, vec3 sample_radiance) {
    vec3 w = sample_pos - shade_pos;
    float d = max(length(w), 0.0001);
    w /= d;
    float NdotL = max(dot(shade_normal, w), 0.0);
    vec3 f_diffuse = albedo * (1.0 - metallic) / 3.14159;
    vec3 contrib = sample_radiance * f_diffuse * NdotL;
    return max(contrib.r, max(contrib.g, contrib.b));
}

// GI reservoir merge. The jacobian term corrects the geometric change when a sample is reused
// from a neighboring pixel's shading point. For temporal reuse the
// shading point is unchanged, so callers pass jacobian = 1.0.
void merge_reservoirs_gi(inout ReservoirGI r, ReservoirGI new_r, float p_hat_new, float jacobian, float random_val) {
    r.M += new_r.M;
    float weight = p_hat_new * new_r.W * new_r.M * jacobian;
    r.w_sum += weight;

    if (random_val < (weight / max(r.w_sum, 0.0001))) {
        r.sample_pos = new_r.sample_pos;
        r.sample_normal_packed = new_r.sample_normal_packed;
        r.sample_radiance = new_r.sample_radiance;
    }
}

#endif