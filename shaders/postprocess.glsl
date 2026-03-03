#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, r11f_g11f_b10f) uniform readonly image2D input_image;
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D output_image;

// ACES Fitted (Narkowicz approximation)
vec3 ACESFilm(vec3 x) {
    // Clamp to prevent Inf/Inf division if x is extremely high
    x = clamp(x, 0.0, 100.0);
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    ivec2 size = imageSize(input_image);
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) {
        return;
    }

    vec4 hdr_data = imageLoad(input_image, pixel_coords);
    vec3 color = hdr_data.rgb;

    if (any(isnan(color)) || any(isinf(color))) {
        color = vec3(0.0);
    }

    vec3 mapped = ACESFilm(color);

    vec3 final_color = pow(mapped, vec3(1.0 / 2.2));

    imageStore(output_image, pixel_coords, vec4(final_color, 1.0));
}