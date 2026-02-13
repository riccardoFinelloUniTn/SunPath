#version 460
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba32f) uniform image2D inputImage;
layout(set = 0, binding = 1, rgba32f) uniform image2D outputImage;

// Tunable parameters
const int KERNEL_RADIUS = 2;
const float SIGMA_SPATIAL = 10.0; // Distance weight
const float SIGMA_COLOR = 0.5;    // Color edge weight (Low = strictly preserve edges)

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(inputImage);

    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) {
        return;
    }

    vec4 center_color = imageLoad(inputImage, pixel_coords);

    // Optimization: If the pixel is pure black/empty, skip
    if (center_color.a == 0.0) {
        imageStore(outputImage, pixel_coords, center_color);
        return;
    }

    vec3 sum_color = vec3(0.0);
    float sum_weight = 0.0;

    for (int x = -KERNEL_RADIUS; x <= KERNEL_RADIUS; ++x) {
        for (int y = -KERNEL_RADIUS; y <= KERNEL_RADIUS; ++y) {
            ivec2 neighbor_coords = pixel_coords + ivec2(x, y);

            // Bounds check
            if (neighbor_coords.x < 0 || neighbor_coords.x >= size.x ||
            neighbor_coords.y < 0 || neighbor_coords.y >= size.y) {
                continue;
            }

            vec4 neighbor_color = imageLoad(inputImage, neighbor_coords);

            // Gaussian weight
            float dist2 = float(x*x + y*y);
            float w_spatial = exp(-(dist2) / (2.0 * SIGMA_SPATIAL * SIGMA_SPATIAL));

            //Pixels with similar color matter more (Edge preserving)
            vec3 diff_color = center_color.rgb - neighbor_color.rgb;
            float color_dist2 = dot(diff_color, diff_color);
            float w_color = exp(-(color_dist2) / (2.0 * SIGMA_COLOR * SIGMA_COLOR));

            // Combined weight
            float weight = w_spatial * w_color;

            sum_color += neighbor_color.rgb * weight;
            sum_weight += weight;
        }
    }

    // Normalize
    vec3 final_color = sum_color / sum_weight;

    // Store result
    imageStore(outputImage, pixel_coords, vec4(final_color, 1.0));
}