#version 460
#extension GL_EXT_ray_tracing : require
#include <shaders/common.glsl>

layout(location = 0) rayPayloadInEXT ray_payload_t payload;

void main() {
    payload.type = 1; // 1 = Miss/Sky
    payload.albedo = vec3(0.0);
    payload.emission = vec3(0.0);
}