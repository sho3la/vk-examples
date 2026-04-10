#version 450

// Minimal vertex shader for unlit emissive objects.
// The emissive color lives entirely in the push constant and is only read by the fragment stage.

layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 color; // HDR emissive color (values > 1.0 cause bloom)
} pc;

layout(location = 0) in vec3 inPosition;
// inNormal and inUV are not declared — the emissive shader only needs position.
// The vertex buffer stride is still sizeof(Vertex); the GPU skips the unused bytes.

void main()
{
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
