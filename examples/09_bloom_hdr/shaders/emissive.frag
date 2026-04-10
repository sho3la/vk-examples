#version 450

// Outputs the emissive push-constant color directly to the HDR attachment.
// Values above 1.0 will be extracted by the bright-filter pass and bloom.

layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 color;
} pc;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = pc.color;
}
