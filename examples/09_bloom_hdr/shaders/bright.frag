#version 450

// Bright-filter pass: extracts pixels whose perceived luminance exceeds the threshold.
// Pixels below the threshold are zeroed out so they contribute nothing to the bloom.

layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outBright;

layout(set = 0, binding = 0) uniform sampler2D hdrSampler;

layout(push_constant) uniform PC {
    float threshold; // luminance cutoff (e.g. 0.8)
} pc;

void main()
{
    vec3  color      = texture(hdrSampler, fragUV).rgb;
    // BT.709 luminance coefficients
    float luminance  = dot(color, vec3(0.2126, 0.7152, 0.0722));
    outBright        = luminance > pc.threshold ? vec4(color, 1.0) : vec4(0.0);
}
