#version 450

// Final composite pass:
//   1. Add bloom on top of the raw HDR scene
//   2. Apply exposure-based tone mapping  (1 - e^{-x * exposure})
//   3. Apply gamma correction (sRGB, γ = 2.2)
//
// Input images are in linear HDR space (R16G16B16A16_SFLOAT).
// Output is written to the swapchain image (LDR, display-referred sRGB).

layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrSampler;
layout(set = 0, binding = 1) uniform sampler2D bloomSampler;

layout(push_constant) uniform PC {
    float exposure;      // overall brightness multiplier before tone mapping
    float bloomStrength; // how strongly the bloom layer is mixed in
} pc;

void main()
{
    vec3 hdr   = texture(hdrSampler,   fragUV).rgb;
    vec3 bloom = texture(bloomSampler, fragUV).rgb;

    // Additive bloom
    vec3 result = hdr + bloom * pc.bloomStrength;

    // Exposure tone mapping:  maps [0, ∞) → [0, 1)
    result = vec3(1.0) - exp(-result * pc.exposure);

    // Gamma correction
    result = pow(result, vec3(1.0 / 2.2));

    outColor = vec4(result, 1.0);
}
