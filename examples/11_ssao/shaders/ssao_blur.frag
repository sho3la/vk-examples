#version 450

// ---------------------------------------------------------------------------
// SSAO blur fragment shader — 4x4 box blur.
//
// The raw SSAO image is noisy (one random kernel rotation per pixel).
// A simple box blur over 16 neighbours smooths the result without
// introducing significant halo artefacts, because SSAO already operates
// at full resolution and the noise pattern has no spatial structure.
// ---------------------------------------------------------------------------

layout(location = 0) in vec2 inUV;

layout(set = 0, binding = 0) uniform sampler2D ssaoInput;

layout(location = 0) out float outOcclusion;

void main()
{
    vec2 texelSize = 1.0 / vec2(textureSize(ssaoInput, 0));
    float result = 0.0;
    for (int x = -2; x < 2; ++x)
        for (int y = -2; y < 2; ++y)
            result += texture(ssaoInput, inUV + vec2(x, y) * texelSize).r;
    outOcclusion = result / 16.0;
}
