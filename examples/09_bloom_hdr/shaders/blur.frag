#version 450

// Separable 9-tap Gaussian blur (σ ≈ 2).
// Reused for both the horizontal and vertical passes — the direction is
// controlled by the push constant.  Run H then V for a full 2-D Gaussian.

layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D inputSampler;

layout(push_constant) uniform PC {
    int horizontal; // 1 = horizontal pass,  0 = vertical pass
} pc;

// 9-tap kernel weights for σ ≈ 2 (sum = 1.0)
const float W[5] = float[5](
    0.2270270270,
    0.1945945946,
    0.1216216216,
    0.0540540541,
    0.0162162162
);

void main()
{
    vec2 texelSize = 1.0 / textureSize(inputSampler, 0);
    vec2 dir = (pc.horizontal == 1) ? vec2(texelSize.x, 0.0)
                                    : vec2(0.0, texelSize.y);

    vec3 result = texture(inputSampler, fragUV).rgb * W[0];
    for (int i = 1; i < 5; ++i) {
        result += texture(inputSampler, fragUV + dir * float(i)).rgb * W[i];
        result += texture(inputSampler, fragUV - dir * float(i)).rgb * W[i];
    }

    outColor = vec4(result, 1.0);
}
