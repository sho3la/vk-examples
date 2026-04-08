#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec4 fragLightSpacePos;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    vec4 lightDir;
    vec4 lightColor;
    vec4 viewPos;
    mat4 lightSpaceMatrix;
    vec4 ambient;
} ubo;

layout(set = 0, binding = 1) uniform sampler2D    diffuseSampler;
layout(set = 0, binding = 2) uniform sampler2DShadow shadowSampler;

float shadowFactor(vec4 lightSpacePos)
{
    vec3 ndc = lightSpacePos.xyz / lightSpacePos.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;

    // Outside light frustum → fully lit
    if (ndc.z > 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return 1.0;

	float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowSampler, 0);

    // Widen the sampling radius to blur out the pixelated stair-steps!
    // Tweak this number: 1.5 gives a soft edge, 2.0 gives a very soft/blurry edge
    float filterRadius = 1.5; 

    // 3x3 PCF Kernel
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 offset = vec2(x, y) * texelSize * filterRadius;
            shadow += texture(shadowSampler, vec3(uv + offset, ndc.z));
        }
    }
    
    return shadow / 9.0;
}

void main()
{
    vec3 albedo = texture(diffuseSampler, fragUV).rgb;

    vec3 N = normalize(fragNormal);
    vec3 L = normalize(ubo.lightDir.xyz);
    vec3 V = normalize(ubo.viewPos.xyz - fragWorldPos);
    vec3 H = normalize(L + V);

    // Ambient
    vec3 result = albedo * ubo.ambient.xyz * ubo.ambient.w;

    float shadow = shadowFactor(fragLightSpacePos);

    // Diffuse + specular, scaled by shadow factor
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 32.0);

    result += shadow * ubo.lightColor.w *
              (diff * albedo + spec * vec3(0.3)) * ubo.lightColor.xyz;

    outColor = vec4(result, 1.0);
}
