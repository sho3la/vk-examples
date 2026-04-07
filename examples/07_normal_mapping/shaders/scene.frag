#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in mat3 fragTBN; // uses locations 2, 3, 4

layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position;  // xyz = position, w = unused
    vec4 color;     // xyz = color,    w = intensity
};

layout(set = 0, binding = 0) uniform LightUBO {
    PointLight lights[3];
    vec4       viewPos;   // xyz = camera position
    vec4       ambient;   // xyz = color,  w = strength
} ubo;

layout(set = 0, binding = 1) uniform sampler2D diffuseSampler;
layout(set = 0, binding = 2) uniform sampler2D normalSampler;

void main()
{
    // --- Diffuse albedo ---
    vec3 albedo = texture(diffuseSampler, fragUV).rgb;

    // --- Normal from normal map (OpenGL convention, linear space) ---
    // Decode [0,1] → [-1,1] then rotate into world space via TBN.
    vec3 normalTS = texture(normalSampler, fragUV).rgb * 2.0 - 1.0;
    vec3 N = normalize(fragTBN * normalTS);

    vec3 V = normalize(ubo.viewPos.xyz - fragWorldPos);

    // --- Ambient term ---
    vec3 result = albedo * ubo.ambient.xyz * ubo.ambient.w;

    // --- Per-light Blinn-Phong ---
    for (int i = 0; i < 3; ++i)
    {
        vec3  L    = ubo.lights[i].position.xyz - fragWorldPos;
        float dist = length(L);
        L /= dist;

        float attenuation = ubo.lights[i].color.w /
                            (1.0 + 0.35 * dist + 0.44 * dist * dist);

        float diff = max(dot(N, L), 0.0);

        vec3  H    = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), 64.0);

        result += (diff * albedo + spec * vec3(0.4)) *
                   ubo.lights[i].color.xyz * attenuation;
    }

    outColor = vec4(result, 1.0);
}
