#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position;   // xyz = position, w = unused
    vec4 color;      // xyz = color,    w = intensity
};

// Binding 0: uniform buffer (light data + camera)
layout(set = 0, binding = 0) uniform LightUBO {
    PointLight lights[3];
    vec4       viewPos;   // xyz = camera position
    vec4       ambient;   // xyz = ambient color, w = ambient strength
} ubo;

// Binding 1: combined image sampler (the diffuse texture)
layout(set = 0, binding = 1) uniform sampler2D texSampler;

void main()
{
    // Sample the texture
    vec3 albedo = texture(texSampler, fragUV).rgb;

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.viewPos.xyz - fragWorldPos);

    // Ambient
    vec3 result = albedo * ubo.ambient.xyz * ubo.ambient.w;

    // Accumulate contribution from each point light
    for (int i = 0; i < 3; ++i)
    {
        vec3  lightPos   = ubo.lights[i].position.xyz;
        vec3  lightColor = ubo.lights[i].color.xyz;
        float intensity  = ubo.lights[i].color.w;

        vec3  L    = lightPos - fragWorldPos;
        float dist = length(L);
        L = L / dist;

        // Quadratic attenuation
        float attenuation = intensity / (1.0 + 0.35 * dist + 0.44 * dist * dist);

        // Diffuse (Lambert)
        float diff = max(dot(N, L), 0.0);

        // Specular (Blinn-Phong)
        vec3  H    = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), 64.0);

        result += (diff * albedo + spec * vec3(0.5)) * lightColor * attenuation;
    }

    outColor = vec4(result, 1.0);
}
