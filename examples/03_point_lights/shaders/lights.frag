#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position;   // xyz = position, w = unused
    vec4 color;      // xyz = color,    w = intensity
};

layout(set = 0, binding = 0) uniform LightUBO {
    PointLight lights[3];
    vec4       viewPos;    // xyz = camera position
    vec4       ambient;    // xyz = ambient color, w = ambient strength
} ubo;

// Procedural checkerboard pattern on XZ plane
vec3 checkerboard(vec2 xz)
{
    float scale = 2.0;
    ivec2 cell  = ivec2(floor(xz * scale));
    float check = mod(float(cell.x + cell.y), 2.0);
	
    vec3 dark   = vec3(0.25, 0.24, 0.22);
    vec3 light  = vec3(0.33, 0.32, 0.30);
	
    return mix(dark, light, check);
}

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.viewPos.xyz - fragWorldPos);

    // Base colour from checkerboard
    vec3 baseColor = checkerboard(fragWorldPos.xz);

    // Ambient
    vec3 result = ubo.ambient.xyz * ubo.ambient.w * baseColor;

    // Accumulate contribution from each point light
    for (int i = 0; i < 3; ++i)
    {
        vec3  lightPos   = ubo.lights[i].position.xyz;
        vec3  lightColor = ubo.lights[i].color.xyz;
        float intensity  = ubo.lights[i].color.w;

        vec3  L    = lightPos - fragWorldPos;
        float dist = length(L);
        L = L / dist;  // normalize

        // Attenuation (quadratic)
        float attenuation = intensity / (1.0 + 0.35 * dist + 0.44 * dist * dist);

        // Diffuse (Lambert)
        float diff = max(dot(N, L), 0.0);

        // Specular (Blinn-Phong)
        vec3  H    = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), 64.0);

        result += (diff * baseColor + spec * vec3(0.5)) * lightColor * attenuation;
    }

    outColor = vec4(result, 1.0);
}
