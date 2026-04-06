#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

struct SpotLight {
    vec4 position;    // xyz = position,  w = unused
    vec4 direction;   // xyz = direction, w = unused
    vec4 color;       // xyz = color,     w = intensity
    vec4 cutoffs;     // x = cos(innerCutOff), y = cos(outerCutOff), zw = unused
};

layout(set = 0, binding = 0) uniform LightUBO {
    SpotLight lights[3];
    vec4      viewPos;   // xyz = camera position
    vec4      ambient;   // xyz = ambient color, w = ambient strength
} ubo;

// Procedural checkerboard on XZ plane
vec3 checkerboard(vec2 xz)
{
    float scale = 2.0;
    ivec2 cell  = ivec2(floor(xz * scale));
    float check = mod(float(cell.x + cell.y), 2.0);
    vec3 dark   = vec3(0.15, 0.15, 0.18);
    vec3 bright = vec3(0.6, 0.6, 0.65);
    return mix(dark, bright, check);
}

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.viewPos.xyz - fragWorldPos);

    // Base colour from checkerboard
    vec3 baseColor = checkerboard(fragWorldPos.xz);

    // Ambient
    vec3 result = ubo.ambient.xyz * ubo.ambient.w * baseColor;

    // Accumulate contribution from each spot light
    for (int i = 0; i < 3; ++i)
    {
        vec3  lightPos   = ubo.lights[i].position.xyz;
        vec3  lightDir   = normalize(ubo.lights[i].direction.xyz);
        vec3  lightColor = ubo.lights[i].color.xyz;
        float intensity  = ubo.lights[i].color.w;
        float innerCos   = ubo.lights[i].cutoffs.x;
        float outerCos   = ubo.lights[i].cutoffs.y;

        vec3  L    = lightPos - fragWorldPos;
        float dist = length(L);
        L = L / dist;  // normalize

        // Spotlight cone: angle between -L and light direction
        float theta   = dot(-L, lightDir);
        float epsilon = innerCos - outerCos;
        float spotFactor = clamp((theta - outerCos) / epsilon, 0.0, 1.0);

        // Attenuation (quadratic)
        float attenuation = intensity / (1.0 + 0.22 * dist + 0.20 * dist * dist);

        // Diffuse (Lambert)
        float diff = max(dot(N, L), 0.0);

        // Specular (Blinn-Phong)
        vec3  H    = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), 64.0);

        result += (diff * baseColor + spec * vec3(0.5)) * lightColor * attenuation * spotFactor;
    }

    outColor = vec4(result, 1.0);
}
