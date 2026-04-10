#version 450

// Outputs raw HDR linear colour — no tone mapping here.
// Three coloured point lights illuminate the textured ground plane.
// The spheres themselves are unlit emissive objects drawn by emissive.frag.

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 pos;   // xyz = world position
    vec4 color; // xyz = colour, w = intensity
};

layout(set = 0, binding = 0) uniform SceneUBO {
    PointLight lights[3]; // red, green, blue — orbit the scene
    vec4       viewPos;   // xyz = camera world position
    vec4       ambient;   // xyz = colour, w = strength
} ubo;

layout(set = 0, binding = 1) uniform sampler2D groundTex;

void main()
{
    vec3 albedo = texture(groundTex, fragUV).rgb;

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.viewPos.xyz - fragWorldPos);

    // Start with ambient
    vec3 result = albedo * ubo.ambient.xyz * ubo.ambient.w;

    // Accumulate contribution from all three point lights
    for (int i = 0; i < 3; ++i) {
        vec3  L     = normalize(ubo.lights[i].pos.xyz - fragWorldPos);
        vec3  H     = normalize(L + V);
        float dist  = length(ubo.lights[i].pos.xyz - fragWorldPos);
        float atten = 1.0 / (1.0 + 0.045 * dist + 0.0075 * dist * dist);
        float diff  = max(dot(N, L), 0.0);
        float spec  = pow(max(dot(N, H), 0.0), 32.0);

        result += atten * ubo.lights[i].color.w * ubo.lights[i].color.xyz *
                  (diff * albedo + spec * vec3(0.25));
    }

    outColor = vec4(result, 1.0);
}
