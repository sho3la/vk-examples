#version 450

// Deferred lighting pass.
// Reads the three G-buffers written by the geometry pass and evaluates
// all 16 point lights in a single fullscreen draw — O(pixels × lights),
// independent of scene geometry complexity.
//
// Lighting model: Blinn-Phong with radius-based smooth falloff.
// Tone mapping: Reinhard, followed by sRGB gamma correction.

layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;

// G-buffer samplers (set 0)
layout(set = 0, binding = 0) uniform sampler2D gPosition; // xyz=world pos, a=geometry flag
layout(set = 0, binding = 1) uniform sampler2D gNormal;   // xyz=world normal
layout(set = 0, binding = 2) uniform sampler2D gAlbedo;   // rgb=albedo, a=roughness

// Light data (set 1)
struct PointLight {
    vec4 pos;   // xyz = world position,  w = radius
    vec4 color; // xyz = colour,          w = intensity
};

layout(set = 1, binding = 0) uniform LightingUBO {
    PointLight lights[16];
    vec4       viewPos; // xyz = camera world position
} ubo;

void main()
{
    // ---- Sample G-buffer ------------------------------------------------
    vec4  posA   = texture(gPosition, fragUV);
    vec3  worldPos = posA.xyz;
    float geomFlag = posA.a; // 1.0 = real geometry, 0.0 = background

    // Background: no geometry was drawn to this pixel
    if (geomFlag < 0.5) {
        outColor = vec4(0.01, 0.01, 0.02, 1.0); // near-black void
        return;
    }

    vec3  N      = normalize(texture(gNormal,  fragUV).xyz);
    vec4  albedoR= texture(gAlbedo,  fragUV);
    vec3  albedo = albedoR.rgb;
    float rough  = albedoR.a;

    vec3 V = normalize(ubo.viewPos.xyz - worldPos);

    // Ambient — very dim so multiple coloured lights really pop
    vec3 result = albedo * 0.025;

    // ---- Accumulate all 16 point lights ---------------------------------
    for (int i = 0; i < 16; ++i)
    {
        vec3  toLight = ubo.lights[i].pos.xyz - worldPos;
        float dist    = length(toLight);
        float radius  = ubo.lights[i].pos.w;

        if (dist >= radius) continue; // outside light radius — skip

        vec3  L    = toLight / dist;
        vec3  H    = normalize(L + V);

        // Smooth quadratic falloff: 1 → 0 over the radius
        float t     = 1.0 - dist / radius;
        float atten = t * t;

        float diff  = max(dot(N, L), 0.0);
        // Specular sharpness inversely proportional to roughness
        float shine = max(2.0 / (rough * rough + 0.001) - 2.0, 1.0);
        float spec  = pow(max(dot(N, H), 0.0), shine);

        float intensity = ubo.lights[i].color.w;
        vec3  lcolor    = ubo.lights[i].color.xyz;

        result += atten * intensity * lcolor *
                  (diff * albedo + spec * (1.0 - rough) * 0.4);
    }

    // ---- Reinhard tone mapping + gamma ----------------------------------
    result   = result / (1.0 + result);           // per-channel Reinhard
    result   = pow(result, vec3(1.0 / 2.2));      // gamma correction
    outColor = vec4(result, 1.0);
}
