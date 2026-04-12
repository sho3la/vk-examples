#version 450

// ---------------------------------------------------------------------------
// Lighting fragment shader — Blinn-Phong with SSAO ambient occlusion.
//
// All lighting is computed in view space:
//   • Camera is at the origin (0,0,0).
//   • Light positions are stored in view space in the UBO.
//   • G-buffer positions and normals are in view space.
//
// The blurred SSAO factor (0 = fully occluded, 1 = fully lit) scales the
// ambient term, darkening corners and crevices without affecting direct
// illumination.
// ---------------------------------------------------------------------------

layout(location = 0) in vec2 inUV;

layout(set = 0, binding = 0) uniform sampler2D gPosition; // view-space position
layout(set = 0, binding = 1) uniform sampler2D gNormal;   // view-space normal
layout(set = 0, binding = 2) uniform sampler2D gAlbedo;   // rgb=colour, a=roughness
layout(set = 0, binding = 3) uniform sampler2D ssaoBlur;  // blurred AO factor [0,1]

struct PointLight {
    vec4 pos;   // xyz = view-space position, w = radius
    vec4 color; // xyz = colour,              w = intensity
};

layout(set = 0, binding = 4) uniform LightingUBO {
    PointLight lights[16];
    vec4       viewPos; // camera world position (informational)
} ubo;

layout(location = 0) out vec4 outColor;

void main()
{
    // Background sentinel: gPosition.a == 0 means no geometry
    vec4 posData = texture(gPosition, inUV);
    if (posData.a < 0.5) {
        outColor = vec4(0.01, 0.01, 0.02, 1.0); // near-black background
        return;
    }

    vec3  fragPos  = posData.xyz;
    vec3  normal   = normalize(texture(gNormal, inUV).xyz);
    vec4  albData  = texture(gAlbedo, inUV);
    vec3  albedo   = albData.rgb;
    float roughness = albData.a;
    float ao        = texture(ssaoBlur, inUV).r;

    // In view space the camera sits at the origin
    vec3 viewDir = normalize(-fragPos);

    // Ambient term — SSAO modulates how much ambient reaches this point
    vec3 result = albedo * 0.08 * ao;

    // Accumulate 16 point lights (Blinn-Phong)
    for (int i = 0; i < 16; ++i)
    {
        vec3  lightPos    = ubo.lights[i].pos.xyz;
        float lightRadius = ubo.lights[i].pos.w;
        vec3  lightColor  = ubo.lights[i].color.xyz;
        float intensity   = ubo.lights[i].color.w;

        vec3  L    = lightPos - fragPos;
        float dist = length(L);
        if (dist > lightRadius) continue;

        L = normalize(L);
        float NdL  = max(dot(normal, L), 0.0);
        float att  = 1.0 - smoothstep(0.0, lightRadius, dist);

        // Specular (Blinn-Phong half-vector) — damped by roughness
        vec3  H     = normalize(L + viewDir);
        float shine = mix(64.0, 4.0, roughness); // rough = soft highlights
        float spec  = pow(max(dot(normal, H), 0.0), shine) * (1.0 - roughness);

        result += (albedo * NdL + vec3(spec)) * lightColor * intensity * att;
    }

    outColor = vec4(result, 1.0);
}
