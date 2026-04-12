#version 450

// ---------------------------------------------------------------------------
// Scene fragment shader — Blinn-Phong with Cascaded Shadow Maps (CSM).
//
// Algorithm:
//   1. Determine which of the 4 cascades this fragment belongs to by
//      computing its view-space depth and comparing against the split table.
//   2. Transform the world-space position into the selected cascade's
//      light-space clip coordinates.
//   3. Sample the shadow map array with a 3×3 PCF kernel using a hardware
//      comparison sampler (sampler2DArrayShadow) for smooth shadow edges.
//   4. Apply Blinn-Phong shading with the PCF result as a shadow factor.
//
// The scene is lit by a single directional light (sun).
//
// A subtle per-cascade tint (red/green/blue/yellow) is blended in at 8%
// opacity so cascade boundaries are visible during development.
// ---------------------------------------------------------------------------

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

// Comparison sampler — VkSampler must have compareEnable = VK_TRUE,
// compareOp = VK_COMPARE_OP_LESS_OR_EQUAL, borderColor = OPAQUE_WHITE.
layout(set = 0, binding = 0) uniform sampler2DArrayShadow shadowMap;

layout(set = 0, binding = 1) uniform CascadeUBO {
    mat4  lightSpaceMat[4]; // per-cascade light-space VP matrices
    vec4  splitDepths;      // positive view-space distances: cascade 0-3 far planes
    vec4  lightDir;         // world-space direction TOWARD the light (normalised)
    vec4  lightColor;       // xyz = colour, w = intensity
    vec4  viewPos;          // camera world position
    mat4  view;             // view matrix (for cascade selection)
} ubo;

layout(location = 0) out vec4 outColor;

// ---------------------------------------------------------------------------
// Cascade selection
// ---------------------------------------------------------------------------
uint selectCascade()
{
    // Compute positive view-space depth (distance from camera plane).
    // GLM right-handed: objects in front have negative view-space Z.
    float depth = abs((ubo.view * vec4(inWorldPos, 1.0)).z);

    if (depth < ubo.splitDepths.x) return 0u;
    if (depth < ubo.splitDepths.y) return 1u;
    if (depth < ubo.splitDepths.z) return 2u;
    return 3u;
}

// ---------------------------------------------------------------------------
// 3x3 PCF shadow lookup using a hardware comparison sampler
// ---------------------------------------------------------------------------
float shadowFactor(uint cascade)
{
    vec4 lsPos = ubo.lightSpaceMat[cascade] * vec4(inWorldPos, 1.0);
    lsPos.xyz /= lsPos.w;                   // perspective divide (no-op for ortho)
    lsPos.xy   = lsPos.xy * 0.5 + 0.5;     // NDC [-1,1] → UV [0,1]

    // Fragments outside the shadow map boundary are fully lit.
    if (any(lessThan   (lsPos.xy, vec2(0.0))) ||
        any(greaterThan(lsPos.xy, vec2(1.0))) ||
        lsPos.z < 0.0 || lsPos.z > 1.0)
        return 1.0;

    // Slope-scaled bias — reduces self-shadowing on surfaces angled away from
    // the light; larger (farther) cascades cover wider areas so need more bias.
    float cosAngle = max(dot(inNormal, ubo.lightDir.xyz), 0.0);
    float bias = mix(0.003, 0.0005, cosAngle) * (1.0 + float(cascade) * 0.5);
    float cmpDepth = lsPos.z - bias;

    // 3×3 PCF — texture() on sampler2DArrayShadow returns hardware-filtered
    // comparison result in [0,1]: 1.0 = fully lit, 0.0 = fully in shadow.
    float shadow = 0.0;
    vec2  texel  = 1.0 / vec2(textureSize(shadowMap, 0));
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
            shadow += texture(shadowMap,
                vec4(lsPos.xy + vec2(x, y) * texel, float(cascade), cmpDepth));
    return shadow / 9.0;
}

void main()
{
    vec3 N = normalize(inNormal);
    vec3 L = normalize(ubo.lightDir.xyz);
    vec3 V = normalize(ubo.viewPos.xyz - inWorldPos);
    vec3 H = normalize(L + V);

    float NdL = max(dot(N, L), 0.0);

    vec3 ambient  = inColor * 0.12;
    vec3 diffuse  = inColor * NdL * ubo.lightColor.xyz * ubo.lightColor.w;
    float spec    = pow(max(dot(N, H), 0.0), 32.0) * NdL;
    vec3 specular = vec3(spec) * ubo.lightColor.xyz * ubo.lightColor.w * 0.25;

    uint  cascade = selectCascade();
    float shadow  = shadowFactor(cascade);

    vec3 result = ambient + (diffuse + specular) * shadow;

    // Subtle cascade debug tint (8% blend) — comment out to disable.
    vec3 tints[4] = vec3[](
        vec3(1.0, 0.3, 0.3),   // cascade 0 — red   (nearest)
        vec3(0.3, 1.0, 0.3),   // cascade 1 — green
        vec3(0.3, 0.5, 1.0),   // cascade 2 — blue
        vec3(1.0, 1.0, 0.3)    // cascade 3 — yellow (farthest)
    );
    result = mix(result, result * tints[cascade], 0.08);

    outColor = vec4(result, 1.0);
}
