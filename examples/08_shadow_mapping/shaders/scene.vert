#version 450

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
} pc;

// UBO is also read in the vertex stage to pre-multiply the light-space position,
// avoiding an extra matrix multiply per fragment.
layout(set = 0, binding = 0) uniform SceneUBO {
    vec4 lightDir;          // xyz = direction FROM surface TO light (normalized)
    vec4 lightColor;        // xyz = color,  w = intensity
    vec4 viewPos;           // xyz = camera world position
    mat4 lightSpaceMatrix;  // orthographic projection × light view
    vec4 ambient;           // xyz = color,  w = strength
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;
layout(location = 3) out vec4 fragLightSpacePos; // homogeneous light-space coordinate

void main()
{
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    fragWorldPos  = worldPos.xyz;
    
    // Correct normal matrix
    mat3 normalMatrix = transpose(inverse(mat3(pc.model)));
    vec3 worldNormal  = normalize(normalMatrix * inNormal);
    fragNormal        = worldNormal;
    
    fragUV = inUV;

    // --- NORMAL OFFSET BIAS ---
    vec3 L = normalize(ubo.lightDir.xyz);
    
    // Calculate an offset that scales up at grazing light angles
    // The 0.05 is a tunable factor based on your scene scale
    float offsetScale = 0.05 * max(1.0 - dot(worldNormal, L), 0.0);
    
    // Shift the sampling position slightly outward along the normal
    vec3 biasedPos = worldPos.xyz + worldNormal * offsetScale;
    
    // Use the biased position to calculate the light-space coordinates
    fragLightSpacePos = ubo.lightSpaceMatrix * vec4(biasedPos, 1.0);

    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
