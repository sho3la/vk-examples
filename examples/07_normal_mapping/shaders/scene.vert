#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inTangent;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out mat3 fragTBN; // tangent-space → world-space (uses locations 2, 3, 4)

void main()
{
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    fragWorldPos  = worldPos.xyz;
    fragUV        = inUV;

    // Build TBN in world space.
    // For uniform-scale models the normal matrix is just mat3(model).
    mat3 M = mat3(pc.model);
    vec3 T = normalize(M * inTangent);
    vec3 N = normalize(M * inNormal);
    vec3 B = cross(N, T);   // recompute bitangent — no need to store it per-vertex

    fragTBN = mat3(T, B, N);

    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
