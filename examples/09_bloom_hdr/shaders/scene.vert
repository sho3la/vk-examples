#version 450

// Transforms vertices into world space and passes lighting data to the fragment stage.
// All lighting math is done per-fragment; the vertex stage just interpolates positions.

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;

void main()
{
    vec4 worldPos  = pc.model * vec4(inPosition, 1.0);
    fragWorldPos   = worldPos.xyz;
    fragNormal     = normalize(transpose(inverse(mat3(pc.model))) * inNormal);
    fragUV         = inUV;
    gl_Position    = pc.mvp * vec4(inPosition, 1.0);
}
