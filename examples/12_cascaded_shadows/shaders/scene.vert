#version 450

// ---------------------------------------------------------------------------
// Scene vertex shader — outputs world-space position, normal and vertex colour.
//
// World-space outputs are used in the fragment shader for:
//   • Cascade selection (view-space depth computed from UBO view matrix).
//   • Shadow map lookup (light-space projection from UBO cascade matrices).
//   • Diffuse/specular shading (light direction and camera position from UBO).
//
// The geometry uses uniform scaling only, so mat3(model) is a valid
// normal matrix without the transpose-inverse.
// ---------------------------------------------------------------------------

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(push_constant) uniform PC {
    mat4 mvp;   // proj * view * model
    mat4 model; // model matrix
} pc;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outColor;

void main()
{
    gl_Position = pc.mvp * vec4(inPos, 1.0);
    outWorldPos = vec3(pc.model * vec4(inPos, 1.0));
    outNormal   = normalize(mat3(pc.model) * inNormal);
    outColor    = inColor;
}
