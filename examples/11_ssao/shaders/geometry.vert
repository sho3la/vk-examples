#version 450

// ---------------------------------------------------------------------------
// Geometry pass vertex shader — outputs view-space position and normal.
// The vault geometry has no non-uniform scaling so mat3(modelView) is a
// valid normal matrix (no need for transpose-inverse).
// ---------------------------------------------------------------------------

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform PC {
    mat4 mvp;        // proj * view * model
    mat4 modelView;  // view * model  (for view-space outputs)
} pc;

layout(location = 0) out vec3 outViewPos;
layout(location = 1) out vec3 outViewNormal;

void main()
{
    gl_Position   = pc.mvp * vec4(inPos, 1.0);
    outViewPos    = vec3(pc.modelView * vec4(inPos, 1.0));
    outViewNormal = normalize(mat3(pc.modelView) * inNormal);
}
