#version 450

// ---------------------------------------------------------------------------
// Shadow pass vertex shader — depth-only.
//
// Renders scene geometry into one layer of the shadow map array from the
// directional light's perspective.  The light-space MVP (= lightSpaceVP *
// model) is supplied as a push constant so each draw call can set it with a
// single vkCmdPushConstants call.
//
// No fragment shader is used — only the depth buffer is written.
// ---------------------------------------------------------------------------

layout(location = 0) in vec3 inPos;

layout(push_constant) uniform PC {
    mat4 lightSpaceMVP; // lightSpaceVP * model  (64 bytes)
} pc;

void main()
{
    gl_Position = pc.lightSpaceMVP * vec4(inPos, 1.0);
}
