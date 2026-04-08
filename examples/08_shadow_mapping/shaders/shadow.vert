#version 450

// Shadow pass — vertex only, writes depth from the light's point of view.
// No fragment shader needed; the GPU writes gl_FragDepth automatically.

layout(push_constant) uniform PC {
    mat4 lightSpaceMVP;
} pc;

layout(location = 0) in vec3 inPosition;

void main()
{
    gl_Position = pc.lightSpaceMVP * vec4(inPosition, 1.0);
}
