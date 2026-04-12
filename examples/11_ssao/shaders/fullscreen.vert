#version 450

// ---------------------------------------------------------------------------
// Fullscreen triangle vertex shader.
// Drives a single vkCmdDraw(3) with no vertex buffers.
// gl_VertexIndex = 0,1,2 generates a triangle that covers the entire screen.
// UV (0,0) = top-left, (1,1) = bottom-right (Vulkan convention).
// ---------------------------------------------------------------------------

layout(location = 0) out vec2 outUV;

void main()
{
    outUV       = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
