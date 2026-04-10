#version 450

// Generates a fullscreen triangle from gl_VertexIndex alone — no vertex buffer needed.
// Call vkCmdDraw(3, 1, 0, 0).

layout(location = 0) out vec2 fragUV;

void main()
{
    vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 pos  = positions[gl_VertexIndex];
    fragUV    = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
