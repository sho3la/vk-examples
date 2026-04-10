#version 450

// Generates a full-screen covering triangle from three vertex indices — no vertex buffer needed.
// Call vkCmdDraw(cmd, 3, 1, 0, 0) to invoke this shader.
//
//  Index 0 → clip (-1,-1)  UV (0,0)  — top-left
//  Index 1 → clip ( 3,-1)  UV (2,0)  — far right  (clips to right edge)
//  Index 2 → clip (-1, 3)  UV (0,2)  — far bottom  (clips to bottom edge)
//
// The triangle covers the entire [-1,1]×[-1,1] viewport.
// UV (0,0)→(1,1) maps exactly to the visible screen area.

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
