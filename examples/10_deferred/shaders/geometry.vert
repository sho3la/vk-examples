#version 450

// Geometry pass — writes world-space position and normal to G-buffer.
// The MVP is pre-multiplied on the CPU; the model matrix is passed separately
// so the fragment shader can compute correct world-space quantities.

layout(push_constant) uniform PC {
    mat4 mvp;   // proj * view * model — for gl_Position
    mat4 model; // model matrix — for world-space position and normal
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;     // not used in geometry pass but keeps stride correct

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;

void main()
{
    vec4 worldPos   = pc.model * vec4(inPosition, 1.0);
    fragWorldPos    = worldPos.xyz;

    // Normal matrix: handles non-uniform scale correctly
    mat3 normalMat  = transpose(inverse(mat3(pc.model)));
    fragNormal      = normalize(normalMat * inNormal);

    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
