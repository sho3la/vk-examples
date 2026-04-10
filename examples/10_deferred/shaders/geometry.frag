#version 450

// Geometry pass fragment shader.
// Writes three G-buffer render targets in a single pass (MRT):
//   location 0 — gPosition : world-space XYZ, A=1 (marks real geometry)
//   location 1 — gNormal   : world-space normal XYZ (A unused)
//   location 2 — gAlbedo   : RGB albedo + A roughness

layout(location = 0) in  vec3 fragWorldPos;
layout(location = 1) in  vec3 fragNormal;

layout(location = 0) out vec4 gPosition; // R16G16B16A16_SFLOAT
layout(location = 1) out vec4 gNormal;   // R16G16B16A16_SFLOAT
layout(location = 2) out vec4 gAlbedo;   // R8G8B8A8_UNORM  (A = roughness)

// Per-material data — switched per draw call via descriptor set
layout(set = 0, binding = 0) uniform MaterialUBO {
    vec4 albedo;   // rgb = diffuse colour,  a = roughness  [0..1]
} mat;

void main()
{
    gPosition = vec4(fragWorldPos, 1.0); // A=1 → real geometry (not background)
    gNormal   = vec4(normalize(fragNormal), 0.0);
    gAlbedo   = mat.albedo;
}
