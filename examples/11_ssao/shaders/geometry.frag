#version 450

// ---------------------------------------------------------------------------
// Geometry pass fragment shader — writes three G-buffer targets:
//   attachment 0 (position): view-space XYZ, a=1 to mark geometry present
//   attachment 1 (normal):   view-space normal XYZ (normalised)
//   attachment 2 (albedo):   rgb = diffuse colour, a = roughness [0..1]
//
// Background pixels keep a=0 in the position buffer; SSAO and lighting
// shaders use this as a "no geometry" sentinel.
// ---------------------------------------------------------------------------

layout(location = 0) in vec3 inViewPos;
layout(location = 1) in vec3 inViewNormal;

layout(set = 0, binding = 0) uniform MaterialUBO {
    vec4 albedo; // rgb = diffuse colour, a = roughness [0..1]
} mat;

layout(location = 0) out vec4 outPosition; // view-space position  (a = 1 = geometry)
layout(location = 1) out vec4 outNormal;   // view-space normal
layout(location = 2) out vec4 outAlbedo;   // rgb = colour, a = roughness

void main()
{
    outPosition = vec4(inViewPos, 1.0);
    outNormal   = vec4(normalize(inViewNormal), 0.0);
    outAlbedo   = mat.albedo;
}
