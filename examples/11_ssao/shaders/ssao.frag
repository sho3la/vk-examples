#version 450

// ---------------------------------------------------------------------------
// SSAO fragment shader.
//
// Algorithm:
//   1. Read view-space position and normal from G-buffers.
//   2. Build a tangent-space basis from the surface normal, randomised
//      by tiling a small 4x4 noise texture to reduce banding.
//   3. For each of 64 hemisphere samples (from the kernel UBO), project
//      the offset position back into screen space and compare its view-space
//      Z against the value stored in the position G-buffer.
//   4. Accumulate occlusion; output 1 - occlusion/64 as the AO factor.
//
// View-space convention (GLM right-handed, Y-up):
//   Objects in front of the camera have NEGATIVE Z.
//   "Closer to camera" means HIGHER Z (less negative).
//   A sample at samplePos is occluded when the geometry depth (sampleDepth)
//   is >= samplePos.z + bias (geometry is in front of the sample point).
// ---------------------------------------------------------------------------

layout(location = 0) in vec2 inUV;

// G-buffer inputs
layout(set = 0, binding = 0) uniform sampler2D gPosition; // view-space pos (a=1 = geometry)
layout(set = 0, binding = 1) uniform sampler2D gNormal;   // view-space normal

// 4x4 noise texture (REPEAT) — stores random xy rotation vectors in [0,1]
// decoded in shader: xy * 2 - 1 → [-1, 1]
layout(set = 0, binding = 2) uniform sampler2D texNoise;

// Hemisphere sample kernel (64 points, pre-generated on CPU)
layout(set = 0, binding = 3) uniform KernelUBO {
    vec4 samples[64]; // xyz = hemisphere direction+length, w = unused
} kernel;

layout(push_constant) uniform Push {
    mat4  proj;        // projection matrix (with Vulkan Y-flip)
    vec2  noiseScale;  // (screenW / 4, screenH / 4) — noise tile repeat count
    float radius;      // sampling hemisphere radius in view space
    float bias;        // depth bias to avoid self-occlusion artefacts
} push;

layout(location = 0) out float outOcclusion;

void main()
{
    // Background check — position alpha is 0 for sky, 1 for geometry
    vec4 posData = texture(gPosition, inUV);
    if (posData.a < 0.5) { outOcclusion = 1.0; return; }

    vec3 fragPos   = posData.xyz;
    vec3 normal    = normalize(texture(gNormal, inUV).xyz);

    // Random rotation vector from tiled noise (decoded from [0,1] → [-1,1])
    vec3 randomVec = normalize(vec3(texture(texNoise, inUV * push.noiseScale).rg * 2.0 - 1.0, 0.0));

    // Gram-Schmidt: build TBN aligned to surface normal, rotated by noise
    vec3 tangent   = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < 64; ++i)
    {
        // Rotate kernel sample into view space
        vec3 samplePos = TBN * kernel.samples[i].xyz;
        samplePos = fragPos + samplePos * push.radius;

        // Project sample to get its screen-space UV
        vec4 offset = push.proj * vec4(samplePos, 1.0);
        offset.xy  /= offset.w;
        offset.xy   = offset.xy * 0.5 + 0.5; // NDC → texture UV

        // Sample the stored view-space Z at the projected position
        float sampleDepth = texture(gPosition, offset.xy).z;

        // Range check: discard contribution from samples beyond hemisphere radius
        float rangeCheck = smoothstep(0.0, 1.0, push.radius / abs(fragPos.z - sampleDepth));

        // Occluded if geometry is in front of (closer to camera than) samplePos
        occlusion += (sampleDepth >= samplePos.z + push.bias ? 1.0 : 0.0) * rangeCheck;
    }

    outOcclusion = 1.0 - (occlusion / 64.0);
}
