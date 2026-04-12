// ===========================================================================
// Example 12 — Cascaded Shadow Maps (CSM)
//
// Technique overview
// ------------------
// A directional "sun" light illuminates the scene.  The view frustum is
// partitioned into 4 sub-frusta (cascades) using the practical split-scheme
// (blend of logarithmic and uniform splits).  Each cascade has its own
// 2048×2048 shadow map stored as one layer of a Vulkan 2-D texture array.
//
// Per frame:
//   Pass 1 (×4) — Shadow passes
//       Render scene geometry depth-only into shadow map array layers 0-3
//       using the per-cascade orthographic light-space VP matrix.
//   Pass 2 — Scene pass
//       Forward Blinn-Phong shading with PCF 3×3 shadow lookups.
//       The fragment shader selects the appropriate cascade by view-space
//       depth and samples sampler2DArrayShadow for hardware comparison.
//
// Cascade stabilisation
// ---------------------
// The orthographic AABB is snapped to shadow-map texel boundaries so the
// shadow pattern does not shimmer as the camera or light rotates.
//
// Light direction
// ---------------
// The sun orbits the scene (yaw animated each frame).  lightDir is the
// direction FROM the surface TOWARD the light (i.e. "toward sun").
//
// Image layout sequence
// ---------------------
//   Shadow image — 4 layers:
//     UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL  (before all shadow passes)
//     DEPTH_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL (after all passes)
//   Scene depth:
//     UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL
//   Swapchain:
//     UNDEFINED → COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR
// ===========================================================================

#include "../common/vk_common.h"
#include "../common/vk_pipelines.h"
#include "../common/vk_descriptors.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>

#include <array>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

// ===========================================================================
// Constants
// ===========================================================================

static constexpr int   NUM_CASCADES    = 4;
static constexpr int   SHADOW_MAP_SIZE = 2048;
static constexpr float NEAR_PLANE      = 0.1f;
static constexpr float FAR_PLANE       = 120.0f;
static constexpr float FOV_Y           = 60.0f;  // degrees
static constexpr float SPLIT_LAMBDA    = 0.75f;  // log/linear blend weight

// ===========================================================================
// Data structures
// ===========================================================================

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
};

// Push constant for the shadow pass (vertex stage only, 64 bytes).
struct ShadowPush {
    glm::mat4 lightSpaceMVP;
};

// Push constant for the scene pass (vertex stage only, 128 bytes).
struct ScenePush {
    glm::mat4 mvp;
    glm::mat4 model;
};

// Per-frame uniform buffer consumed by scene.frag.
struct CascadeUBO {
    glm::mat4 lightSpaceMat[4]; // 256 bytes
    glm::vec4 splitDepths;      //  16 bytes — positive view-space far planes
    glm::vec4 lightDir;         //  16 bytes — toward-light world direction
    glm::vec4 lightColor;       //  16 bytes — rgb colour + intensity
    glm::vec4 viewPos;          //  16 bytes — camera world position
    glm::mat4 view;             //  64 bytes — view matrix for cascade selection
};                              // 384 bytes total

// One draw call entry — geometry range + model transform.
struct DrawCall {
    uint32_t  firstIndex;
    uint32_t  indexCount;
    glm::mat4 transform;
};

// ===========================================================================
// Application
// ===========================================================================

class CsmApp : public VkAppBase
{
    // -----------------------------------------------------------------------
    // Shadow map resources (fixed size, not swapchain-sized)
    // -----------------------------------------------------------------------
    VkImage        shadowImage                      = VK_NULL_HANDLE;
    VkDeviceMemory shadowMemory                     = VK_NULL_HANDLE;
    VkImageView    shadowArrayView                  = VK_NULL_HANDLE; // sampled view (all layers)
    VkImageView    shadowLayerViews[NUM_CASCADES]   = {};             // per-layer depth attachment
    VkSampler      shadowSampler                    = VK_NULL_HANDLE;

    // -----------------------------------------------------------------------
    // Scene depth (swapchain-sized)
    // -----------------------------------------------------------------------
    VkImage        depthImage  = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView    depthView   = VK_NULL_HANDLE;
    VkFormat       depthFormat = VK_FORMAT_D32_SFLOAT;

    // -----------------------------------------------------------------------
    // Pipelines
    // -----------------------------------------------------------------------
    VkPipeline       shadowPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;

    VkPipeline       scenePipeline       = VK_NULL_HANDLE;
    VkPipelineLayout scenePipelineLayout = VK_NULL_HANDLE;

    // -----------------------------------------------------------------------
    // Descriptors
    // -----------------------------------------------------------------------
    VkDescriptorSetLayout sceneLayout    = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet       sceneDescSets[MAX_FRAMES_IN_FLIGHT] = {};

    // -----------------------------------------------------------------------
    // Uniform buffers (per frame-in-flight)
    // -----------------------------------------------------------------------
    std::vector<VkBuffer>       cascadeUBOs;
    std::vector<VkDeviceMemory> cascadeUBOMem;
    std::vector<void*>          cascadeUBOMapped;

    // -----------------------------------------------------------------------
    // Geometry
    // -----------------------------------------------------------------------
    VkBuffer       sceneVB    = VK_NULL_HANDLE;
    VkDeviceMemory sceneVBMem = VK_NULL_HANDLE;
    VkBuffer       sceneIB    = VK_NULL_HANDLE;
    VkDeviceMemory sceneIBMem = VK_NULL_HANDLE;
    std::vector<DrawCall> draws;

    // -----------------------------------------------------------------------
    // Per-frame state (updated in onBeforeRecord)
    // -----------------------------------------------------------------------
    glm::mat4 cascadeMats[NUM_CASCADES]{};
    float     splitDepthValues[NUM_CASCADES]{};
    glm::mat4 viewMatrix{};
    glm::mat4 projMatrix{};
    glm::vec3 cameraEye{};
    glm::vec3 lightDirWorld{};

    float cameraYaw = 0.0f;  // degrees, animated
    float lightYaw  = 45.0f; // degrees, animated
    float lightPitch = 52.0f; // degrees above horizon (fixed)

    // -----------------------------------------------------------------------
    // Shadow map creation
    // -----------------------------------------------------------------------

    void createShadowMap()
    {
        // One 2-D image with NUM_CASCADES array layers.
        VkImageCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType     = VK_IMAGE_TYPE_2D;
        ci.format        = VK_FORMAT_D32_SFLOAT;
        ci.extent        = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
        ci.mipLevels     = 1;
        ci.arrayLayers   = NUM_CASCADES;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ci.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_SAMPLED_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &ci, nullptr, &shadowImage) != VK_SUCCESS)
            throw std::runtime_error("failed to create shadow map image!");

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, shadowImage, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &ai, nullptr, &shadowMemory) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate shadow map memory!");
        vkBindImageMemory(device, shadowImage, shadowMemory, 0);

        // Per-layer 2-D views used as depth attachments during shadow passes.
        for (int i = 0; i < NUM_CASCADES; ++i)
        {
            VkImageViewCreateInfo vci{};
            vci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image            = shadowImage;
            vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
            vci.format           = VK_FORMAT_D32_SFLOAT;
            vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1,
                                    static_cast<uint32_t>(i), 1};
            if (vkCreateImageView(device, &vci, nullptr, &shadowLayerViews[i]) != VK_SUCCESS)
                throw std::runtime_error("failed to create shadow layer view!");
        }

        // 2-D array view used by the scene fragment shader.
        VkImageViewCreateInfo vci{};
        vci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image            = shadowImage;
        vci.viewType         = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format           = VK_FORMAT_D32_SFLOAT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0,
                                static_cast<uint32_t>(NUM_CASCADES)};
        if (vkCreateImageView(device, &vci, nullptr, &shadowArrayView) != VK_SUCCESS)
            throw std::runtime_error("failed to create shadow array view!");

        // Comparison sampler — border = OPAQUE_WHITE = depth 1.0 (fully lit).
        shadowSampler = createSampler(VK_FILTER_LINEAR,
                                      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                      /*enableCompare=*/true,
                                      VK_COMPARE_OP_LESS_OR_EQUAL);
    }

    void destroyShadowMap()
    {
        vkDestroySampler   (device, shadowSampler,   nullptr);
        vkDestroyImageView (device, shadowArrayView, nullptr);
        for (int i = 0; i < NUM_CASCADES; ++i)
            vkDestroyImageView(device, shadowLayerViews[i], nullptr);
        vkDestroyImage     (device, shadowImage,   nullptr);
        vkFreeMemory       (device, shadowMemory,  nullptr);
    }

    // -----------------------------------------------------------------------
    // Descriptor layout
    // -----------------------------------------------------------------------

    void createDescriptorSetLayout()
    {
        sceneLayout = DescriptorLayoutBuilder()
            // binding 0: shadow map array (comparison sampler)
            .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        VK_SHADER_STAGE_FRAGMENT_BIT)
            // binding 1: cascade UBO (light matrices, split depths, etc.)
            .addBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(device);
    }

    // -----------------------------------------------------------------------
    // Pipelines
    // -----------------------------------------------------------------------

    void createShadowPipeline()
    {
        auto vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/shadow.vert.spv"));

        // Depth-only: no colour output, no fragment shader.
        auto [pip, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert)
            .vertexBinding<Vertex>()
            .vertexAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
            .pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(ShadowPush))
            .noColorOutput()
            .depthFormat(VK_FORMAT_D32_SFLOAT)
            // Front-face culling reduces Peter Panning for thick geometry.
            .cullMode(VK_CULL_MODE_FRONT_BIT)
            .build();

        shadowPipeline       = pip;
        shadowPipelineLayout = layout;

        vkDestroyShaderModule(device, vert, nullptr);
    }

    void createScenePipeline()
    {
        auto vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/scene.vert.spv"));
        auto frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/scene.frag.spv"));

        auto [pip, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert)
            .fragShader(frag)
            .vertexBinding<Vertex>()
            .vertexAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
            .vertexAttribute(1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
            .vertexAttribute(2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
            .pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(ScenePush))
            .descriptorSetLayout(sceneLayout)
            .colorFormat(swapChainImageFormat)
            .depthFormat(depthFormat)
            .build();

        scenePipeline       = pip;
        scenePipelineLayout = layout;

        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    // -----------------------------------------------------------------------
    // Descriptor pool and sets
    // -----------------------------------------------------------------------

    void createDescriptorPool()
    {
        // 2 sets × (1 CIS + 1 UBO)
        int n = MAX_FRAMES_IN_FLIGHT;
        descriptorPool = DescriptorPoolBuilder()
            .addSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, n)
            .addSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         n)
            .build(device, n);
    }

    void allocateDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, sceneLayout);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descriptorPool;
        ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        ai.pSetLayouts        = layouts.data();
        if (vkAllocateDescriptorSets(device, &ai, sceneDescSets) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate descriptor sets!");
    }

    void updateDescriptorSets()
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            DescriptorWriter()
                .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            shadowArrayView, shadowSampler,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeBuffer(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                             cascadeUBOs[i], 0, sizeof(CascadeUBO))
                .update(device, sceneDescSets[i]);
        }
    }

    // -----------------------------------------------------------------------
    // Geometry helpers
    // -----------------------------------------------------------------------

    static void addBox(std::vector<Vertex>& verts, std::vector<uint32_t>& inds,
                       glm::vec3 center, glm::vec3 half, glm::vec3 color)
    {
        // 6 faces; each face = 2 triangles, 4 vertices.
        struct Face { glm::vec3 n; glm::vec3 u; glm::vec3 v; float du; float dv; };
        const Face faces[] = {
            // normal       u-axis        v-axis          half-u   half-v
            { { 0,  1, 0}, { 1, 0, 0}, { 0, 0,-1}, half.x, half.z },  // +Y top
            { { 0, -1, 0}, { 1, 0, 0}, { 0, 0, 1}, half.x, half.z },  // -Y bottom
            { { 1,  0, 0}, { 0, 0,-1}, { 0, 1, 0}, half.z, half.y },  // +X right
            { {-1,  0, 0}, { 0, 0, 1}, { 0, 1, 0}, half.z, half.y },  // -X left
            { { 0,  0, 1}, { 1, 0, 0}, { 0, 1, 0}, half.x, half.y },  // +Z front
            { { 0,  0,-1}, {-1, 0, 0}, { 0, 1, 0}, half.x, half.y },  // -Z back
        };

        for (auto& f : faces)
        {
            uint32_t base = static_cast<uint32_t>(verts.size());
            glm::vec3 c = center + f.n * (f.n.x * half.x + f.n.y * half.y + f.n.z * half.z);
            verts.push_back({c - f.u * f.du - f.v * f.dv, f.n, color});
            verts.push_back({c + f.u * f.du - f.v * f.dv, f.n, color});
            verts.push_back({c + f.u * f.du + f.v * f.dv, f.n, color});
            verts.push_back({c - f.u * f.du + f.v * f.dv, f.n, color});
            inds.insert(inds.end(), {base,base+1,base+2, base,base+2,base+3});
        }
    }

    void createGeometry()
    {
        std::vector<Vertex>   verts;
        std::vector<uint32_t> inds;

        auto addDraw = [&](glm::mat4 transform = glm::mat4(1.0f))
        {
            // The last-added box becomes one draw call.
            // firstIndex / indexCount are tracked per call.
            (void)transform; // transform stored in draws.back().transform
        };
        (void)addDraw;

        auto push = [&](glm::vec3 center, glm::vec3 half, glm::vec3 color,
                        glm::mat4 xform = glm::mat4(1.0f))
        {
            uint32_t first = static_cast<uint32_t>(inds.size());
            addBox(verts, inds, center, half, color);
            uint32_t count = static_cast<uint32_t>(inds.size()) - first;
            draws.push_back({first, count, xform});
        };

        // Ground plane — large, warm grey.
        push({0, -0.15f, 0}, {52, 0.15f, 52}, {0.45f, 0.44f, 0.40f});

        // Central tower.
        push({0,  5.0f, 0}, {2.5f, 5.0f, 2.5f}, {0.65f, 0.70f, 0.80f});

        // Corner pillars.
        for (float sx : {-12.0f, 12.0f})
            for (float sz : {-12.0f, 12.0f})
                push({sx, 3.5f, sz}, {0.6f, 3.5f, 0.6f}, {0.72f, 0.65f, 0.52f});

        // Row of archway blocks along +Z axis.
        for (int i = -3; i <= 3; ++i)
            push({static_cast<float>(i) * 5.0f, 1.2f, 20.0f}, {1.0f, 1.2f, 1.0f},
                 {0.60f, 0.58f, 0.55f});

        // Scattered crates — near cascade.
        push({ 4,  0.5f,  5}, {0.5f, 0.5f, 0.5f}, {0.80f, 0.62f, 0.40f});
        push({-3,  0.8f,  7}, {0.8f, 0.8f, 0.8f}, {0.75f, 0.60f, 0.35f});
        push({ 6,  0.3f, -4}, {0.3f, 0.3f, 0.3f}, {0.85f, 0.70f, 0.45f});

        // Mid-range wall segment.
        push({-20, 2.0f, 10}, {8.0f, 2.0f, 0.5f}, {0.68f, 0.62f, 0.55f});
        push({ 22, 1.5f,-15}, {0.5f, 1.5f, 6.0f}, {0.65f, 0.60f, 0.52f});

        // Far objects (cascade 3) — low pyramid-like steps.
        push({-35, 0.5f,-35}, {5.0f, 0.5f, 5.0f}, {0.55f, 0.50f, 0.45f});
        push({-35, 1.5f,-35}, {3.5f, 1.0f, 3.5f}, {0.60f, 0.55f, 0.50f});
        push({-35, 2.5f,-35}, {2.0f, 1.0f, 2.0f}, {0.65f, 0.60f, 0.55f});
        push({ 38, 1.0f, 38}, {4.0f, 1.0f, 4.0f}, {0.58f, 0.52f, 0.48f});

        uploadStagedBuffer(verts.data(), sizeof(Vertex) * verts.size(),
                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, sceneVB, sceneVBMem);
        uploadStagedBuffer(inds.data(),  sizeof(uint32_t) * inds.size(),
                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT,  sceneIB, sceneIBMem);
    }

    // -----------------------------------------------------------------------
    // Cascade computation
    // -----------------------------------------------------------------------

    // Return the 8 world-space corners of the sub-frustum [nearZ, farZ].
    std::array<glm::vec3, 8> frustumCornersWorldSpace(float nearZ, float farZ) const
    {
        float aspect = static_cast<float>(swapChainExtent.width) /
                       static_cast<float>(swapChainExtent.height);
        glm::mat4 proj = glm::perspective(glm::radians(FOV_Y), aspect, nearZ, farZ);
        proj[1][1] *= -1; // Vulkan Y-flip
        glm::mat4 invVP = glm::inverse(proj * viewMatrix);

        // NDC cube corners — Vulkan depth [0, 1].
        std::array<glm::vec3, 8> corners;
        int idx = 0;
        for (float x : {-1.0f, 1.0f})
            for (float y : {-1.0f, 1.0f})
                for (float z : {0.0f, 1.0f})
                {
                    glm::vec4 p = invVP * glm::vec4(x, y, z, 1.0f);
                    corners[idx++] = glm::vec3(p) / p.w;
                }
        return corners;
    }

    // Compute the orthographic light-space VP that tightly fits the sub-frustum.
    glm::mat4 computeLightSpaceMatrix(float nearZ, float farZ) const
    {
        auto corners = frustumCornersWorldSpace(nearZ, farZ);

        // Frustum centre — light looks at this point.
        glm::vec3 center(0.0f);
        for (auto& c : corners) center += c;
        center /= 8.0f;

        // Bounding sphere radius — used to keep a stable, fixed-size AABB.
        float radius = 0.0f;
        for (auto& c : corners)
            radius = std::max(radius, glm::length(c - center));
        radius = std::ceil(radius * 16.0f) / 16.0f; // snap to 1/16 world unit

        // Light view: eye is 100 units from center along the light direction.
        glm::vec3 up = (std::abs(lightDirWorld.y) > 0.99f)
                     ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
        glm::mat4 lightView = glm::lookAt(center + lightDirWorld * 100.0f,
                                          center, up);

        // Compute AABB of frustum corners in light view space.
        float minX =  FLT_MAX, maxX = -FLT_MAX;
        float minY =  FLT_MAX, maxY = -FLT_MAX;
        float minZ =  FLT_MAX, maxZ = -FLT_MAX;
        for (auto& c : corners)
        {
            glm::vec3 lc = glm::vec3(lightView * glm::vec4(c, 1.0f));
            minX = std::min(minX, lc.x); maxX = std::max(maxX, lc.x);
            minY = std::min(minY, lc.y); maxY = std::max(maxY, lc.y);
            minZ = std::min(minZ, lc.z); maxZ = std::max(maxZ, lc.z);
        }

        // Snap XY bounds to texel grid — eliminates shadow shimmering.
        float texelW = (maxX - minX) / SHADOW_MAP_SIZE;
        float texelH = (maxY - minY) / SHADOW_MAP_SIZE;
        minX = std::floor(minX / texelW) * texelW;
        minY = std::floor(minY / texelH) * texelH;
        maxX = minX + std::ceil((2.0f * radius) / texelW) * texelW;
        maxY = minY + std::ceil((2.0f * radius) / texelH) * texelH;

        // Extend near plane back to capture shadow casters behind the frustum.
        // In GLM view space objects in front have negative Z; glm::ortho takes
        // positive near/far distances where -near < z < -far in view space.
        float zRange = maxZ - minZ;
        float near   = std::max(0.1f, -maxZ - zRange * 0.5f);
        float far    = -minZ + zRange * 0.1f;

        glm::mat4 lightProj = glm::ortho(minX, maxX, minY, maxY, near, far);
        lightProj[1][1] *= -1; // Vulkan Y-flip
        return lightProj * lightView;
    }

    void computeCascades()
    {
        float aspect = static_cast<float>(swapChainExtent.width) /
                       static_cast<float>(swapChainExtent.height);

        // Camera orbit.
        float cr = 28.0f;
        cameraEye = glm::vec3(cr * std::sin(glm::radians(cameraYaw)),
                              12.0f,
                              cr * std::cos(glm::radians(cameraYaw)));
        viewMatrix = glm::lookAt(cameraEye, glm::vec3(0, 1, 0), glm::vec3(0, 1, 0));
        projMatrix = glm::perspective(glm::radians(FOV_Y), aspect, NEAR_PLANE, FAR_PLANE);
        projMatrix[1][1] *= -1;

        // Sun direction (toward light, above horizon).
        float cy = glm::radians(lightYaw);
        float cp = glm::radians(lightPitch);
        lightDirWorld = glm::normalize(glm::vec3(
            std::cos(cp) * std::sin(cy),
            std::sin(cp),
            std::cos(cp) * std::cos(cy)));

        // PSSM practical split scheme (logarithmic/uniform blend).
        float range = FAR_PLANE - NEAR_PLANE;
        float ratio = FAR_PLANE / NEAR_PLANE;
        float prevSplit = NEAR_PLANE;
        for (int i = 0; i < NUM_CASCADES; ++i)
        {
            float p   = (i + 1) / static_cast<float>(NUM_CASCADES);
            float log = NEAR_PLANE * std::pow(ratio, p);
            float uni = NEAR_PLANE + range * p;
            float split = SPLIT_LAMBDA * log + (1.0f - SPLIT_LAMBDA) * uni;
            splitDepthValues[i] = split;
            cascadeMats[i]      = computeLightSpaceMatrix(prevSplit, split);
            prevSplit = split;
        }
    }

    // -----------------------------------------------------------------------
    // Virtual hooks
    // -----------------------------------------------------------------------

    void onInitBeforeCommandPool() override
    {
        // Shadow map is fixed-size — create once here.
        createShadowMap();

        // Scene depth is swapchain-sized.
        createDepthResources(depthFormat, depthImage, depthMemory, depthView);

        createDescriptorSetLayout();
        createShadowPipeline();
        createScenePipeline();
    }

    void onInitAfterCommandPool() override
    {
        createGeometry();
        createPersistentUBOs(sizeof(CascadeUBO), MAX_FRAMES_IN_FLIGHT,
                             cascadeUBOs, cascadeUBOMem, cascadeUBOMapped);
        createDescriptorPool();
        allocateDescriptorSets();
        updateDescriptorSets();
    }

    void onBeforeRecord(uint32_t frameIndex) override
    {
        // Animate camera orbit (0.015°/frame) and sun (0.04°/frame).
        cameraYaw += 0.015f;
        lightYaw  += 0.04f;

        computeCascades();

        CascadeUBO ubo{};
        for (int i = 0; i < NUM_CASCADES; ++i)
            ubo.lightSpaceMat[i] = cascadeMats[i];
        ubo.splitDepths = glm::vec4(splitDepthValues[0], splitDepthValues[1],
                                    splitDepthValues[2], splitDepthValues[3]);
        ubo.lightDir   = glm::vec4(lightDirWorld, 0.0f);
        ubo.lightColor = glm::vec4(1.00f, 0.95f, 0.85f, 1.8f); // warm sunlight
        ubo.viewPos    = glm::vec4(cameraEye, 1.0f);
        ubo.view       = viewMatrix;

        std::memcpy(cascadeUBOMapped[frameIndex], &ubo, sizeof(CascadeUBO));
    }

    // -----------------------------------------------------------------------
    // recordCommandBuffer
    // -----------------------------------------------------------------------

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) override
    {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
            throw std::runtime_error("failed to begin command buffer!");

        // Helper lambda — inline barrier for the shadow image (all 4 layers).
        auto shadowBarrier = [&](VkImageLayout oldLayout, VkImageLayout newLayout,
                                 VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                                 VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess)
        {
            VkImageMemoryBarrier2 b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask        = srcStage;
            b.srcAccessMask       = srcAccess;
            b.dstStageMask        = dstStage;
            b.dstAccessMask       = dstAccess;
            b.oldLayout           = oldLayout;
            b.newLayout           = newLayout;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = shadowImage;
            b.subresourceRange    = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0,
                                     static_cast<uint32_t>(NUM_CASCADES)};
            VkDependencyInfo dep{};
            dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        };

        // ==================================================================
        // Shadow passes (4 cascades)
        // ==================================================================

        // Transition all shadow layers to depth attachment.
        shadowBarrier(
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,      VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        VkBuffer     vertexBuffers[] = {sceneVB};
        VkDeviceSize offsets[]       = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer  (cmd, sceneIB, 0, VK_INDEX_TYPE_UINT32);

        for (int cascade = 0; cascade < NUM_CASCADES; ++cascade)
        {
            VkRenderingAttachmentInfo depthAtt{};
            depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAtt.imageView   = shadowLayerViews[cascade];
            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            depthAtt.clearValue.depthStencil = {1.0f, 0};

            VkRenderingInfo ri{};
            ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea           = {{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
            ri.layerCount           = 1;
            ri.pDepthAttachment     = &depthAtt;

            VkViewport vp{0, 0,
                static_cast<float>(SHADOW_MAP_SIZE), static_cast<float>(SHADOW_MAP_SIZE),
                0.0f, 1.0f};
            VkRect2D sc{{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor (cmd, 0, 1, &sc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);

            for (auto& d : draws)
            {
                ShadowPush push{};
                push.lightSpaceMVP = cascadeMats[cascade] * d.transform;
                vkCmdPushConstants(cmd, shadowPipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT,
                                   0, sizeof(ShadowPush), &push);
                vkCmdDrawIndexed(cmd, d.indexCount, 1, d.firstIndex, 0, 0);
            }

            vkCmdEndRendering(cmd);
        }

        // Transition shadow map to shader-readable.
        shadowBarrier(
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT);

        // ==================================================================
        // Scene pass — forward Blinn-Phong + CSM
        // ==================================================================

        transitionImageLayout(cmd, depthImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,         VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        transitionImageLayout(cmd, swapChainImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,          VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        VkRenderingAttachmentInfo colorAtt{};
        colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAtt.imageView   = swapChainImageViews[imageIndex];
        colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.clearValue  = {{{0.01f, 0.01f, 0.02f, 1.0f}}};

        VkRenderingAttachmentInfo depthAtt{};
        depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAtt.imageView   = depthView;
        depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo ri{};
        ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea           = {{0, 0}, swapChainExtent};
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments    = &colorAtt;
        ri.pDepthAttachment     = &depthAtt;

        VkViewport vp{0, 0,
            static_cast<float>(swapChainExtent.width),
            static_cast<float>(swapChainExtent.height),
            0.0f, 1.0f};
        VkRect2D sc{{0, 0}, swapChainExtent};

        vkCmdBeginRendering(cmd, &ri);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor (cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                scenePipelineLayout, 0, 1,
                                &sceneDescSets[currentFrame], 0, nullptr);

        for (auto& d : draws)
        {
            ScenePush push{};
            push.mvp   = projMatrix * viewMatrix * d.transform;
            push.model = d.transform;
            vkCmdPushConstants(cmd, scenePipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(ScenePush), &push);
            vkCmdDrawIndexed(cmd, d.indexCount, 1, d.firstIndex, 0, 0);
        }

        vkCmdEndRendering(cmd);

        transitionImageLayout(cmd, swapChainImages[imageIndex],
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
            throw std::runtime_error("failed to record command buffer!");
    }

    // -----------------------------------------------------------------------
    // Swap-chain lifecycle hooks
    // -----------------------------------------------------------------------

    void onCleanupSwapChain() override
    {
        // Scene depth is swapchain-sized — destroy on resize.
        destroyDepthResources(depthImage, depthMemory, depthView);
    }

    void onRecreateSwapChain() override
    {
        createDepthResources(depthFormat, depthImage, depthMemory, depthView);
        // Shadow map and descriptor sets do not change on resize.
    }

    void onCleanup() override
    {
        destroyUBOs(cascadeUBOs, cascadeUBOMem);

        vkDestroyBuffer    (device, sceneIB,    nullptr);
        vkFreeMemory       (device, sceneIBMem, nullptr);
        vkDestroyBuffer    (device, sceneVB,    nullptr);
        vkFreeMemory       (device, sceneVBMem, nullptr);

        vkDestroyDescriptorPool      (device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout (device, sceneLayout,    nullptr);

        vkDestroyPipeline      (device, scenePipeline,        nullptr);
        vkDestroyPipelineLayout(device, scenePipelineLayout,  nullptr);
        vkDestroyPipeline      (device, shadowPipeline,       nullptr);
        vkDestroyPipelineLayout(device, shadowPipelineLayout, nullptr);

        destroyShadowMap();
    }
};

// ===========================================================================

int main()
{
    CsmApp app;
    try {
        app.run(1280, 720, "12 – Cascaded Shadow Maps");
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }
}
