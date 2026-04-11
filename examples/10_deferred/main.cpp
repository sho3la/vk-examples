
#include "../common/vk_common.h"
#include "../common/vk_pipelines.h"
#include "../common/vk_descriptors.h"

// ---------------------------------------------------------------------------
// 10 – Deferred Rendering
//
// Scene: "The Vault" — a stone chamber with 4 pillars and a raised central
// dais, lit by 16 coloured point lights (8 orbiting, 4 flickering torches,
// 4 cool ceiling accents).  The scene showcases the key strength of deferred
// shading: all 16 lights are evaluated in a single fullscreen pass regardless
// of scene complexity.
//
// Passes (all via Dynamic Rendering — no VkRenderPass objects):
//   Pass 1 – Geometry : renders scene geometry to 3 G-buffer images (MRT)
//   Pass 2 – Lighting : fullscreen pass reading G-buffers, applies all lights
//
// New Vulkan concepts:
//   • Multiple Render Targets (MRT): colorAttachmentCount = 3
//   • layout(location = N) out — multiple fragment outputs in one shader
//   • Per-material descriptor switching inside a render pass
//   • Two descriptor set layouts bound simultaneously in lighting pass
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU data layouts
// ---------------------------------------------------------------------------

struct MaterialData
{
    glm::vec4 albedo; // rgb = diffuse colour,  a = roughness [0..1]
};

struct PointLight
{
    glm::vec4 pos;   // xyz = world position,  w = radius
    glm::vec4 color; // xyz = colour,          w = intensity
};

struct LightingUBO
{
    PointLight lights[16];
    glm::vec4  viewPos;
};

struct GeomPush { glm::mat4 mvp; glm::mat4 model; }; // 128 B — vertex only

// ---------------------------------------------------------------------------
// CPU vertex
// ---------------------------------------------------------------------------

struct Vertex { glm::vec3 pos; glm::vec3 normal; glm::vec2 uv; };

// ---------------------------------------------------------------------------
// Scene geometry helpers
// ---------------------------------------------------------------------------

// Appends a quad (2 triangles) with a given normal.
// Vertices must be in CCW order as seen from the normal direction.
static void addQuad(std::vector<Vertex>& verts, std::vector<uint16_t>& idx,
    glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 n)
{
    auto base = static_cast<uint16_t>(verts.size());
    for (auto& v : {v0, v1, v2, v3})
        verts.push_back({v, n, {0.0f, 0.0f}});
    idx.insert(idx.end(), {base, uint16_t(base+1), uint16_t(base+2),
                            base, uint16_t(base+2), uint16_t(base+3)});
}

// Appends an axis-aligned box (all 6 faces, outward normals).
// Returns the index of the first index appended.
static uint32_t addBox(std::vector<Vertex>& verts, std::vector<uint16_t>& idx,
    glm::vec3 mn, glm::vec3 mx)
{
    uint32_t first = static_cast<uint32_t>(idx.size());
    float ax=mn.x, ay=mn.y, az=mn.z, bx=mx.x, by=mx.y, bz=mx.z;

    addQuad(verts,idx, {bx,by,az},{ax,by,az},{ax,by,bz},{bx,by,bz}, { 0, 1, 0}); // +Y
    addQuad(verts,idx, {ax,ay,az},{bx,ay,az},{bx,ay,bz},{ax,ay,bz}, { 0,-1, 0}); // -Y
    addQuad(verts,idx, {ax,ay,bz},{bx,ay,bz},{bx,by,bz},{ax,by,bz}, { 0, 0, 1}); // +Z
    addQuad(verts,idx, {bx,ay,az},{ax,ay,az},{ax,by,az},{bx,by,az}, { 0, 0,-1}); // -Z
    addQuad(verts,idx, {bx,ay,bz},{bx,ay,az},{bx,by,az},{bx,by,bz}, { 1, 0, 0}); // +X
    addQuad(verts,idx, {ax,ay,az},{ax,ay,bz},{ax,by,bz},{ax,by,az}, {-1, 0, 0}); // -X
    return first;
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

class DeferredApp : public VkAppBase
{
private:
    // == G-buffer formats ====================================================
    static constexpr VkFormat FMT_POSITION = VK_FORMAT_R16G16B16A16_SFLOAT;
    static constexpr VkFormat FMT_NORMAL   = VK_FORMAT_R16G16B16A16_SFLOAT;
    static constexpr VkFormat FMT_ALBEDO   = VK_FORMAT_R8G8B8A8_UNORM;

    // == Per-frame G-buffer ==================================================
    struct GBuffer {
        VkImage        position = VK_NULL_HANDLE,
                       normal   = VK_NULL_HANDLE,
                       albedo   = VK_NULL_HANDLE;
        VkDeviceMemory positionMem = VK_NULL_HANDLE,
                       normalMem   = VK_NULL_HANDLE,
                       albedoMem   = VK_NULL_HANDLE;
        VkImageView    positionView = VK_NULL_HANDLE,
                       normalView   = VK_NULL_HANDLE,
                       albedoView   = VK_NULL_HANDLE;
    };
    std::vector<GBuffer> gBuffers; // [MAX_FRAMES_IN_FLIGHT]

    VkSampler gbufferSampler = VK_NULL_HANDLE; // nearest-clamp — G-buffer readback

    // == Depth (shared, single, recreated on resize) ========================
    VkImage        depthImage     = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory    = VK_NULL_HANDLE;
    VkImageView    depthImageView = VK_NULL_HANDLE;
    const VkFormat depthFormat    = VK_FORMAT_D32_SFLOAT;

    // == Descriptor layouts ==================================================
    VkDescriptorSetLayout geomMatLayout  = VK_NULL_HANDLE; // binding 0: MaterialUBO
    VkDescriptorSetLayout gbufferLayout  = VK_NULL_HANDLE; // bindings 0,1,2: G-buffer CIS
    VkDescriptorSetLayout lightUBOLayout = VK_NULL_HANDLE; // binding 0: LightingUBO

    // == Descriptor pool + sets ==============================================
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    // Material sets — one per material, static (no per-frame duplication)
    static constexpr int MAT_COUNT = 5;
    std::vector<VkDescriptorSet> materialDescSets; // [MAT_COUNT]

    // Lighting sets — per frame
    std::vector<VkDescriptorSet> gbufferDescSets;  // [MAX_FRAMES] — reads 3 G-buffers
    std::vector<VkDescriptorSet> lightUBODescSets; // [MAX_FRAMES] — reads lighting UBO

    // == Pipelines ===========================================================
    VkPipelineLayout geomPipelineLayout    = VK_NULL_HANDLE;
    VkPipeline       geomPipeline          = VK_NULL_HANDLE;
    VkPipelineLayout lightPipelineLayout   = VK_NULL_HANDLE;
    VkPipeline       lightPipeline         = VK_NULL_HANDLE;

    // == Buffers =============================================================
    // Material UBOs — static, one per material
    std::vector<VkBuffer>       materialUBOs;
    std::vector<VkDeviceMemory> materialUBOMem;

    // Lighting UBOs — per frame
    std::vector<VkBuffer>       lightingUBOs;
    std::vector<VkDeviceMemory> lightingUBOMem;
    std::vector<void*>          lightingUBOMapped;

    // == Scene geometry ======================================================
    struct SceneObject {
        glm::mat4 transform;
        uint32_t  materialIndex;
        uint32_t  firstIndex;
        uint32_t  indexCount;
    };
    std::vector<SceneObject> sceneObjects;

    VkBuffer       sceneVB  = VK_NULL_HANDLE; VkDeviceMemory sceneVBMem  = VK_NULL_HANDLE;
    VkBuffer       sceneIB  = VK_NULL_HANDLE; VkDeviceMemory sceneIBMem  = VK_NULL_HANDLE;

    // =======================================================================
    // Barrier shorthand
    // =======================================================================

    void toColorWrite(VkCommandBuffer cmd, VkImage img)
    {
        transitionImageLayout(cmd, img,
            VK_IMAGE_LAYOUT_UNDEFINED,                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,             VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    }

    void colorToRead(VkCommandBuffer cmd, VkImage img)
    {
        transitionImageLayout(cmd, img,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,         VK_ACCESS_2_SHADER_READ_BIT);
    }

    // =======================================================================
    // G-buffer images
    // =======================================================================

    void createGBuffers()
    {
        constexpr VkImageUsageFlags usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        uint32_t w = swapChainExtent.width, h = swapChainExtent.height;
        gBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        for (auto& g : gBuffers)
        {
            createImage(w, h, FMT_POSITION, VK_IMAGE_TILING_OPTIMAL, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, g.position, g.positionMem);
            createImage(w, h, FMT_NORMAL,   VK_IMAGE_TILING_OPTIMAL, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, g.normal,   g.normalMem);
            createImage(w, h, FMT_ALBEDO,   VK_IMAGE_TILING_OPTIMAL, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, g.albedo,   g.albedoMem);
            g.positionView = createImageView(g.position, FMT_POSITION);
            g.normalView   = createImageView(g.normal,   FMT_NORMAL);
            g.albedoView   = createImageView(g.albedo,   FMT_ALBEDO);
        }
    }

    void destroyGBuffers()
    {
        for (auto& g : gBuffers)
        {
            vkDestroyImageView(device, g.albedoView,   nullptr);
            vkDestroyImageView(device, g.normalView,   nullptr);
            vkDestroyImageView(device, g.positionView, nullptr);
            vkDestroyImage(device, g.albedo,   nullptr); vkFreeMemory(device, g.albedoMem,   nullptr);
            vkDestroyImage(device, g.normal,   nullptr); vkFreeMemory(device, g.normalMem,   nullptr);
            vkDestroyImage(device, g.position, nullptr); vkFreeMemory(device, g.positionMem, nullptr);
        }
        gBuffers.clear();
    }

    // =======================================================================
    // Sampler
    // =======================================================================

    void createSamplers()
    {
        // Nearest-clamp for G-buffer reads — no interpolation between geometry boundaries
        gbufferSampler = createSampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    }

    // =======================================================================
    // Descriptor set layouts
    // =======================================================================

    void createDescriptorSetLayouts()
    {
        // Geometry pass — material UBO (fragment only)
        geomMatLayout = DescriptorLayoutBuilder()
            .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(device);

        // Lighting pass — 3 G-buffer CIS (fragment only)
        gbufferLayout = DescriptorLayoutBuilder()
            .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(device);

        // Lighting pass — lighting UBO (fragment only)
        lightUBOLayout = DescriptorLayoutBuilder()
            .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(device);
    }

    // =======================================================================
    // Descriptor pool + sets
    // =======================================================================

    void createDescriptorPool()
    {
        uint32_t n = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        // UBO: MAT_COUNT material UBOs (static) + n lighting UBOs (per-frame)
        // CIS: 3 G-buffer samplers per frame
        // sets: MAT_COUNT + n (gbuffer) + n (lightUBO)
        descriptorPool = DescriptorPoolBuilder()
            .addSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAT_COUNT + n)
            .addSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * n)
            .build(device, MAT_COUNT + 2 * n);
    }

    void allocateDescriptorSets()
    {
        uint32_t n = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        // Material sets (MAT_COUNT, all using geomMatLayout)
        {
            std::vector<VkDescriptorSetLayout> layouts(MAT_COUNT, geomMatLayout);
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = descriptorPool;
            ai.descriptorSetCount = MAT_COUNT;
            ai.pSetLayouts = layouts.data();
            materialDescSets.resize(MAT_COUNT);
            if (vkAllocateDescriptorSets(device, &ai, materialDescSets.data()) != VK_SUCCESS)
                throw std::runtime_error("failed to allocate material descriptor sets");
        }

        auto allocN = [&](VkDescriptorSetLayout layout, std::vector<VkDescriptorSet>& sets) {
            std::vector<VkDescriptorSetLayout> layouts(n, layout);
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = descriptorPool; ai.descriptorSetCount = n;
            ai.pSetLayouts = layouts.data();
            sets.resize(n);
            if (vkAllocateDescriptorSets(device, &ai, sets.data()) != VK_SUCCESS)
                throw std::runtime_error("failed to allocate descriptor sets");
        };

        allocN(gbufferLayout,  gbufferDescSets);
        allocN(lightUBOLayout, lightUBODescSets);
    }

    void updateDescriptorSets()
    {
        // Material UBOs (static — one per material, no per-frame difference)
        for (int i = 0; i < MAT_COUNT; ++i)
            DescriptorWriter()
                .writeBuffer(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    materialUBOs[i], 0, sizeof(MaterialData))
                .update(device, materialDescSets[i]);

        // G-buffer + lighting UBO (per frame)
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            DescriptorWriter()
                .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    gBuffers[i].positionView, gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    gBuffers[i].normalView,   gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeImage(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    gBuffers[i].albedoView,   gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .update(device, gbufferDescSets[i]);

            DescriptorWriter()
                .writeBuffer(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    lightingUBOs[i], 0, sizeof(LightingUBO))
                .update(device, lightUBODescSets[i]);
        }
    }

    // =======================================================================
    // Pipelines
    // =======================================================================

    void createGeometryPipeline()
    {
        VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/geometry.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/geometry.frag.spv"));

        // Three colour formats for MRT (position, normal, albedo) + depth
        auto [pipeline, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert)
            .fragShader(frag)
            .vertexBinding<Vertex>()
            .vertexAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
            .vertexAttribute(1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
            .vertexAttribute(2, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv))
            .pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(GeomPush))
            .descriptorSetLayout(geomMatLayout)
            .colorFormat(FMT_POSITION)
            .colorFormat(FMT_NORMAL)
            .colorFormat(FMT_ALBEDO)
            .depthFormat(depthFormat)
            .build();

        geomPipeline       = pipeline;
        geomPipelineLayout = layout;
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    void createLightingPipeline()
    {
        VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/fullscreen.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/lighting.frag.spv"));

        // Fullscreen triangle — no vertex input, no depth, no culling
        // Two descriptor set layouts: set=0 G-buffers, set=1 lighting UBO
        auto [pipeline, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert)
            .fragShader(frag)
            .noVertexInput()
            .noDepth()
            .cullMode(VK_CULL_MODE_NONE)
            .descriptorSetLayout(gbufferLayout)
            .descriptorSetLayout(lightUBOLayout)
            .colorFormat(swapChainImageFormat)
            .build();

        lightPipeline       = pipeline;
        lightPipelineLayout = layout;
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    // =======================================================================
    // Geometry upload
    // =======================================================================

    void createGeometry()
    {
        // The Vault — a stone chamber 18 × 6 × 18 m
        // Camera at (0, 3, 7) inside the room, looking toward (0, 2, 0)
        // Materials:
        //   0 = Floor   (warm stone)
        //   1 = Ceiling (dark concrete)
        //   2 = Wall    (sandstone)
        //   3 = Pillar  (dark granite)
        //   4 = Dais    (metallic slate)

        std::vector<Vertex>   verts;
        std::vector<uint16_t> idx;

        auto addObj = [&](uint32_t firstIdx, uint32_t count,
            uint32_t matIdx, glm::mat4 xform = glm::mat4(1.0f)) {
            sceneObjects.push_back({xform, matIdx, firstIdx, count});
        };

        constexpr float W = 9.0f; // half-width of the vault
        constexpr float H = 6.0f; // height

        // ---- Room faces ----
        // Floor (Y=0, normal +Y) — CCW from above: front-facing when viewed from inside
        {
            uint32_t fi = static_cast<uint32_t>(idx.size());
            addQuad(verts, idx, {W,0,-W},{-W,0,-W},{-W,0,W},{W,0,W}, {0,1,0});
            addObj(fi, 6, 0);
        }
        // Ceiling (Y=H, normal -Y) — CCW from below: front-facing when viewed from inside
        {
            uint32_t fi = static_cast<uint32_t>(idx.size());
            addQuad(verts, idx, {-W,H,-W},{W,H,-W},{W,H,W},{-W,H,W}, {0,-1,0});
            addObj(fi, 6, 1);
        }
        // Back wall (Z=-W, normal +Z) — visible from inside (camera at Z=+7)
        {
            uint32_t fi = static_cast<uint32_t>(idx.size());
            addQuad(verts, idx, {-W,0,-W},{W,0,-W},{W,H,-W},{-W,H,-W}, {0,0,1});
            addObj(fi, 6, 2);
        }
        // Front wall (Z=+W, normal -Z)
        {
            uint32_t fi = static_cast<uint32_t>(idx.size());
            addQuad(verts, idx, {W,0,W},{-W,0,W},{-W,H,W},{W,H,W}, {0,0,-1});
            addObj(fi, 6, 2);
        }
        // Left wall (X=-W, normal +X)
        {
            uint32_t fi = static_cast<uint32_t>(idx.size());
            addQuad(verts, idx, {-W,0,W},{-W,0,-W},{-W,H,-W},{-W,H,W}, {1,0,0});
            addObj(fi, 6, 2);
        }
        // Right wall (X=+W, normal -X)
        {
            uint32_t fi = static_cast<uint32_t>(idx.size());
            addQuad(verts, idx, {W,0,-W},{W,0,W},{W,H,W},{W,H,-W}, {-1,0,0});
            addObj(fi, 6, 2);
        }

        // ---- 4 pillars (1×5.5×1 m, corners at ±5, ±5) ----
        static const glm::vec2 pillarPos[4] = {{-5,-5},{5,-5},{5,5},{-5,5}};
        for (auto& p : pillarPos)
        {
            uint32_t fi = static_cast<uint32_t>(idx.size());
            addBox(verts, idx,
                {p.x - 0.5f, 0.0f, p.y - 0.5f},
                {p.x + 0.5f, 5.5f, p.y + 0.5f});
            addObj(fi, 36, 3);
        }

        // ---- Central raised dais (3×0.4×3 m) ----
        {
            uint32_t fi = static_cast<uint32_t>(idx.size());
            addBox(verts, idx, {-1.5f, 0.0f, -1.5f}, {1.5f, 0.4f, 1.5f});
            addObj(fi, 36, 4);
        }

        uploadStagedBuffer(verts.data(), verts.size() * sizeof(Vertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, sceneVB, sceneVBMem);
        uploadStagedBuffer(idx.data(),   idx.size() * sizeof(uint16_t),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,  sceneIB, sceneIBMem);
    }

    // =======================================================================
    // Material UBOs (static)
    // =======================================================================

    void createMaterialUBOs()
    {
        // Five materials used by the vault geometry
        static const MaterialData kMaterials[MAT_COUNT] = {
            {{0.55f, 0.52f, 0.48f, 0.92f}}, // 0: Floor   — warm stone, very rough
            {{0.14f, 0.14f, 0.17f, 0.97f}}, // 1: Ceiling  — near-black concrete
            {{0.47f, 0.42f, 0.37f, 0.88f}}, // 2: Wall     — sandstone
            {{0.24f, 0.21f, 0.19f, 0.85f}}, // 3: Pillar   — dark granite
            {{0.36f, 0.39f, 0.44f, 0.42f}}, // 4: Dais     — smooth metallic slate
        };

        materialUBOs.resize(MAT_COUNT);
        materialUBOMem.resize(MAT_COUNT);

        for (int i = 0; i < MAT_COUNT; ++i)
        {
            createBuffer(sizeof(MaterialData),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                materialUBOs[i], materialUBOMem[i]);
            void* mapped;
            vkMapMemory(device, materialUBOMem[i], 0, sizeof(MaterialData), 0, &mapped);
            std::memcpy(mapped, &kMaterials[i], sizeof(MaterialData));
            vkUnmapMemory(device, materialUBOMem[i]);
        }
    }

    // =======================================================================
    // Lighting UBOs (per frame)
    // =======================================================================

    void updateLightingUBO(uint32_t frameIndex)
    {
        float t = static_cast<float>(glfwGetTime());

        // Rainbow palette for the 8 orbiting lights
        static const glm::vec3 kOrbitColors[8] = {
            {1.0f, 0.15f, 0.15f}, // red
            {1.0f, 0.50f, 0.10f}, // orange
            {1.0f, 1.00f, 0.15f}, // yellow
            {0.15f,1.00f, 0.15f}, // green
            {0.10f,0.90f, 1.00f}, // cyan
            {0.15f,0.30f, 1.00f}, // blue
            {0.60f,0.10f, 1.00f}, // violet
            {1.00f,0.10f, 0.70f}, // magenta
        };

        // Pillar positions (base centres)
        static const glm::vec2 kPillarXZ[4] = {{-5,-5},{5,-5},{5,5},{-5,5}};

        LightingUBO ubo{};

        // Lights 0-7: orbit the vault at varying heights, rainbow-coloured
        for (int i = 0; i < 8; ++i)
        {
            float angle  = t * 0.35f + i * (glm::two_pi<float>() / 8.0f);
            float height = 1.8f + 1.5f * std::sin(t * 0.25f + i * 0.9f);
            glm::vec3 pos(6.0f * std::cos(angle), height, 6.0f * std::sin(angle));
            ubo.lights[i].pos   = glm::vec4(pos, 9.0f);          // radius = 9 m
            ubo.lights[i].color = glm::vec4(kOrbitColors[i], 3.0f); // intensity = 3
        }

        // Lights 8-11: warm flickering torches near each pillar
        for (int i = 0; i < 4; ++i)
        {
            float flicker = 0.80f + 0.20f * std::sin(t * 9.0f + i * 1.9f);
            glm::vec3 pos(kPillarXZ[i].x, 1.2f, kPillarXZ[i].y);
            ubo.lights[8+i].pos   = glm::vec4(pos, 4.5f);
            ubo.lights[8+i].color = glm::vec4(1.0f, 0.42f, 0.08f, 3.5f * flicker); // warm orange
        }

        // Lights 12-15: cool blue/purple ceiling accents (static)
        static const glm::vec3 kCeilPos[4] = {{-6,5.6f,-6},{6,5.6f,-6},{6,5.6f,6},{-6,5.6f,6}};
        for (int i = 0; i < 4; ++i)
        {
            ubo.lights[12+i].pos   = glm::vec4(kCeilPos[i], 7.0f);
            ubo.lights[12+i].color = glm::vec4(0.25f, 0.15f, 1.0f, 2.0f); // cool blue-violet
        }

        ubo.viewPos = glm::vec4(0.0f, 3.0f, 7.0f, 0.0f);
        std::memcpy(lightingUBOMapped[frameIndex], &ubo, sizeof(ubo));
    }

    // =======================================================================
    // VkAppBase hooks
    // =======================================================================

protected:
    void onInitBeforeCommandPool() override
    {
        createDepthResources(depthFormat, depthImage, depthMemory, depthImageView);
        createGBuffers();
        createSamplers();
        createDescriptorSetLayouts();
        createGeometryPipeline();
        createLightingPipeline();
    }

    void onInitAfterCommandPool() override
    {
        createGeometry();
        createMaterialUBOs();
        createPersistentUBOs(sizeof(LightingUBO), MAX_FRAMES_IN_FLIGHT,
            lightingUBOs, lightingUBOMem, lightingUBOMapped);
        createDescriptorPool();
        allocateDescriptorSets();
        updateDescriptorSets();
    }

    void onBeforeRecord(uint32_t frameIndex) override { updateLightingUBO(frameIndex); }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) override
    {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
            throw std::runtime_error("failed to begin command buffer");

        float    aspect = (float)swapChainExtent.width / (float)swapChainExtent.height;
        glm::mat4 view  = glm::lookAt(glm::vec3(0,3,7), glm::vec3(0,2,0), glm::vec3(0,1,0));
        glm::mat4 proj  = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
        proj[1][1] *= -1.0f;

        VkViewport vp{0,0,(float)swapChainExtent.width,(float)swapChainExtent.height,0,1};
        VkRect2D   sc{{0,0}, swapChainExtent};

        // =====================================================================
        // PASS 1 — Geometry  (writes 3 G-buffers + depth)
        // =====================================================================

        // Transition all three G-buffer images to colour attachment write
        auto& gb = gBuffers[currentFrame];
        toColorWrite(cmd, gb.position);
        toColorWrite(cmd, gb.normal);
        toColorWrite(cmd, gb.albedo);

        // Depth: UNDEFINED → DEPTH_ATTACHMENT
        transitionImageLayout(cmd, depthImage,
            VK_IMAGE_LAYOUT_UNDEFINED,                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,          VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        {
            // Three colour attachments — one per G-buffer, all cleared to zero
            // (gPosition.a == 0 in the lighting shader means background / no geometry)
            auto makeCA = [](VkImageView view) {
                VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                a.imageView   = view;
                a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                a.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
                a.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
                a.clearValue  = {};  // all zeros
                return a;
            };
            VkRenderingAttachmentInfo ca[3] = {
                makeCA(gb.positionView),
                makeCA(gb.normalView),
                makeCA(gb.albedoView),
            };

            VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            da.imageView   = depthImageView;
            da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            da.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
            da.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            da.clearValue.depthStencil = {1.0f, 0};

            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea             = {{0,0}, swapChainExtent};
            ri.layerCount             = 1;
            ri.colorAttachmentCount   = 3; // ← MRT
            ri.pColorAttachments      = ca;
            ri.pDepthAttachment       = &da;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geomPipeline);

            VkBuffer vbuf[] = {sceneVB}; VkDeviceSize offs[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, vbuf, offs);
            vkCmdBindIndexBuffer(cmd, sceneIB, 0, VK_INDEX_TYPE_UINT16);

            // Draw each scene object — switch material descriptor set per draw
            for (const auto& obj : sceneObjects)
            {
                // Bind material descriptor set for this object
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    geomPipelineLayout, 0, 1, &materialDescSets[obj.materialIndex], 0, nullptr);

                // Push mvp + model (128 B, vertex stage only)
                glm::mat4 mvp = proj * view * obj.transform;
                GeomPush push{mvp, obj.transform};
                vkCmdPushConstants(cmd, geomPipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);

                vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.firstIndex, 0, 0);
            }

            vkCmdEndRendering(cmd);
        }

        // =====================================================================
        // PASS 2 — Lighting  (fullscreen → swapchain)
        // =====================================================================

        // G-buffers: colour attachment → shader read
        colorToRead(cmd, gb.position);
        colorToRead(cmd, gb.normal);
        colorToRead(cmd, gb.albedo);

        // Swapchain: UNDEFINED → colour attachment
        transitionImageLayout(cmd, swapChainImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED,                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,             VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        {
            VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            att.imageView   = swapChainImageViews[imageIndex];
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            att.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea           = {{0,0}, swapChainExtent};
            ri.layerCount           = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments    = &att;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lightPipeline);

            // Bind set=0 (G-buffers) and set=1 (lighting UBO) together
            VkDescriptorSet sets[2] = {gbufferDescSets[currentFrame], lightUBODescSets[currentFrame]};
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                lightPipelineLayout, 0, 2, sets, 0, nullptr);

            vkCmdDraw(cmd, 3, 1, 0, 0); // fullscreen triangle
            vkCmdEndRendering(cmd);
        }

        // Swapchain: colour attachment → present
        transitionImageLayout(cmd, swapChainImages[imageIndex],
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,          VK_ACCESS_2_NONE);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
            throw std::runtime_error("failed to record command buffer");
    }

    void onCleanupSwapChain() override
    {
        destroyDepthResources(depthImage, depthMemory, depthImageView);
        destroyGBuffers();
    }

    void onRecreateSwapChain() override
    {
        createDepthResources(depthFormat, depthImage, depthMemory, depthImageView);
        createGBuffers();
        updateDescriptorSets(); // re-point image views after recreation
    }

    void onCleanup() override
    {
        destroyUBOs(lightingUBOs, lightingUBOMem);
        for (int i = 0; i < MAT_COUNT; ++i) {
            vkDestroyBuffer(device, materialUBOs[i], nullptr);
            vkFreeMemory(device, materialUBOMem[i], nullptr);
        }

        vkDestroyBuffer(device, sceneIB, nullptr); vkFreeMemory(device, sceneIBMem, nullptr);
        vkDestroyBuffer(device, sceneVB, nullptr); vkFreeMemory(device, sceneVBMem, nullptr);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, lightUBOLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, gbufferLayout,  nullptr);
        vkDestroyDescriptorSetLayout(device, geomMatLayout,  nullptr);

        vkDestroyPipeline(device,       lightPipeline,        nullptr);
        vkDestroyPipelineLayout(device, lightPipelineLayout,  nullptr);
        vkDestroyPipeline(device,       geomPipeline,         nullptr);
        vkDestroyPipelineLayout(device, geomPipelineLayout,   nullptr);

        vkDestroySampler(device, gbufferSampler, nullptr);
    }
};

// ---------------------------------------------------------------------------

int main()
{
    DeferredApp app;
    try {
        app.run(1280, 720, "10 – Deferred Rendering");
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }
}
