
#include "../common/vk_common.h"
#include "../common/vk_pipelines.h"
#include "../common/vk_descriptors.h"
#include <random>

// ---------------------------------------------------------------------------
// 11 – Screen-Space Ambient Occlusion (SSAO)
//
// Scene: "The Vault" (same geometry as example 10) — a stone chamber with
// pillars and a raised dais, perfect for showcasing corner darkening.
//
// Passes (Dynamic Rendering, no VkRenderPass):
//   Pass 1 – Geometry : MRT → 3 G-buffers (view-space pos/normal/albedo) + depth
//   Pass 2 – SSAO     : fullscreen → raw occlusion image (R8_UNORM)
//   Pass 3 – SSAO Blur: fullscreen 4×4 box blur → smoothed occlusion image
//   Pass 4 – Lighting : fullscreen Blinn-Phong + SSAO ambient → swapchain
//
// New Vulkan / rendering concepts:
//   • Four-pass deferred pipeline chained with image layout transitions
//   • SSAO hemisphere kernel (64 samples, uploaded as static UBO)
//   • Tiled 4×4 rotation-noise texture (R8G8B8A8_UNORM, VK_SAMPLER_ADDRESS_MODE_REPEAT)
//   • View-space G-buffers (camera at origin → no viewPos lookup in shader)
//   • Push constants carrying projection matrix + SSAO parameters
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU data structures (must match shader layout exactly)
// ---------------------------------------------------------------------------

struct MaterialData
{
    glm::vec4 albedo; // rgb = diffuse colour, a = roughness [0..1]
};

struct PointLight
{
    glm::vec4 pos;   // xyz = view-space position, w = radius
    glm::vec4 color; // xyz = colour,              w = intensity
};

struct LightingUBO
{
    PointLight lights[16];
    glm::vec4  viewPos; // camera world position (informational)
};

struct SsaoKernelUBO
{
    glm::vec4 samples[64]; // xyz = hemisphere sample, w = unused
};

struct GeomPush
{
    glm::mat4 mvp;       // proj * view * model
    glm::mat4 modelView; // view * model  (for view-space output)
};

struct SsaoPush
{
    glm::mat4 proj;       // projection matrix (Vulkan Y-flipped)
    glm::vec2 noiseScale; // (screenW / 4, screenH / 4) — noise tile repeat
    float     radius;     // hemisphere sampling radius in view space
    float     bias;       // depth bias to suppress self-occlusion
};

// ---------------------------------------------------------------------------
// CPU vertex
// ---------------------------------------------------------------------------

struct Vertex { glm::vec3 pos; glm::vec3 normal; glm::vec2 uv; };

// ---------------------------------------------------------------------------
// Scene geometry helpers (identical to example 10)
// ---------------------------------------------------------------------------

static void addQuad(std::vector<Vertex>& verts, std::vector<uint16_t>& idx,
    glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 n)
{
    auto base = static_cast<uint16_t>(verts.size());
    for (auto& v : {v0, v1, v2, v3})
        verts.push_back({v, n, {0.0f, 0.0f}});
    idx.insert(idx.end(), {base, uint16_t(base+1), uint16_t(base+2),
                            base, uint16_t(base+2), uint16_t(base+3)});
}

static uint32_t addBox(std::vector<Vertex>& verts, std::vector<uint16_t>& idx,
    glm::vec3 mn, glm::vec3 mx)
{
    uint32_t first = static_cast<uint32_t>(idx.size());
    float ax=mn.x, ay=mn.y, az=mn.z, bx=mx.x, by=mx.y, bz=mx.z;
    addQuad(verts,idx, {bx,by,az},{ax,by,az},{ax,by,bz},{bx,by,bz}, { 0, 1, 0});
    addQuad(verts,idx, {ax,ay,az},{bx,ay,az},{bx,ay,bz},{ax,ay,bz}, { 0,-1, 0});
    addQuad(verts,idx, {ax,ay,bz},{bx,ay,bz},{bx,by,bz},{ax,by,bz}, { 0, 0, 1});
    addQuad(verts,idx, {bx,ay,az},{ax,ay,az},{ax,by,az},{bx,by,az}, { 0, 0,-1});
    addQuad(verts,idx, {bx,ay,bz},{bx,ay,az},{bx,by,az},{bx,by,bz}, { 1, 0, 0});
    addQuad(verts,idx, {ax,ay,az},{ax,ay,bz},{ax,by,bz},{ax,by,az}, {-1, 0, 0});
    return first;
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

class SsaoApp : public VkAppBase
{
private:
    // == Image formats ========================================================
    static constexpr VkFormat FMT_POS    = VK_FORMAT_R16G16B16A16_SFLOAT; // view-space position
    static constexpr VkFormat FMT_NORMAL = VK_FORMAT_R16G16B16A16_SFLOAT; // view-space normal
    static constexpr VkFormat FMT_ALBEDO = VK_FORMAT_R8G8B8A8_UNORM;      // diffuse + roughness
    static constexpr VkFormat FMT_SSAO   = VK_FORMAT_R8_UNORM;            // occlusion factor

    // == Per-frame G-buffer + SSAO images (recreated on swapchain resize) ====
    struct FrameData {
        VkImage        pos    = VK_NULL_HANDLE,
                       normal = VK_NULL_HANDLE,
                       albedo = VK_NULL_HANDLE;
        VkDeviceMemory posMem    = VK_NULL_HANDLE,
                       normalMem = VK_NULL_HANDLE,
                       albedoMem = VK_NULL_HANDLE;
        VkImageView    posView    = VK_NULL_HANDLE,
                       normalView = VK_NULL_HANDLE,
                       albedoView = VK_NULL_HANDLE;

        VkImage        ssao     = VK_NULL_HANDLE,
                       ssaoBlur = VK_NULL_HANDLE;
        VkDeviceMemory ssaoMem     = VK_NULL_HANDLE,
                       ssaoBlurMem = VK_NULL_HANDLE;
        VkImageView    ssaoView     = VK_NULL_HANDLE,
                       ssaoBlurView = VK_NULL_HANDLE;
    };
    std::vector<FrameData> frames;

    // == Depth (single image, recreated on resize) ============================
    VkImage        depthImage     = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory    = VK_NULL_HANDLE;
    VkImageView    depthImageView = VK_NULL_HANDLE;
    const VkFormat depthFormat    = VK_FORMAT_D32_SFLOAT;

    // == Noise texture (static 4x4, R8G8B8A8_UNORM, tiled via REPEAT) ========
    VkImage        noiseImage  = VK_NULL_HANDLE;
    VkDeviceMemory noiseMemory = VK_NULL_HANDLE;
    VkImageView    noiseView   = VK_NULL_HANDLE;
    VkSampler      noiseSampler = VK_NULL_HANDLE;

    // == Samplers =============================================================
    VkSampler gbufferSampler = VK_NULL_HANDLE; // nearest-clamp for G-buffers + SSAO

    // == Descriptor layouts ===================================================
    VkDescriptorSetLayout geomMatLayout  = VK_NULL_HANDLE; // geometry: material UBO
    VkDescriptorSetLayout ssaoLayout     = VK_NULL_HANDLE; // ssao: pos+normal+noise CIS + kernel UBO
    VkDescriptorSetLayout ssaoBlurLayout = VK_NULL_HANDLE; // blur: ssao CIS
    VkDescriptorSetLayout lightingLayout = VK_NULL_HANDLE; // lighting: 4 CIS + lighting UBO

    // == Descriptor pool + sets ===============================================
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    static constexpr int MAT_COUNT = 5;
    std::vector<VkDescriptorSet> materialDescSets; // [MAT_COUNT]   static
    std::vector<VkDescriptorSet> ssaoDescSets;     // [MAX_FRAMES]  per-frame
    std::vector<VkDescriptorSet> ssaoBlurDescSets; // [MAX_FRAMES]  per-frame
    std::vector<VkDescriptorSet> lightingDescSets; // [MAX_FRAMES]  per-frame

    // == Pipelines ============================================================
    VkPipelineLayout geomPipelineLayout     = VK_NULL_HANDLE;
    VkPipeline       geomPipeline           = VK_NULL_HANDLE;
    VkPipelineLayout ssaoPipelineLayout     = VK_NULL_HANDLE;
    VkPipeline       ssaoPipeline           = VK_NULL_HANDLE;
    VkPipelineLayout ssaoBlurPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       ssaoBlurPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout lightingPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       lightingPipeline       = VK_NULL_HANDLE;

    // == Buffers ==============================================================
    std::vector<VkBuffer>       materialUBOs;   // [MAT_COUNT]  static
    std::vector<VkDeviceMemory> materialUBOMem;

    VkBuffer       ssaoKernelUBO = VK_NULL_HANDLE; // static — 64 hemisphere samples
    VkDeviceMemory ssaoKernelMem = VK_NULL_HANDLE;

    std::vector<VkBuffer>       lightingUBOs;  // [MAX_FRAMES]
    std::vector<VkDeviceMemory> lightingUBOMem;
    std::vector<void*>          lightingUBOMapped;

    // == Scene geometry =======================================================
    struct SceneObject {
        glm::mat4 transform;
        uint32_t  materialIndex;
        uint32_t  firstIndex;
        uint32_t  indexCount;
    };
    std::vector<SceneObject> sceneObjects;
    VkBuffer       sceneVB = VK_NULL_HANDLE; VkDeviceMemory sceneVBMem = VK_NULL_HANDLE;
    VkBuffer       sceneIB = VK_NULL_HANDLE; VkDeviceMemory sceneIBMem = VK_NULL_HANDLE;

    // =========================================================================
    // Barrier shorthands (same transitions used repeatedly across 4 passes)
    // =========================================================================

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
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,          VK_ACCESS_2_SHADER_READ_BIT);
    }

    // =========================================================================
    // Per-frame G-buffer + SSAO images
    // =========================================================================

    void createFrameImages()
    {
        constexpr VkImageUsageFlags colorUsage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        uint32_t w = swapChainExtent.width, h = swapChainExtent.height;
        frames.resize(MAX_FRAMES_IN_FLIGHT);
        for (auto& f : frames)
        {
            createImage(w, h, FMT_POS,    VK_IMAGE_TILING_OPTIMAL, colorUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.pos,     f.posMem);
            createImage(w, h, FMT_NORMAL, VK_IMAGE_TILING_OPTIMAL, colorUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.normal,  f.normalMem);
            createImage(w, h, FMT_ALBEDO, VK_IMAGE_TILING_OPTIMAL, colorUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.albedo,  f.albedoMem);
            createImage(w, h, FMT_SSAO,   VK_IMAGE_TILING_OPTIMAL, colorUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.ssao,    f.ssaoMem);
            createImage(w, h, FMT_SSAO,   VK_IMAGE_TILING_OPTIMAL, colorUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.ssaoBlur, f.ssaoBlurMem);

            f.posView      = createImageView(f.pos,     FMT_POS);
            f.normalView   = createImageView(f.normal,  FMT_NORMAL);
            f.albedoView   = createImageView(f.albedo,  FMT_ALBEDO);
            f.ssaoView     = createImageView(f.ssao,    FMT_SSAO);
            f.ssaoBlurView = createImageView(f.ssaoBlur, FMT_SSAO);
        }
    }

    void destroyFrameImages()
    {
        for (auto& f : frames)
        {
            vkDestroyImageView(device, f.ssaoBlurView, nullptr);
            vkDestroyImageView(device, f.ssaoView,     nullptr);
            vkDestroyImageView(device, f.albedoView,   nullptr);
            vkDestroyImageView(device, f.normalView,   nullptr);
            vkDestroyImageView(device, f.posView,      nullptr);
            vkDestroyImage(device, f.ssaoBlur, nullptr); vkFreeMemory(device, f.ssaoBlurMem, nullptr);
            vkDestroyImage(device, f.ssao,     nullptr); vkFreeMemory(device, f.ssaoMem,     nullptr);
            vkDestroyImage(device, f.albedo,   nullptr); vkFreeMemory(device, f.albedoMem,   nullptr);
            vkDestroyImage(device, f.normal,   nullptr); vkFreeMemory(device, f.normalMem,   nullptr);
            vkDestroyImage(device, f.pos,      nullptr); vkFreeMemory(device, f.posMem,      nullptr);
        }
        frames.clear();
    }

    // =========================================================================
    // Noise texture — 4x4 R8G8B8A8_UNORM, random xy in rg, ba=0
    // Decoded in shader: rg * 2 - 1 → [-1, 1] rotation vectors
    // =========================================================================

    void createNoiseTexture()
    {
        constexpr uint32_t NOISE_DIM = 4;
        std::mt19937 rng(42); // fixed seed for reproducible noise
        std::uniform_int_distribution<uint32_t> dist(0, 255);

        struct Pixel { uint8_t r, g, b, a; };
        std::vector<Pixel> pixels(NOISE_DIM * NOISE_DIM);
        for (auto& p : pixels) { p.r = dist(rng); p.g = dist(rng); p.b = 0; p.a = 0; }

        VkDeviceSize sz = pixels.size() * sizeof(Pixel);

        // Staging buffer
        VkBuffer       stagingBuf; VkDeviceMemory stagingMem;
        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuf, stagingMem);
        void* mapped; vkMapMemory(device, stagingMem, 0, sz, 0, &mapped);
        std::memcpy(mapped, pixels.data(), sz);
        vkUnmapMemory(device, stagingMem);

        createImage(NOISE_DIM, NOISE_DIM, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, noiseImage, noiseMemory);

        auto cmd = beginOneTimeCommands();
        transitionImageLayout(cmd, noiseImage,
            VK_IMAGE_LAYOUT_UNDEFINED,       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        endOneTimeCommands(cmd);

        copyBufferToImage(stagingBuf, noiseImage, NOISE_DIM, NOISE_DIM);

        cmd = beginOneTimeCommands();
        transitionImageLayout(cmd, noiseImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,      VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        endOneTimeCommands(cmd);

        vkDestroyBuffer(device, stagingBuf, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);

        noiseView   = createImageView(noiseImage, VK_FORMAT_R8G8B8A8_UNORM);
        noiseSampler = createSampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    }

    // =========================================================================
    // SSAO kernel — 64 hemisphere samples, accelerating distribution
    // =========================================================================

    void createSsaoKernel()
    {
        std::mt19937 rng(123); // fixed seed
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        SsaoKernelUBO kernel{};
        for (int i = 0; i < 64; ++i)
        {
            glm::vec3 sample(
                dist(rng) * 2.0f - 1.0f,
                dist(rng) * 2.0f - 1.0f,
                dist(rng)                  // Z in [0,1] → upper hemisphere
            );
            sample = glm::normalize(sample) * dist(rng);

            // Accelerating distribution: cluster samples closer to fragment
            float scale = float(i) / 64.0f;
            scale = glm::mix(0.1f, 1.0f, scale * scale);
            sample *= scale;

            kernel.samples[i] = glm::vec4(sample, 0.0f);
        }

        createBuffer(sizeof(SsaoKernelUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ssaoKernelUBO, ssaoKernelMem);
        void* mapped; vkMapMemory(device, ssaoKernelMem, 0, sizeof(SsaoKernelUBO), 0, &mapped);
        std::memcpy(mapped, &kernel, sizeof(kernel));
        vkUnmapMemory(device, ssaoKernelMem);
    }

    // =========================================================================
    // Samplers
    // =========================================================================

    void createSamplers()
    {
        // Nearest-clamp: G-buffers and SSAO images — no interpolation at edges
        gbufferSampler = createSampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        // noiseSampler created inside createNoiseTexture() with REPEAT
    }

    // =========================================================================
    // Descriptor set layouts
    // =========================================================================

    void createDescriptorSetLayouts()
    {
        // Geometry pass: one material UBO per draw call
        geomMatLayout = DescriptorLayoutBuilder()
            .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(device);

        // SSAO pass: pos G-buffer, normal G-buffer, noise texture, kernel UBO
        ssaoLayout = DescriptorLayoutBuilder()
            .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(device);

        // SSAO blur pass: raw SSAO image
        ssaoBlurLayout = DescriptorLayoutBuilder()
            .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(device);

        // Lighting pass: pos, normal, albedo, ssaoBlur G-buffers + lighting UBO
        lightingLayout = DescriptorLayoutBuilder()
            .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(device);
    }

    // =========================================================================
    // Descriptor pool
    // =========================================================================

    void createDescriptorPool()
    {
        uint32_t n = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        // UBO:  MAT_COUNT (material) + n (ssao kernel) + n (lighting) = MAT_COUNT + 2n
        // CIS:  n×3 (ssao: pos+normal+noise) + n×1 (ssaoBlur) + n×4 (lighting) = 8n
        // Sets: MAT_COUNT + n (ssao) + n (ssaoBlur) + n (lighting) = MAT_COUNT + 3n
        descriptorPool = DescriptorPoolBuilder()
            .addSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAT_COUNT + 2 * n)
            .addSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 * n)
            .build(device, MAT_COUNT + 3 * n);
    }

    // =========================================================================
    // Descriptor allocation
    // =========================================================================

    void allocateDescriptorSets()
    {
        uint32_t n = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        // Static material sets
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

        auto allocN = [&](VkDescriptorSetLayout layout, std::vector<VkDescriptorSet>& sets)
        {
            std::vector<VkDescriptorSetLayout> layouts(n, layout);
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = descriptorPool; ai.descriptorSetCount = n;
            ai.pSetLayouts = layouts.data();
            sets.resize(n);
            if (vkAllocateDescriptorSets(device, &ai, sets.data()) != VK_SUCCESS)
                throw std::runtime_error("failed to allocate descriptor sets");
        };

        allocN(ssaoLayout,     ssaoDescSets);
        allocN(ssaoBlurLayout, ssaoBlurDescSets);
        allocN(lightingLayout, lightingDescSets);
    }

    // =========================================================================
    // Descriptor writes — called on init and on swapchain recreate
    // =========================================================================

    void updateDescriptorSets()
    {
        // Material UBOs (static — never need to be re-written on resize)
        for (int i = 0; i < MAT_COUNT; ++i)
            DescriptorWriter()
                .writeBuffer(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    materialUBOs[i], 0, sizeof(MaterialData))
                .update(device, materialDescSets[i]);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            auto& f = frames[i];

            // SSAO set: pos G-buffer, normal G-buffer, noise, kernel UBO
            DescriptorWriter()
                .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    f.posView,    gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    f.normalView, gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeImage(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    noiseView,    noiseSampler,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeBuffer(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    ssaoKernelUBO, 0, sizeof(SsaoKernelUBO))
                .update(device, ssaoDescSets[i]);

            // SSAO blur set: raw SSAO image
            DescriptorWriter()
                .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    f.ssaoView, gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .update(device, ssaoBlurDescSets[i]);

            // Lighting set: pos, normal, albedo, ssaoBlur, lighting UBO
            DescriptorWriter()
                .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    f.posView,      gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    f.normalView,   gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeImage(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    f.albedoView,   gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeImage(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    f.ssaoBlurView, gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeBuffer(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    lightingUBOs[i], 0, sizeof(LightingUBO))
                .update(device, lightingDescSets[i]);
        }
    }

    // =========================================================================
    // Pipelines
    // =========================================================================

    void createGeomPipeline()
    {
        VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/geometry.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/geometry.frag.spv"));

        // MRT: 3 colour formats (pos, normal, albedo) + depth
        auto [pipeline, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert).fragShader(frag)
            .vertexBinding<Vertex>()
            .vertexAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
            .vertexAttribute(1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
            .pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(GeomPush))
            .descriptorSetLayout(geomMatLayout)
            .colorFormat(FMT_POS).colorFormat(FMT_NORMAL).colorFormat(FMT_ALBEDO)
            .depthFormat(depthFormat)
            .build();

        geomPipeline       = pipeline;
        geomPipelineLayout = layout;
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    void createSsaoPipeline()
    {
        VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/fullscreen.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/ssao.frag.spv"));

        // Fullscreen — no vertex input, no depth, push constants carry proj + SSAO params
        auto [pipeline, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert).fragShader(frag)
            .noVertexInput().noDepth().cullMode(VK_CULL_MODE_NONE)
            .pushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(SsaoPush))
            .descriptorSetLayout(ssaoLayout)
            .colorFormat(FMT_SSAO)
            .build();

        ssaoPipeline       = pipeline;
        ssaoPipelineLayout = layout;
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    void createSsaoBlurPipeline()
    {
        VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/fullscreen.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/ssao_blur.frag.spv"));

        auto [pipeline, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert).fragShader(frag)
            .noVertexInput().noDepth().cullMode(VK_CULL_MODE_NONE)
            .descriptorSetLayout(ssaoBlurLayout)
            .colorFormat(FMT_SSAO)
            .build();

        ssaoBlurPipeline       = pipeline;
        ssaoBlurPipelineLayout = layout;
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    void createLightingPipeline()
    {
        VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/fullscreen.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/lighting.frag.spv"));

        auto [pipeline, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert).fragShader(frag)
            .noVertexInput().noDepth().cullMode(VK_CULL_MODE_NONE)
            .descriptorSetLayout(lightingLayout)
            .colorFormat(swapChainImageFormat)
            .build();

        lightingPipeline       = pipeline;
        lightingPipelineLayout = layout;
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    // =========================================================================
    // Scene geometry — "The Vault" (identical to example 10)
    // =========================================================================

    void createGeometry()
    {
        std::vector<Vertex>   verts;
        std::vector<uint16_t> idx;

        auto addObj = [&](uint32_t fi, uint32_t count, uint32_t matIdx,
            glm::mat4 xform = glm::mat4(1.0f))
        {
            sceneObjects.push_back({xform, matIdx, fi, count});
        };

        constexpr float W = 9.0f, H = 6.0f;

        { uint32_t fi = (uint32_t)idx.size(); addQuad(verts,idx,{W,0,-W},{-W,0,-W},{-W,0,W},{W,0,W},{0,1,0});   addObj(fi,6,0); } // floor
        { uint32_t fi = (uint32_t)idx.size(); addQuad(verts,idx,{-W,H,-W},{W,H,-W},{W,H,W},{-W,H,W},{0,-1,0}); addObj(fi,6,1); } // ceiling
        { uint32_t fi = (uint32_t)idx.size(); addQuad(verts,idx,{-W,0,-W},{W,0,-W},{W,H,-W},{-W,H,-W},{0,0,1}); addObj(fi,6,2); } // back wall
        { uint32_t fi = (uint32_t)idx.size(); addQuad(verts,idx,{W,0,W},{-W,0,W},{-W,H,W},{W,H,W},{0,0,-1});   addObj(fi,6,2); } // front wall
        { uint32_t fi = (uint32_t)idx.size(); addQuad(verts,idx,{-W,0,W},{-W,0,-W},{-W,H,-W},{-W,H,W},{1,0,0}); addObj(fi,6,2); } // left wall
        { uint32_t fi = (uint32_t)idx.size(); addQuad(verts,idx,{W,0,-W},{W,0,W},{W,H,W},{W,H,-W},{-1,0,0});  addObj(fi,6,2); } // right wall

        static const glm::vec2 pillarPos[4] = {{-5,-5},{5,-5},{5,5},{-5,5}};
        for (auto& p : pillarPos)
        {
            uint32_t fi = (uint32_t)idx.size();
            addBox(verts, idx, {p.x-0.5f,0,p.y-0.5f}, {p.x+0.5f,5.5f,p.y+0.5f});
            addObj(fi, 36, 3);
        }

        { uint32_t fi = (uint32_t)idx.size(); addBox(verts,idx,{-1.5f,0,-1.5f},{1.5f,0.4f,1.5f}); addObj(fi,36,4); } // dais

        uploadStagedBuffer(verts.data(), verts.size() * sizeof(Vertex),   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, sceneVB, sceneVBMem);
        uploadStagedBuffer(idx.data(),   idx.size()   * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,  sceneIB, sceneIBMem);
    }

    // =========================================================================
    // Material UBOs (static — one per material)
    // =========================================================================

    void createMaterialUBOs()
    {
        static const MaterialData kMaterials[MAT_COUNT] = {
            {{0.55f, 0.52f, 0.48f, 0.92f}}, // floor   — warm stone,   very rough
            {{0.14f, 0.14f, 0.17f, 0.97f}}, // ceiling — dark concrete
            {{0.47f, 0.42f, 0.37f, 0.88f}}, // wall    — sandstone
            {{0.24f, 0.21f, 0.19f, 0.85f}}, // pillar  — dark granite
            {{0.36f, 0.39f, 0.44f, 0.42f}}, // dais    — smooth metallic slate
        };

        materialUBOs.resize(MAT_COUNT);
        materialUBOMem.resize(MAT_COUNT);
        for (int i = 0; i < MAT_COUNT; ++i)
        {
            createBuffer(sizeof(MaterialData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                materialUBOs[i], materialUBOMem[i]);
            void* mapped; vkMapMemory(device, materialUBOMem[i], 0, sizeof(MaterialData), 0, &mapped);
            std::memcpy(mapped, &kMaterials[i], sizeof(MaterialData));
            vkUnmapMemory(device, materialUBOMem[i]);
        }
    }

    // =========================================================================
    // Lighting UBO update (per frame) — view-space light positions
    // =========================================================================

    void updateLightingUBO(uint32_t frameIndex)
    {
        float t = static_cast<float>(glfwGetTime());
        glm::mat4 view = glm::lookAt(glm::vec3(0,3,7), glm::vec3(0,2,0), glm::vec3(0,1,0));

        // Rainbow palette (orbiting lights)
        static const glm::vec3 kOrbitColors[8] = {
            {1.0f,0.15f,0.15f},{1.0f,0.50f,0.10f},{1.0f,1.00f,0.15f},{0.15f,1.00f,0.15f},
            {0.10f,0.90f,1.00f},{0.15f,0.30f,1.00f},{0.60f,0.10f,1.00f},{1.00f,0.10f,0.70f},
        };
        static const glm::vec2 kPillarXZ[4] = {{-5,-5},{5,-5},{5,5},{-5,5}};

        LightingUBO ubo{};

        // Lights 0-7: orbit the vault
        for (int i = 0; i < 8; ++i)
        {
            float angle  = t * 0.35f + i * (glm::two_pi<float>() / 8.0f);
            float height = 1.8f + 1.5f * std::sin(t * 0.25f + i * 0.9f);
            glm::vec3 worldPos(6.0f * std::cos(angle), height, 6.0f * std::sin(angle));
            glm::vec3 viewPos = glm::vec3(view * glm::vec4(worldPos, 1.0f));
            ubo.lights[i].pos   = glm::vec4(viewPos, 9.0f); // w = radius
            ubo.lights[i].color = glm::vec4(kOrbitColors[i], 3.0f);
        }

        // Lights 8-11: flickering torches near pillars
        for (int i = 0; i < 4; ++i)
        {
            float flicker = 0.80f + 0.20f * std::sin(t * 9.0f + i * 1.9f);
            glm::vec3 worldPos(kPillarXZ[i].x, 1.2f, kPillarXZ[i].y);
            glm::vec3 viewPos = glm::vec3(view * glm::vec4(worldPos, 1.0f));
            ubo.lights[8+i].pos   = glm::vec4(viewPos, 4.5f);
            ubo.lights[8+i].color = glm::vec4(1.0f, 0.42f, 0.08f, 3.5f * flicker);
        }

        // Lights 12-15: cool ceiling accents
        static const glm::vec3 kCeilPos[4] = {{-6,5.6f,-6},{6,5.6f,-6},{6,5.6f,6},{-6,5.6f,6}};
        for (int i = 0; i < 4; ++i)
        {
            glm::vec3 viewPos = glm::vec3(view * glm::vec4(kCeilPos[i], 1.0f));
            ubo.lights[12+i].pos   = glm::vec4(viewPos, 7.0f);
            ubo.lights[12+i].color = glm::vec4(0.25f, 0.15f, 1.0f, 2.0f);
        }

        ubo.viewPos = glm::vec4(0.0f, 3.0f, 7.0f, 0.0f);
        std::memcpy(lightingUBOMapped[frameIndex], &ubo, sizeof(ubo));
    }

    // =========================================================================
    // VkAppBase hooks
    // =========================================================================

protected:
    void onInitBeforeCommandPool() override
    {
        createDepthResources(depthFormat, depthImage, depthMemory, depthImageView);
        createFrameImages();
        createSamplers();
        createDescriptorSetLayouts();
        createGeomPipeline();
        createSsaoPipeline();
        createSsaoBlurPipeline();
        createLightingPipeline();
    }

    void onInitAfterCommandPool() override
    {
        createNoiseTexture();
        createSsaoKernel();
        createGeometry();
        createMaterialUBOs();
        createPersistentUBOs(sizeof(LightingUBO), MAX_FRAMES_IN_FLIGHT,
            lightingUBOs, lightingUBOMem, lightingUBOMapped);
        createDescriptorPool();
        allocateDescriptorSets();
        updateDescriptorSets();
    }

    void onBeforeRecord(uint32_t frameIndex) override
    {
        updateLightingUBO(frameIndex);
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) override
    {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
            throw std::runtime_error("failed to begin command buffer");

        auto& f = frames[currentFrame];

        float     aspect = (float)swapChainExtent.width / (float)swapChainExtent.height;
        glm::mat4 view   = glm::lookAt(glm::vec3(0,3,7), glm::vec3(0,2,0), glm::vec3(0,1,0));
        glm::mat4 proj   = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
        proj[1][1] *= -1.0f;

        VkViewport vp{0, 0, (float)swapChainExtent.width, (float)swapChainExtent.height, 0, 1};
        VkRect2D   sc{{0, 0}, swapChainExtent};

        // =============================================================
        // PASS 1 – Geometry: 3 G-buffers (MRT) + depth
        // =============================================================

        toColorWrite(cmd, f.pos);
        toColorWrite(cmd, f.normal);
        toColorWrite(cmd, f.albedo);
        transitionImageLayout(cmd, depthImage,
            VK_IMAGE_LAYOUT_UNDEFINED,                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,          VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        {
            auto makeCA = [](VkImageView view)
            {
                VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                a.imageView = view; a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                a.clearValue = {}; // all zeros — position.a=0 marks background
                return a;
            };
            VkRenderingAttachmentInfo ca[3] = { makeCA(f.posView), makeCA(f.normalView), makeCA(f.albedoView) };

            VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            da.imageView = depthImageView; da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            da.clearValue.depthStencil = {1.0f, 0};

            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0,0}, swapChainExtent}; ri.layerCount = 1;
            ri.colorAttachmentCount = 3; ri.pColorAttachments = ca;
            ri.pDepthAttachment = &da;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geomPipeline);

            VkBuffer vbuf[] = {sceneVB}; VkDeviceSize offs[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, vbuf, offs);
            vkCmdBindIndexBuffer(cmd, sceneIB, 0, VK_INDEX_TYPE_UINT16);

            for (const auto& obj : sceneObjects)
            {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    geomPipelineLayout, 0, 1, &materialDescSets[obj.materialIndex], 0, nullptr);
                glm::mat4 modelView = view * obj.transform;
                GeomPush push{proj * modelView, modelView};
                vkCmdPushConstants(cmd, geomPipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
                vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.firstIndex, 0, 0);
            }
            vkCmdEndRendering(cmd);
        }

        // =============================================================
        // PASS 2 – SSAO: reads pos + normal → raw occlusion image
        // =============================================================

        colorToRead(cmd, f.pos);
        colorToRead(cmd, f.normal);
        toColorWrite(cmd, f.ssao);

        {
            VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            ca.imageView = f.ssaoView; ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0,0}, swapChainExtent}; ri.layerCount = 1;
            ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssaoPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                ssaoPipelineLayout, 0, 1, &ssaoDescSets[currentFrame], 0, nullptr);

            SsaoPush ssaoPush{};
            ssaoPush.proj       = proj;
            ssaoPush.noiseScale = glm::vec2(float(swapChainExtent.width)  / 4.0f,
                                            float(swapChainExtent.height) / 4.0f);
            ssaoPush.radius = 0.5f;
            ssaoPush.bias   = 0.025f;
            vkCmdPushConstants(cmd, ssaoPipelineLayout,
                VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ssaoPush), &ssaoPush);

            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
        }

        // =============================================================
        // PASS 3 – SSAO Blur: 4×4 box blur → smoothed occlusion
        // =============================================================

        colorToRead(cmd, f.ssao);
        toColorWrite(cmd, f.ssaoBlur);

        {
            VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            ca.imageView = f.ssaoBlurView; ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0,0}, swapChainExtent}; ri.layerCount = 1;
            ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssaoBlurPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                ssaoBlurPipelineLayout, 0, 1, &ssaoBlurDescSets[currentFrame], 0, nullptr);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
        }

        // =============================================================
        // PASS 4 – Lighting: G-buffers + ssaoBlur → swapchain
        // =============================================================

        // ssaoBlur: COLOR_ATTACHMENT → SHADER_READ_ONLY
        colorToRead(cmd, f.ssaoBlur);
        // albedo: COLOR_ATTACHMENT (from pass 1) → SHADER_READ_ONLY
        colorToRead(cmd, f.albedo);
        // pos and normal are already SHADER_READ_ONLY from pass 2

        transitionImageLayout(cmd, swapChainImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED,                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,             VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        {
            VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            ca.imageView = swapChainImageViews[imageIndex];
            ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0,0}, swapChainExtent}; ri.layerCount = 1;
            ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lightingPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                lightingPipelineLayout, 0, 1, &lightingDescSets[currentFrame], 0, nullptr);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
        }

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
        destroyFrameImages();
    }

    void onRecreateSwapChain() override
    {
        createDepthResources(depthFormat, depthImage, depthMemory, depthImageView);
        createFrameImages();
        updateDescriptorSets(); // re-point views after resize
    }

    void onCleanup() override
    {
        destroyUBOs(lightingUBOs, lightingUBOMem);

        vkDestroyBuffer(device, ssaoKernelUBO, nullptr);
        vkFreeMemory(device, ssaoKernelMem, nullptr);

        for (int i = 0; i < MAT_COUNT; ++i)
        {
            vkDestroyBuffer(device, materialUBOs[i], nullptr);
            vkFreeMemory(device, materialUBOMem[i], nullptr);
        }

        vkDestroyBuffer(device, sceneIB, nullptr); vkFreeMemory(device, sceneIBMem, nullptr);
        vkDestroyBuffer(device, sceneVB, nullptr); vkFreeMemory(device, sceneVBMem, nullptr);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, lightingLayout,  nullptr);
        vkDestroyDescriptorSetLayout(device, ssaoBlurLayout,  nullptr);
        vkDestroyDescriptorSetLayout(device, ssaoLayout,      nullptr);
        vkDestroyDescriptorSetLayout(device, geomMatLayout,   nullptr);

        vkDestroyPipeline(device,       lightingPipeline,        nullptr);
        vkDestroyPipelineLayout(device, lightingPipelineLayout,  nullptr);
        vkDestroyPipeline(device,       ssaoBlurPipeline,        nullptr);
        vkDestroyPipelineLayout(device, ssaoBlurPipelineLayout,  nullptr);
        vkDestroyPipeline(device,       ssaoPipeline,            nullptr);
        vkDestroyPipelineLayout(device, ssaoPipelineLayout,      nullptr);
        vkDestroyPipeline(device,       geomPipeline,            nullptr);
        vkDestroyPipelineLayout(device, geomPipelineLayout,      nullptr);

        vkDestroySampler(device, noiseSampler,    nullptr);
        vkDestroySampler(device, gbufferSampler,  nullptr);
        vkDestroyImageView(device, noiseView,  nullptr);
        vkDestroyImage(device, noiseImage, nullptr);
        vkFreeMemory(device, noiseMemory, nullptr);
    }
};

// ===========================================================================

int main()
{
    SsaoApp app;
    try {
        app.run(1280, 720, "11 – SSAO (Screen-Space Ambient Occlusion)");
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }
}
