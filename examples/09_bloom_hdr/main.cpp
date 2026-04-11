
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "../common/vk_common.h"
#include "../common/vk_pipelines.h"
#include "../common/vk_descriptors.h"

// ---------------------------------------------------------------------------
// 09 – Bloom / HDR Post-processing
//
// Scene: textured ground plane lit by three coloured point lights (red, green,
// blue) that orbit above it.  The lights are visualised as glowing spheres
// whose emissive colour exceeds 1.0 in HDR space, triggering bloom.
//
// Render passes (all via Dynamic Rendering — no VkRenderPass objects):
//   Pass 1 – Scene      : ground (lit, textured) + 3 emissive spheres → hdrImage
//   Pass 2 – Bright     : fullscreen luminance filter               → brightImage
//   Pass 3 – Blur H     : 9-tap horizontal Gaussian                 → blurHImage
//   Pass 4 – Blur V     : 9-tap vertical Gaussian                   → blurVImage
//   Pass 5 – Composite  : HDR + bloom, tone-map, gamma              → swapchain
//
// New Vulkan concepts:
//   • HDR offscreen colour image  (VK_FORMAT_R16G16B16A16_SFLOAT)
//   • Fullscreen triangle  (gl_VertexIndex, zero vertex buffers)
//   • Per-frame-in-flight offscreen images (no concurrent write/read hazard)
//   • Pipeline barriers between every render-to-texture pass
//   • Multiple VkPipelineLayout / VkPipeline objects  (scene, emissive, bright,
//     blur, composite)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU data layouts
// ---------------------------------------------------------------------------

struct PointLight
{
    glm::vec4 pos;   // xyz = world position
    glm::vec4 color; // xyz = colour, w = intensity
};

struct SceneUBO
{
    PointLight lights[3]; // three coloured point-lights
    glm::vec4  viewPos;
    glm::vec4  ambient;   // xyz = colour, w = strength
};

struct ScenePush    { glm::mat4 mvp; glm::mat4 model; }; // 128 B — vertex only
struct EmissivePush { glm::mat4 mvp; glm::vec4 color; }; //  80 B — vert + frag

// ---------------------------------------------------------------------------
// CPU vertex
// ---------------------------------------------------------------------------

struct Vertex { glm::vec3 pos; glm::vec3 normal; glm::vec2 uv; };

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

// UV sphere  (diameter = 1 m,  centred at origin)
static void buildSphereMesh(std::vector<Vertex>& verts, std::vector<uint16_t>& idx,
    int stacks = 24, int slices = 24)
{
    for (int i = 0; i <= stacks; ++i)
    {
        float phi = glm::pi<float>() * i / stacks;
        float y   =  std::cos(phi);
        float r   =  std::sin(phi);
        for (int j = 0; j <= slices; ++j)
        {
            float theta = 2.0f * glm::pi<float>() * j / slices;
            glm::vec3 n(r * std::cos(theta), y, r * std::sin(theta));
            verts.push_back({n * 0.5f, n, {(float)j / slices, (float)i / stacks}});
        }
    }
    for (int i = 0; i < stacks; ++i)
        for (int j = 0; j < slices; ++j)
        {
            uint16_t a = static_cast<uint16_t>(i * (slices + 1) + j);
            uint16_t b = a + static_cast<uint16_t>(slices + 1);
            idx.insert(idx.end(), {a, b, uint16_t(a+1), b, uint16_t(b+1), uint16_t(a+1)});
        }
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

class BloomHDRApp : public VkAppBase
{
private:
    // == Offscreen images (per frame-in-flight) ==============================

    static constexpr VkFormat HDR_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;

    struct PerFrame
    {
        VkImage        hdr = VK_NULL_HANDLE, bright = VK_NULL_HANDLE,
                       blurH = VK_NULL_HANDLE, blurV = VK_NULL_HANDLE;
        VkDeviceMemory hdrMem = VK_NULL_HANDLE, brightMem = VK_NULL_HANDLE,
                       blurHMem = VK_NULL_HANDLE, blurVMem = VK_NULL_HANDLE;
        VkImageView    hdrView = VK_NULL_HANDLE, brightView = VK_NULL_HANDLE,
                       blurHView = VK_NULL_HANDLE, blurVView = VK_NULL_HANDLE;
    };
    std::vector<PerFrame> frames; // [MAX_FRAMES_IN_FLIGHT]

    VkSampler offscreenSampler = VK_NULL_HANDLE; // linear-clamp, shared by post-process passes

    // == Screen depth ========================================================
    VkImage        depthImage     = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory    = VK_NULL_HANDLE;
    VkImageView    depthImageView = VK_NULL_HANDLE;
    const VkFormat depthFormat    = VK_FORMAT_D32_SFLOAT;

    // == Ground texture ======================================================
    VkImage        groundTexImage  = VK_NULL_HANDLE;
    VkDeviceMemory groundTexMemory = VK_NULL_HANDLE;
    VkImageView    groundTexView   = VK_NULL_HANDLE;
    VkSampler      groundTexSampler= VK_NULL_HANDLE; // repeat + linear

    // == Descriptor infrastructure ===========================================
    VkDescriptorSetLayout sceneDescLayout  = VK_NULL_HANDLE; // binding 0=UBO, 1=groundTex
    VkDescriptorSetLayout oneSamplerLayout = VK_NULL_HANDLE; // binding 0=sampler2D
    VkDescriptorSetLayout twoSamplerLayout = VK_NULL_HANDLE; // binding 0,1=sampler2D

    VkDescriptorPool             descriptorPool    = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneDescSets;     // UBO + ground texture
    std::vector<VkDescriptorSet> brightDescSets;    // reads hdr
    std::vector<VkDescriptorSet> blurHDescSets;     // reads bright
    std::vector<VkDescriptorSet> blurVDescSets;     // reads blurH
    std::vector<VkDescriptorSet> compositeDescSets; // reads hdr + blurV

    // == Pipelines ===========================================================
    VkPipelineLayout scenePipelineLayout     = VK_NULL_HANDLE;
    VkPipeline       scenePipeline           = VK_NULL_HANDLE;
    VkPipelineLayout emissivePipelineLayout  = VK_NULL_HANDLE;
    VkPipeline       emissivePipeline        = VK_NULL_HANDLE;
    VkPipelineLayout brightPipelineLayout    = VK_NULL_HANDLE;
    VkPipeline       brightPipeline          = VK_NULL_HANDLE;
    VkPipelineLayout blurPipelineLayout      = VK_NULL_HANDLE;
    VkPipeline       blurPipeline            = VK_NULL_HANDLE;
    VkPipelineLayout compositePipelineLayout = VK_NULL_HANDLE;
    VkPipeline       compositePipeline       = VK_NULL_HANDLE;

    // == Uniform buffers (per frame-in-flight) ===============================
    std::vector<VkBuffer>       uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*>          uniformBuffersMapped;

    // == Geometry ============================================================
    VkBuffer       groundVB = VK_NULL_HANDLE; VkDeviceMemory groundVBMem = VK_NULL_HANDLE;
    VkBuffer       groundIB = VK_NULL_HANDLE; VkDeviceMemory groundIBMem = VK_NULL_HANDLE;
    uint32_t       groundIndexCount = 0;

    VkBuffer       sphereVB = VK_NULL_HANDLE; VkDeviceMemory sphereVBMem = VK_NULL_HANDLE;
    VkBuffer       sphereIB = VK_NULL_HANDLE; VkDeviceMemory sphereIBMem = VK_NULL_HANDLE;
    uint32_t       sphereIndexCount = 0;

    // =======================================================================
    // Ground texture
    // =======================================================================

    void createGroundTexture()
    {
        const std::string path = std::string(DATA_DIR) + "/brick/short_bricks_floor_diff_1k.jpg";
        int w, h, ch;
        stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &ch, STBI_rgb_alpha);
        if (!pixels)
            throw std::runtime_error(std::string("stbi_load failed: ") + stbi_failure_reason());
        VkDeviceSize size = static_cast<VkDeviceSize>(w) * h * 4;

        VkBuffer staging; VkDeviceMemory stagingMem;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging, stagingMem);
        void* mapped;
        vkMapMemory(device, stagingMem, 0, size, 0, &mapped);
        std::memcpy(mapped, pixels, size);
        vkUnmapMemory(device, stagingMem);
        stbi_image_free(pixels);

        createImage(static_cast<uint32_t>(w), static_cast<uint32_t>(h),
            VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, groundTexImage, groundTexMemory);

        { auto cmd = beginOneTimeCommands();
          transitionImageLayout(cmd, groundTexImage,
            VK_IMAGE_LAYOUT_UNDEFINED,           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,    VK_ACCESS_2_TRANSFER_WRITE_BIT);
          endOneTimeCommands(cmd); }

        copyBufferToImage(staging, groundTexImage, static_cast<uint32_t>(w), static_cast<uint32_t>(h));

        { auto cmd = beginOneTimeCommands();
          transitionImageLayout(cmd, groundTexImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,        VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
          endOneTimeCommands(cmd); }

        vkDestroyBuffer(device, staging, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);

        groundTexView   = createImageView(groundTexImage, VK_FORMAT_R8G8B8A8_SRGB);
        groundTexSampler = createSampler();
    }

    void destroyGroundTexture()
    {
        vkDestroySampler(device, groundTexSampler, nullptr);   groundTexSampler = VK_NULL_HANDLE;
        vkDestroyImageView(device, groundTexView, nullptr);    groundTexView    = VK_NULL_HANDLE;
        vkDestroyImage(device, groundTexImage, nullptr);       groundTexImage   = VK_NULL_HANDLE;
        vkFreeMemory(device, groundTexMemory, nullptr);        groundTexMemory  = VK_NULL_HANDLE;
    }

    // =======================================================================
    // Offscreen images
    // =======================================================================

    void createOffscreenImages()
    {
        uint32_t w = swapChainExtent.width;
        uint32_t h = swapChainExtent.height;
        constexpr VkImageUsageFlags colorUsage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        frames.resize(MAX_FRAMES_IN_FLIGHT);
        for (auto& f : frames)
        {
            createImage(w, h, HDR_FORMAT, VK_IMAGE_TILING_OPTIMAL, colorUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.hdr,    f.hdrMem);
            createImage(w, h, HDR_FORMAT, VK_IMAGE_TILING_OPTIMAL, colorUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.bright, f.brightMem);
            createImage(w, h, HDR_FORMAT, VK_IMAGE_TILING_OPTIMAL, colorUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.blurH,  f.blurHMem);
            createImage(w, h, HDR_FORMAT, VK_IMAGE_TILING_OPTIMAL, colorUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.blurV,  f.blurVMem);
            f.hdrView    = createImageView(f.hdr,    HDR_FORMAT);
            f.brightView = createImageView(f.bright, HDR_FORMAT);
            f.blurHView  = createImageView(f.blurH,  HDR_FORMAT);
            f.blurVView  = createImageView(f.blurV,  HDR_FORMAT);
        }
    }

    void destroyOffscreenImages()
    {
        for (auto& f : frames)
        {
            vkDestroyImageView(device, f.blurVView,  nullptr);
            vkDestroyImageView(device, f.blurHView,  nullptr);
            vkDestroyImageView(device, f.brightView, nullptr);
            vkDestroyImageView(device, f.hdrView,    nullptr);
            vkDestroyImage(device, f.blurV,  nullptr); vkFreeMemory(device, f.blurVMem,  nullptr);
            vkDestroyImage(device, f.blurH,  nullptr); vkFreeMemory(device, f.blurHMem,  nullptr);
            vkDestroyImage(device, f.bright, nullptr); vkFreeMemory(device, f.brightMem, nullptr);
            vkDestroyImage(device, f.hdr,    nullptr); vkFreeMemory(device, f.hdrMem,    nullptr);
        }
        frames.clear();
    }

    // =======================================================================
    // Depth buffer
    // =======================================================================

    void createSamplers()
    {
        offscreenSampler = createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    }

    // =======================================================================
    // Descriptor set layouts
    // =======================================================================

    void createDescriptorSetLayouts()
    {
        sceneDescLayout = DescriptorLayoutBuilder()
            .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(device);
        oneSamplerLayout = DescriptorLayoutBuilder()
            .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(device);
        twoSamplerLayout = DescriptorLayoutBuilder()
            .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(device);
    }

    // =======================================================================
    // Descriptor pool + sets
    // =======================================================================

    void createDescriptorPool()
    {
        // scene(UBO) x N + samplers: scene(1) + bright(1) + blurH(1) + blurV(1) + composite(2) = 6 per frame
        descriptorPool = DescriptorPoolBuilder()
            .addSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAX_FRAMES_IN_FLIGHT)
            .addSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES_IN_FLIGHT * 6)
            .build(device, MAX_FRAMES_IN_FLIGHT * 5);
    }

    void allocateDescriptorSets()
    {
        auto alloc = [&](VkDescriptorSetLayout layout, std::vector<VkDescriptorSet>& sets) {
            std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, layout);
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = descriptorPool; ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
            ai.pSetLayouts    = layouts.data();
            sets.resize(MAX_FRAMES_IN_FLIGHT);
            vkAllocateDescriptorSets(device, &ai, sets.data());
        };
        alloc(sceneDescLayout,  sceneDescSets);
        alloc(oneSamplerLayout, brightDescSets);
        alloc(oneSamplerLayout, blurHDescSets);
        alloc(oneSamplerLayout, blurVDescSets);
        alloc(twoSamplerLayout, compositeDescSets);
    }

    // Re-writes all image-view bindings. Called once after allocation and again
    // after swapchain / offscreen image recreation.
    void updateDescriptorSets()
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            DescriptorWriter()
                .writeBuffer(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    uniformBuffers[i], 0, sizeof(SceneUBO))
                .writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    groundTexView, groundTexSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .update(device, sceneDescSets[i]);

            DescriptorWriter()
                .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    frames[i].hdrView, offscreenSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .update(device, brightDescSets[i]);

            DescriptorWriter()
                .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    frames[i].brightView, offscreenSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .update(device, blurHDescSets[i]);

            DescriptorWriter()
                .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    frames[i].blurHView, offscreenSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .update(device, blurVDescSets[i]);

            DescriptorWriter()
                .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    frames[i].hdrView,   offscreenSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    frames[i].blurVView, offscreenSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .update(device, compositeDescSets[i]);
        }
    }

    // =======================================================================
    // Pipeline creation
    // =======================================================================

    void createScenePipeline()
    {
        VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/scene.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/scene.frag.spv"));

        auto [pipeline, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert).fragShader(frag)
            .vertexBinding<Vertex>()
            .vertexAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
            .vertexAttribute(1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
            .vertexAttribute(2, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv))
            .pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(ScenePush))
            .descriptorSetLayout(sceneDescLayout)
            .colorFormat(HDR_FORMAT)
            .depthFormat(depthFormat)
            .build();

        scenePipeline = pipeline; scenePipelineLayout = layout;
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    void createEmissivePipeline()
    {
        VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/emissive.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/emissive.frag.spv"));

        auto [pipeline, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert).fragShader(frag)
            .vertexBinding<Vertex>()
            .vertexAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
            .pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(EmissivePush))
            .colorFormat(HDR_FORMAT)
            .depthFormat(depthFormat)
            .build();

        emissivePipeline = pipeline; emissivePipelineLayout = layout;
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    void createBrightPipeline()
    {
        VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/fullscreen.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/bright.frag.spv"));

        auto [pipeline, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert).fragShader(frag)
            .noVertexInput()
            .pushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(float))
            .descriptorSetLayout(oneSamplerLayout)
            .colorFormat(HDR_FORMAT)
            .noDepth().cullMode(VK_CULL_MODE_NONE)
            .build();

        brightPipeline = pipeline; brightPipelineLayout = layout;
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    void createBlurPipeline()
    {
        VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/fullscreen.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/blur.frag.spv"));

        auto [pipeline, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert).fragShader(frag)
            .noVertexInput()
            .pushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(int32_t))
            .descriptorSetLayout(oneSamplerLayout)
            .colorFormat(HDR_FORMAT)
            .noDepth().cullMode(VK_CULL_MODE_NONE)
            .build();

        blurPipeline = pipeline; blurPipelineLayout = layout;
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    void createCompositePipeline()
    {
        VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/fullscreen.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/composite.frag.spv"));

        auto [pipeline, layout] = GraphicsPipelineBuilder(device)
            .vertShader(vert).fragShader(frag)
            .noVertexInput()
            .pushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(float) * 2)
            .descriptorSetLayout(twoSamplerLayout)
            .colorFormat(swapChainImageFormat)
            .noDepth().cullMode(VK_CULL_MODE_NONE)
            .build();

        compositePipeline = pipeline; compositePipelineLayout = layout;
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
    }

    // =======================================================================
    // Geometry upload
    // =======================================================================

    void createGeometry()
    {
        // Ground plane (20 × 20 m, Y = 0).  UV 0..8 tiles the texture.
        {
            constexpr float H = 10.0f;
            std::vector<Vertex> v = {
                {{-H,0,-H},{0,1,0},{0,0}}, {{ H,0,-H},{0,1,0},{8,0}},
                {{ H,0, H},{0,1,0},{8,8}}, {{-H,0, H},{0,1,0},{0,8}},
            };
            std::vector<uint16_t> i = {0,2,1, 0,3,2};
            groundIndexCount = static_cast<uint32_t>(i.size());
            uploadStagedBuffer(v.data(), v.size()*sizeof(Vertex),   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, groundVB, groundVBMem);
            uploadStagedBuffer(i.data(), i.size()*sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,  groundIB, groundIBMem);
        }
        {
            std::vector<Vertex>   verts;
            std::vector<uint16_t> idx;
            buildSphereMesh(verts, idx);
            sphereIndexCount = static_cast<uint32_t>(idx.size());
            uploadStagedBuffer(verts.data(), verts.size()*sizeof(Vertex),  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, sphereVB, sphereVBMem);
            uploadStagedBuffer(idx.data(),   idx.size()*sizeof(uint16_t),  VK_BUFFER_USAGE_INDEX_BUFFER_BIT,  sphereIB, sphereIBMem);
        }
    }

    void updateUniformBuffer(uint32_t frameIndex)
    {
        float t = static_cast<float>(glfwGetTime());

        // Three coloured point lights orbiting the scene at Y = 2.5 m, radius = 3.5 m
        static const glm::vec3 lightColors[3] = {
            {1.0f, 0.08f, 0.04f}, // red
            {0.08f, 1.0f, 0.08f}, // green
            {0.08f, 0.15f, 1.0f}, // blue
        };
        constexpr float orbitRadius = 3.0f;
        constexpr float orbitY      = 1.5f;
        constexpr float orbitSpeed  = 0.5f;

        SceneUBO ubo{};
        for (int i = 0; i < 3; ++i)
        {
            float angle     = t * orbitSpeed + i * (2.0f * glm::pi<float>() / 3.0f);
            glm::vec3 pos(orbitRadius * std::cos(angle), orbitY,
                          orbitRadius * std::sin(angle));
            ubo.lights[i].pos   = glm::vec4(pos, 0.0f);
            ubo.lights[i].color = glm::vec4(lightColors[i], 10.0f); // intensity = 10
        }
        ubo.viewPos = glm::vec4(0.0f, 8.0f, 8.0f, 0.0f);
        ubo.ambient = glm::vec4(0.12f, 0.12f, 0.15f, 1.0f); // soft blue-ish ambient

        std::memcpy(uniformBuffersMapped[frameIndex], &ubo, sizeof(ubo));
    }

    // =======================================================================
    // Barrier shorthand
    // =======================================================================

    void colorToRead(VkCommandBuffer cmd, VkImage img)
    {
        transitionImageLayout(cmd, img,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,         VK_ACCESS_2_SHADER_READ_BIT);
    }

    void toColorWrite(VkCommandBuffer cmd, VkImage img)
    {
        transitionImageLayout(cmd, img,
            VK_IMAGE_LAYOUT_UNDEFINED,                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,             VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    }

    // =======================================================================
    // VkAppBase hooks
    // =======================================================================

protected:
    void onInitBeforeCommandPool() override
    {
        createDepthResources(depthFormat, depthImage, depthMemory, depthImageView);
        createOffscreenImages();
        createSamplers();
        createDescriptorSetLayouts();
        createScenePipeline();
        createEmissivePipeline();
        createBrightPipeline();
        createBlurPipeline();
        createCompositePipeline();
    }

    void onInitAfterCommandPool() override
    {
        createGroundTexture();
        createGeometry();
        createPersistentUBOs(sizeof(SceneUBO), MAX_FRAMES_IN_FLIGHT,
            uniformBuffers, uniformBuffersMemory, uniformBuffersMapped);
        createDescriptorPool();
        allocateDescriptorSets();
        updateDescriptorSets();
    }

    void onBeforeRecord(uint32_t frameIndex) override { updateUniformBuffer(frameIndex); }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) override
    {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
            throw std::runtime_error("failed to begin command buffer");

        float t      = static_cast<float>(glfwGetTime());
        float aspect = (float)swapChainExtent.width / (float)swapChainExtent.height;

        glm::mat4 view = glm::lookAt(glm::vec3(0,8,8), glm::vec3(0,0,0), glm::vec3(0,1,0));
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        proj[1][1] *= -1.0f;

        // Emissive sphere colours — match the point light colours, HDR values > 1.0 bloom
        static const glm::vec4 sphereEmissive[3] = {
            {8.0f, 0.4f, 0.2f, 1.0f}, // red
            {0.2f, 8.0f, 0.4f, 1.0f}, // green
            {0.2f, 0.5f, 8.0f, 1.0f}, // blue
        };

        VkViewport vp{0,0,(float)swapChainExtent.width,(float)swapChainExtent.height,0,1};
        VkRect2D   sc{{0,0}, swapChainExtent};

        // =====================================================================
        // PASS 1 — Scene
        // Draw textured ground (lit by 3 point lights) + 3 emissive spheres.
        // =====================================================================

        toColorWrite(cmd, frames[currentFrame].hdr);
        transitionImageLayout(cmd, depthImage,
            VK_IMAGE_LAYOUT_UNDEFINED,                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,           VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        {
            VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            ca.imageView = frames[currentFrame].hdrView;
            ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            ca.clearValue = {{{0,0,0,1}}};

            VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            da.imageView = depthImageView;
            da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            da.clearValue.depthStencil = {1.0f, 0};

            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0,0},swapChainExtent}; ri.layerCount = 1;
            ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca; ri.pDepthAttachment = &da;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);

            // Ground — lit + textured
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                scenePipelineLayout, 0, 1, &sceneDescSets[currentFrame], 0, nullptr);
            {
                VkBuffer bufs[] = {groundVB}; VkDeviceSize offs[] = {0};
                vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offs);
                vkCmdBindIndexBuffer(cmd, groundIB, 0, VK_INDEX_TYPE_UINT16);
                glm::mat4 model(1.0f);
                ScenePush pc{proj*view*model, model};
                vkCmdPushConstants(cmd, scenePipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
                vkCmdDrawIndexed(cmd, groundIndexCount, 1, 0, 0, 0);
            }

            // 3 emissive spheres — orbiting as point-light surrogates
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, emissivePipeline);
            {
                VkBuffer bufs[] = {sphereVB}; VkDeviceSize offs[] = {0};
                vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offs);
                vkCmdBindIndexBuffer(cmd, sphereIB, 0, VK_INDEX_TYPE_UINT16);

                for (int i = 0; i < 3; ++i)
                {
                    float angle = t * 0.5f + i * (2.0f * glm::pi<float>() / 3.0f);
                    glm::vec3 pos(3.0f * std::cos(angle), 1.5f, 3.0f * std::sin(angle));
                    glm::mat4 model = glm::translate(glm::mat4(1.0f), pos);
                    model = glm::scale(model, glm::vec3(0.7f)); // radius ≈ 0.35 m

                    EmissivePush ep{proj*view*model, sphereEmissive[i]};
                    vkCmdPushConstants(cmd, emissivePipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,
                        0, sizeof(ep), &ep);
                    vkCmdDrawIndexed(cmd, sphereIndexCount, 1, 0, 0, 0);
                }
            }
            vkCmdEndRendering(cmd);
        }

        // =====================================================================
        // PASS 2 — Bright filter
        // =====================================================================

        colorToRead(cmd, frames[currentFrame].hdr);
        toColorWrite(cmd, frames[currentFrame].bright);
        {
            VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            att.imageView = frames[currentFrame].brightView;
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0,0},swapChainExtent}; ri.layerCount = 1;
            ri.colorAttachmentCount = 1; ri.pColorAttachments = &att;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, brightPipeline);
            vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                brightPipelineLayout, 0, 1, &brightDescSets[currentFrame], 0, nullptr);
            float threshold = 0.7f;
            vkCmdPushConstants(cmd, brightPipelineLayout,
                VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &threshold);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
        }

        // =====================================================================
        // PASS 3 — Blur H
        // =====================================================================

        colorToRead(cmd, frames[currentFrame].bright);
        toColorWrite(cmd, frames[currentFrame].blurH);
        {
            VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            att.imageView = frames[currentFrame].blurHView;
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0,0},swapChainExtent}; ri.layerCount = 1;
            ri.colorAttachmentCount = 1; ri.pColorAttachments = &att;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blurPipeline);
            vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                blurPipelineLayout, 0, 1, &blurHDescSets[currentFrame], 0, nullptr);
            int32_t horizontal = 1;
            vkCmdPushConstants(cmd, blurPipelineLayout,
                VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int32_t), &horizontal);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
        }

        // =====================================================================
        // PASS 4 — Blur V
        // =====================================================================

        colorToRead(cmd, frames[currentFrame].blurH);
        toColorWrite(cmd, frames[currentFrame].blurV);
        {
            VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            att.imageView = frames[currentFrame].blurVView;
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0,0},swapChainExtent}; ri.layerCount = 1;
            ri.colorAttachmentCount = 1; ri.pColorAttachments = &att;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blurPipeline);
            vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                blurPipelineLayout, 0, 1, &blurVDescSets[currentFrame], 0, nullptr);
            int32_t horizontal = 0;
            vkCmdPushConstants(cmd, blurPipelineLayout,
                VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int32_t), &horizontal);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
        }

        // =====================================================================
        // PASS 5 — Composite  (tone map + bloom → swapchain)
        // =====================================================================

        colorToRead(cmd, frames[currentFrame].blurV);
        transitionImageLayout(cmd, swapChainImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED,                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,             VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        {
            VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            att.imageView = swapChainImageViews[imageIndex];
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0,0},swapChainExtent}; ri.layerCount = 1;
            ri.colorAttachmentCount = 1; ri.pColorAttachments = &att;

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline);
            vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                compositePipelineLayout, 0, 1, &compositeDescSets[currentFrame], 0, nullptr);
            float push[2] = {1.0f, 2.5f}; // exposure, bloomStrength
            vkCmdPushConstants(cmd, compositePipelineLayout,
                VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), push);
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
        destroyOffscreenImages();
    }

    void onRecreateSwapChain() override
    {
        createDepthResources(depthFormat, depthImage, depthMemory, depthImageView);
        createOffscreenImages();
        updateDescriptorSets();
    }

    void onCleanup() override
    {
        destroyUBOs(uniformBuffers, uniformBuffersMemory);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, twoSamplerLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, oneSamplerLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, sceneDescLayout,  nullptr);

        destroyGroundTexture();

        vkDestroyBuffer(device, sphereIB, nullptr); vkFreeMemory(device, sphereIBMem, nullptr);
        vkDestroyBuffer(device, sphereVB, nullptr); vkFreeMemory(device, sphereVBMem, nullptr);
        vkDestroyBuffer(device, groundIB, nullptr); vkFreeMemory(device, groundIBMem, nullptr);
        vkDestroyBuffer(device, groundVB, nullptr); vkFreeMemory(device, groundVBMem, nullptr);

        vkDestroyPipeline(device,       compositePipeline,        nullptr);
        vkDestroyPipelineLayout(device, compositePipelineLayout,  nullptr);
        vkDestroyPipeline(device,       blurPipeline,             nullptr);
        vkDestroyPipelineLayout(device, blurPipelineLayout,       nullptr);
        vkDestroyPipeline(device,       brightPipeline,           nullptr);
        vkDestroyPipelineLayout(device, brightPipelineLayout,     nullptr);
        vkDestroyPipeline(device,       emissivePipeline,         nullptr);
        vkDestroyPipelineLayout(device, emissivePipelineLayout,   nullptr);
        vkDestroyPipeline(device,       scenePipeline,            nullptr);
        vkDestroyPipelineLayout(device, scenePipelineLayout,      nullptr);

        vkDestroySampler(device, offscreenSampler, nullptr);
    }
};

// ---------------------------------------------------------------------------

int main()
{
    BloomHDRApp app;
    try {
        app.run(1280, 720, "09 – Bloom / HDR Post-processing");
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }
}
