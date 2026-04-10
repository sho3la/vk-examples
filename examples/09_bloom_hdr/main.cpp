
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "../common/vk_common.h"

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
// Shader loader
// ---------------------------------------------------------------------------

static VkShaderModule loadShader(VkDevice device, const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("Cannot open shader: " + path);
    auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint32_t> code(sz / 4);
    f.read(reinterpret_cast<char*>(code.data()), sz);

    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = sz;
    ci.pCode    = code.data();
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module: " + path);
    return mod;
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
    // Image helpers
    // =======================================================================

    void createImage2D(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
        VkImage& img, VkDeviceMemory& mem)
    {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D; ci.format = fmt;
        ci.extent    = {w, h, 1};        ci.mipLevels = 1; ci.arrayLayers = 1;
        ci.samples   = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling    = VK_IMAGE_TILING_OPTIMAL;
        ci.usage     = usage;
        if (vkCreateImage(device, &ci, nullptr, &img) != VK_SUCCESS)
            throw std::runtime_error("failed to create image");

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, img, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate image memory");
        vkBindImageMemory(device, img, mem, 0);
    }

    VkImageView createImageView2D(VkImage img, VkFormat fmt, VkImageAspectFlags aspect)
    {
        VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ci.image            = img;
        ci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        ci.format           = fmt;
        ci.subresourceRange = {aspect, 0, 1, 0, 1};
        VkImageView view;
        if (vkCreateImageView(device, &ci, nullptr, &view) != VK_SUCCESS)
            throw std::runtime_error("failed to create image view");
        return view;
    }

    // One-shot command buffer helper
    template<typename Fn>
    void oneShot(Fn&& fn)
    {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = commandPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device, &cai, &cmd);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        fn(cmd);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
    }

    // =======================================================================
    // Ground texture
    // =======================================================================

    void createGroundTexture()
    {
        int w, h, ch;
        stbi_uc* pixels = stbi_load(DATA_DIR "/brick/short_bricks_floor_diff_1k.jpg",
            &w, &h, &ch, STBI_rgb_alpha);
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

        createImage2D(w, h, VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            groundTexImage, groundTexMemory);

        oneShot([&](VkCommandBuffer cmd) {
            transitionImageLayout(cmd, groundTexImage,
                VK_IMAGE_LAYOUT_UNDEFINED,              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,   VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,       VK_ACCESS_2_TRANSFER_WRITE_BIT);
        });
        oneShot([&](VkCommandBuffer cmd) {
            VkBufferImageCopy r{};
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.imageExtent      = {(uint32_t)w, (uint32_t)h, 1};
            vkCmdCopyBufferToImage(cmd, staging, groundTexImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        });
        oneShot([&](VkCommandBuffer cmd) {
            transitionImageLayout(cmd, groundTexImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,  VK_ACCESS_2_SHADER_READ_BIT);
        });

        vkDestroyBuffer(device, staging, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);

        groundTexView = createImageView2D(groundTexImage, VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_ASPECT_COLOR_BIT);

        // Repeat sampler for tiling on the large ground plane
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.anisotropyEnable = VK_FALSE;
        if (vkCreateSampler(device, &si, nullptr, &groundTexSampler) != VK_SUCCESS)
            throw std::runtime_error("failed to create ground texture sampler");
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
            createImage2D(w, h, HDR_FORMAT, colorUsage, f.hdr,    f.hdrMem);
            createImage2D(w, h, HDR_FORMAT, colorUsage, f.bright,  f.brightMem);
            createImage2D(w, h, HDR_FORMAT, colorUsage, f.blurH,   f.blurHMem);
            createImage2D(w, h, HDR_FORMAT, colorUsage, f.blurV,   f.blurVMem);
            f.hdrView    = createImageView2D(f.hdr,    HDR_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT);
            f.brightView = createImageView2D(f.bright,  HDR_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT);
            f.blurHView  = createImageView2D(f.blurH,   HDR_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT);
            f.blurVView  = createImageView2D(f.blurV,   HDR_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT);
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

    void createDepthResources()
    {
        createImage2D(swapChainExtent.width, swapChainExtent.height, depthFormat,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImage, depthMemory);
        depthImageView = createImageView2D(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    void cleanupDepthResources()
    {
        vkDestroyImageView(device, depthImageView, nullptr); depthImageView = VK_NULL_HANDLE;
        vkDestroyImage(device, depthImage, nullptr);         depthImage     = VK_NULL_HANDLE;
        vkFreeMemory(device, depthMemory, nullptr);          depthMemory    = VK_NULL_HANDLE;
    }

    // =======================================================================
    // Samplers
    // =======================================================================

    void createSamplers()
    {
        // Linear clamp — used by all post-process passes to read offscreen images
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter    = VK_FILTER_LINEAR;
        ci.minFilter    = VK_FILTER_LINEAR;
        ci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device, &ci, nullptr, &offscreenSampler) != VK_SUCCESS)
            throw std::runtime_error("failed to create offscreen sampler");
    }

    // =======================================================================
    // Descriptor set layouts
    // =======================================================================

    void createDescriptorSetLayouts()
    {
        // Scene layout: binding 0 = UBO, binding 1 = ground texture
        {
            VkDescriptorSetLayoutBinding b[2]{};
            b[0].binding        = 0;
            b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            b[0].descriptorCount= 1;
            b[0].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
            b[1].binding        = 1;
            b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[1].descriptorCount= 1;
            b[1].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            ci.bindingCount = 2; ci.pBindings = b;
            vkCreateDescriptorSetLayout(device, &ci, nullptr, &sceneDescLayout);
        }
        // One combined-image-sampler (bright, blurH, blurV passes)
        {
            VkDescriptorSetLayoutBinding b{};
            b.binding        = 0;
            b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount= 1;
            b.stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            ci.bindingCount = 1; ci.pBindings = &b;
            vkCreateDescriptorSetLayout(device, &ci, nullptr, &oneSamplerLayout);
        }
        // Two combined-image-samplers (composite pass: hdr + bloom)
        {
            VkDescriptorSetLayoutBinding b[2]{};
            b[0].binding        = 0;
            b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[0].descriptorCount= 1;
            b[0].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
            b[1] = b[0]; b[1].binding = 1;
            VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            ci.bindingCount = 2; ci.pBindings = b;
            vkCreateDescriptorSetLayout(device, &ci, nullptr, &twoSamplerLayout);
        }
    }

    // =======================================================================
    // Descriptor pool + sets
    // =======================================================================

    void createDescriptorPool()
    {
        uint32_t n = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[0].descriptorCount = n;          // scene UBO
        sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        // scene(groundTex) + bright + blurH + blurV + composite(hdr+blurV) = 6 per frame
        sizes[1].descriptorCount = n * 6;

        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount = 2; ci.pPoolSizes = sizes;
        ci.maxSets       = n * 5; // scene + bright + blurH + blurV + composite
        if (vkCreateDescriptorPool(device, &ci, nullptr, &descriptorPool) != VK_SUCCESS)
            throw std::runtime_error("failed to create descriptor pool");
    }

    void allocateDescriptorSets()
    {
        uint32_t n = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        auto alloc = [&](VkDescriptorSetLayout layout, std::vector<VkDescriptorSet>& sets) {
            std::vector<VkDescriptorSetLayout> layouts(n, layout);
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = descriptorPool; ai.descriptorSetCount = n;
            ai.pSetLayouts    = layouts.data();
            sets.resize(n);
            if (vkAllocateDescriptorSets(device, &ai, sets.data()) != VK_SUCCESS)
                throw std::runtime_error("failed to allocate descriptor sets");
        };
        alloc(sceneDescLayout,  sceneDescSets);
        alloc(oneSamplerLayout, brightDescSets);
        alloc(oneSamplerLayout, blurHDescSets);
        alloc(oneSamplerLayout, blurVDescSets);
        alloc(twoSamplerLayout, compositeDescSets);
    }

    // Re-writes all image-view bindings.  Called once after allocation and
    // again after swapchain / offscreen image recreation.
    void updateDescriptorSets()
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            auto mkBuf = [&](VkBuffer buf, VkDeviceSize sz) {
                VkDescriptorBufferInfo bi{}; bi.buffer = buf; bi.range = sz; return bi;
            };
            auto mkImg = [&](VkImageView view, VkSampler samp) {
                VkDescriptorImageInfo ii{};
                ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                ii.imageView   = view; ii.sampler = samp;
                return ii;
            };
            auto wBuf = [&](VkDescriptorSet set, uint32_t bind, VkDescriptorBufferInfo& bi) {
                VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w.dstSet = set; w.dstBinding = bind;
                w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                w.descriptorCount = 1; w.pBufferInfo = &bi; return w;
            };
            auto wImg = [&](VkDescriptorSet set, uint32_t bind, VkDescriptorImageInfo& ii) {
                VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w.dstSet = set; w.dstBinding = bind;
                w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.descriptorCount = 1; w.pImageInfo = &ii; return w;
            };

            auto uboBuf   = mkBuf(uniformBuffers[i], sizeof(SceneUBO));
            auto groundII = mkImg(groundTexView,             groundTexSampler);
            auto hdrII    = mkImg(frames[i].hdrView,         offscreenSampler);
            auto brightII = mkImg(frames[i].brightView,      offscreenSampler);
            auto blurHII  = mkImg(frames[i].blurHView,       offscreenSampler);
            auto blurVII  = mkImg(frames[i].blurVView,       offscreenSampler);

            std::array<VkWriteDescriptorSet, 8> writes = {
                wBuf(sceneDescSets[i],     0, uboBuf),
                wImg(sceneDescSets[i],     1, groundII),
                wImg(brightDescSets[i],    0, hdrII),
                wImg(blurHDescSets[i],     0, brightII),
                wImg(blurVDescSets[i],     0, blurHII),
                wImg(compositeDescSets[i], 0, hdrII),
                wImg(compositeDescSets[i], 1, blurVII),
                VkWriteDescriptorSet{}, // pad
            };
            vkUpdateDescriptorSets(device, 7, writes.data(), 0, nullptr);
        }
    }

    // =======================================================================
    // Pipeline helpers
    // =======================================================================

    static VkPipelineShaderStageCreateInfo shaderStage(
        VkShaderStageFlagBits stage, VkShaderModule mod)
    {
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.stage = stage; s.module = mod; s.pName = "main"; return s;
    }

    VkPipeline buildFullscreenPipeline(VkShaderModule fragMod,
        VkPipelineLayout layout, VkFormat colorFmt)
    {
        VkShaderModule vertMod = loadShader(device, SHADER_DIR "/fullscreen.vert.spv");

        VkPipelineShaderStageCreateInfo stages[2] = {
            shaderStage(VK_SHADER_STAGE_VERTEX_BIT,   vertMod),
            shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragMod),
        };
        VkPipelineVertexInputStateCreateInfo   vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vs.viewportCount = 1; vs.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState att{};
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|
                             VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1; blend.pAttachments = &att;
        std::array<VkDynamicState,2> dyn{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynCI{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynCI.dynamicStateCount = 2; dynCI.pDynamicStates = dyn.data();
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = 1; rendering.pColorAttachmentFormats = &colorFmt;
        VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pci.pNext = &rendering; pci.stageCount = 2; pci.pStages = stages;
        pci.pVertexInputState = &vi; pci.pInputAssemblyState = &ia;
        pci.pViewportState = &vs; pci.pRasterizationState = &rs;
        pci.pMultisampleState = &ms; pci.pDepthStencilState = &ds;
        pci.pColorBlendState = &blend; pci.pDynamicState = &dynCI;
        pci.layout = layout;
        VkPipeline pipe;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe) != VK_SUCCESS)
            throw std::runtime_error("failed to create fullscreen pipeline");
        vkDestroyShaderModule(device, vertMod, nullptr);
        return pipe;
    }

    // =======================================================================
    // Pipeline creation
    // =======================================================================

    void createScenePipeline()
    {
        VkShaderModule vert = loadShader(device, SHADER_DIR "/scene.vert.spv");
        VkShaderModule frag = loadShader(device, SHADER_DIR "/scene.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2] = {
            shaderStage(VK_SHADER_STAGE_VERTEX_BIT,   vert),
            shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, frag),
        };
        VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription,3> attrs{{
            {0,0,VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex,pos)},
            {1,0,VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex,normal)},
            {2,0,VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex,uv)},
        }};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vs.viewportCount = 1; vs.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp  = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState att{};
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|
                             VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1; blend.pAttachments = &att;
        std::array<VkDynamicState,2> dyn{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynCI{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynCI.dynamicStateCount = 2; dynCI.pDynamicStates = dyn.data();

        VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ScenePush)};
        VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutCI.setLayoutCount = 1; layoutCI.pSetLayouts = &sceneDescLayout;
        layoutCI.pushConstantRangeCount = 1; layoutCI.pPushConstantRanges = &push;
        vkCreatePipelineLayout(device, &layoutCI, nullptr, &scenePipelineLayout);

        VkFormat hdrFmt = HDR_FORMAT;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = 1; rendering.pColorAttachmentFormats = &hdrFmt;
        rendering.depthAttachmentFormat = depthFormat;

        VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pci.pNext = &rendering; pci.stageCount = 2; pci.pStages = stages;
        pci.pVertexInputState = &vi; pci.pInputAssemblyState = &ia;
        pci.pViewportState = &vs; pci.pRasterizationState = &rs;
        pci.pMultisampleState = &ms; pci.pDepthStencilState = &ds;
        pci.pColorBlendState = &blend; pci.pDynamicState = &dynCI;
        pci.layout = scenePipelineLayout;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &scenePipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create scene pipeline");

        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);
    }

    void createEmissivePipeline()
    {
        VkShaderModule vert = loadShader(device, SHADER_DIR "/emissive.vert.spv");
        VkShaderModule frag = loadShader(device, SHADER_DIR "/emissive.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2] = {
            shaderStage(VK_SHADER_STAGE_VERTEX_BIT,   vert),
            shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, frag),
        };
        VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex,pos)};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 1; vi.pVertexAttributeDescriptions = &attr;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vs.viewportCount = 1; vs.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp  = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState att{};
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|
                             VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1; blend.pAttachments = &att;
        std::array<VkDynamicState,2> dyn{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynCI{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynCI.dynamicStateCount = 2; dynCI.pDynamicStates = dyn.data();

        // No descriptor sets — push constant only
        VkPushConstantRange push{
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(EmissivePush)};
        VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutCI.pushConstantRangeCount = 1; layoutCI.pPushConstantRanges = &push;
        vkCreatePipelineLayout(device, &layoutCI, nullptr, &emissivePipelineLayout);

        VkFormat hdrFmt = HDR_FORMAT;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = 1; rendering.pColorAttachmentFormats = &hdrFmt;
        rendering.depthAttachmentFormat = depthFormat;

        VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pci.pNext = &rendering; pci.stageCount = 2; pci.pStages = stages;
        pci.pVertexInputState = &vi; pci.pInputAssemblyState = &ia;
        pci.pViewportState = &vs; pci.pRasterizationState = &rs;
        pci.pMultisampleState = &ms; pci.pDepthStencilState = &ds;
        pci.pColorBlendState = &blend; pci.pDynamicState = &dynCI;
        pci.layout = emissivePipelineLayout;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &emissivePipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create emissive pipeline");

        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);
    }

    void createBrightPipeline()
    {
        VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};
        VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutCI.setLayoutCount = 1; layoutCI.pSetLayouts = &oneSamplerLayout;
        layoutCI.pushConstantRangeCount = 1; layoutCI.pPushConstantRanges = &push;
        vkCreatePipelineLayout(device, &layoutCI, nullptr, &brightPipelineLayout);
        VkShaderModule frag = loadShader(device, SHADER_DIR "/bright.frag.spv");
        brightPipeline = buildFullscreenPipeline(frag, brightPipelineLayout, HDR_FORMAT);
        vkDestroyShaderModule(device, frag, nullptr);
    }

    void createBlurPipeline()
    {
        VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int32_t)};
        VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutCI.setLayoutCount = 1; layoutCI.pSetLayouts = &oneSamplerLayout;
        layoutCI.pushConstantRangeCount = 1; layoutCI.pPushConstantRanges = &push;
        vkCreatePipelineLayout(device, &layoutCI, nullptr, &blurPipelineLayout);
        VkShaderModule frag = loadShader(device, SHADER_DIR "/blur.frag.spv");
        blurPipeline = buildFullscreenPipeline(frag, blurPipelineLayout, HDR_FORMAT);
        vkDestroyShaderModule(device, frag, nullptr);
    }

    void createCompositePipeline()
    {
        VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float) * 2};
        VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutCI.setLayoutCount = 1; layoutCI.pSetLayouts = &twoSamplerLayout;
        layoutCI.pushConstantRangeCount = 1; layoutCI.pPushConstantRanges = &push;
        vkCreatePipelineLayout(device, &layoutCI, nullptr, &compositePipelineLayout);
        VkShaderModule frag = loadShader(device, SHADER_DIR "/composite.frag.spv");
        compositePipeline = buildFullscreenPipeline(frag, compositePipelineLayout, swapChainImageFormat);
        vkDestroyShaderModule(device, frag, nullptr);
    }

    // =======================================================================
    // Geometry upload
    // =======================================================================

    void uploadBuffer(const void* data, VkDeviceSize sz, VkBufferUsageFlags usage,
        VkBuffer& buf, VkDeviceMemory& mem)
    {
        VkBuffer staging; VkDeviceMemory stagingMem;
        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging, stagingMem);
        void* mapped;
        vkMapMemory(device, stagingMem, 0, sz, 0, &mapped);
        std::memcpy(mapped, data, sz);
        vkUnmapMemory(device, stagingMem);
        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buf, mem);
        oneShot([&](VkCommandBuffer cmd) {
            VkBufferCopy r{0, 0, sz};
            vkCmdCopyBuffer(cmd, staging, buf, 1, &r);
        });
        vkDestroyBuffer(device, staging, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);
    }

    void createGeometry()
    {
        // Ground plane (20 × 20 m, Y = 0).  UV 0..8 tiles the texture.
        {
            constexpr float H = 10.0f;
            std::vector<Vertex> v = {
                {{-H,0,-H},{0,1,0},{0,0}}, {{ H,0,-H},{0,1,0},{8,0}},
                {{ H,0, H},{0,1,0},{8,8}}, {{-H,0, H},{0,1,0},{0,8}},
            };
            // Winding: v0,v2,v1 → cross product (v2-v0)×(v1-v0) = +Y → front face visible from above
            std::vector<uint16_t> i = {0,2,1, 0,3,2};
            groundIndexCount = static_cast<uint32_t>(i.size());
            uploadBuffer(v.data(), v.size()*sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, groundVB, groundVBMem);
            uploadBuffer(i.data(), i.size()*sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, groundIB, groundIBMem);
        }
        // Sphere mesh shared by all three emissive spheres
        {
            std::vector<Vertex>   verts;
            std::vector<uint16_t> idx;
            buildSphereMesh(verts, idx);
            sphereIndexCount = static_cast<uint32_t>(idx.size());
            uploadBuffer(verts.data(), verts.size()*sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, sphereVB, sphereVBMem);
            uploadBuffer(idx.data(),  idx.size()*sizeof(uint16_t),  VK_BUFFER_USAGE_INDEX_BUFFER_BIT,  sphereIB, sphereIBMem);
        }
    }

    // =======================================================================
    // Uniform buffers
    // =======================================================================

    void createUniformBuffers()
    {
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                uniformBuffers[i], uniformBuffersMemory[i]);
            vkMapMemory(device, uniformBuffersMemory[i], 0, sizeof(SceneUBO), 0,
                &uniformBuffersMapped[i]);
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
        createDepthResources();
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
        createUniformBuffers();
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

    void onCleanupSwapChain() override { cleanupDepthResources(); destroyOffscreenImages(); }

    void onRecreateSwapChain() override
    {
        createDepthResources();
        createOffscreenImages();
        updateDescriptorSets();
    }

    void onCleanup() override
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

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
