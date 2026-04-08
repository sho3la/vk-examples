
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "../common/vk_common.h"

// ---------------------------------------------------------------------------
// 08 – Shadow Mapping
//
// Introduces two-pass rendering:
//   Pass 1 – Shadow pass   : render depth from the light's point of view into
//                             a 2048×2048 shadow map (orthographic projection).
//   Pass 2 – Main pass     : render the scene, sample the shadow map with a
//                             hardware-comparison sampler (sampler2DShadow)
//                             and apply 3×3 PCF for soft shadow edges.
//
// New Vulkan concepts:
//   • Offscreen depth image used as BOTH depth attachment AND sampled texture
//     (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)
//   • VkSampler with compareEnable + VK_COMPARE_OP_LESS_OR_EQUAL
//   • Pipeline barrier between passes:
//       DEPTH_STENCIL_ATTACHMENT_WRITE → SHADER_READ (Sync2)
//       layout: DEPTH_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
//   • Two separate VkPipelineLayout / VkPipeline objects
//   • Depth-only rendering (no color attachments in the shadow pass)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU data structures
// ---------------------------------------------------------------------------

struct SceneUBO
{
	glm::vec4 lightDir;         // xyz = toward-light direction (normalized)
	glm::vec4 lightColor;       // xyz = color,  w = intensity
	glm::vec4 viewPos;          // xyz = camera world position
	glm::mat4 lightSpaceMatrix; // ortho-proj × light-view  (used in vertex stage)
	glm::vec4 ambient;          // xyz = color,  w = strength
};

struct PushConstants
{
	glm::mat4 mvp;
	glm::mat4 model;
};

struct ShadowPushConstants
{
	glm::mat4 lightSpaceMVP;
};

// ---------------------------------------------------------------------------
// Vertex layout: position + normal + UV (no tangent — no normal mapping here)
// ---------------------------------------------------------------------------

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 uv;
};

// ---------------------------------------------------------------------------
// Ground plane  – 20 m × 20 m, UV tiles 4×
// ---------------------------------------------------------------------------

static const glm::vec3 groundNormal = {0.0f, 1.0f, 0.0f};

static const std::vector<Vertex> groundVertices = {
	{{-10.0f, 0.0f,  10.0f}, groundNormal, {0.0f, 0.0f}},
	{{ 10.0f, 0.0f,  10.0f}, groundNormal, {4.0f, 0.0f}},
	{{ 10.0f, 0.0f, -10.0f}, groundNormal, {4.0f, 4.0f}},
	{{-10.0f, 0.0f, -10.0f}, groundNormal, {0.0f, 4.0f}},
};

static const std::vector<uint16_t> groundIndices = {0, 1, 2, 2, 3, 0};

// ---------------------------------------------------------------------------
// Unit cube with per-face normals
// ---------------------------------------------------------------------------

static void pushFace(std::vector<Vertex>& verts, std::vector<uint16_t>& idxs,
	glm::vec3 bl, glm::vec3 br, glm::vec3 tr, glm::vec3 tl, glm::vec3 n)
{
	uint16_t base = static_cast<uint16_t>(verts.size());
	verts.push_back({bl, n, {0.0f, 1.0f}});
	verts.push_back({br, n, {1.0f, 1.0f}});
	verts.push_back({tr, n, {1.0f, 0.0f}});
	verts.push_back({tl, n, {0.0f, 0.0f}});
	idxs.insert(idxs.end(), {base, uint16_t(base + 1), uint16_t(base + 2),
	                          uint16_t(base + 2), uint16_t(base + 3), base});
}

static void buildCubeMesh(std::vector<Vertex>& v, std::vector<uint16_t>& i)
{
	const float h = 0.5f;
	pushFace(v, i, {-h,-h, h}, { h,-h, h}, { h, h, h}, {-h, h, h}, { 0, 0, 1}); // Front
	pushFace(v, i, { h,-h,-h}, {-h,-h,-h}, {-h, h,-h}, { h, h,-h}, { 0, 0,-1}); // Back
	pushFace(v, i, {-h,-h,-h}, {-h,-h, h}, {-h, h, h}, {-h, h,-h}, {-1, 0, 0}); // Left
	pushFace(v, i, { h,-h, h}, { h,-h,-h}, { h, h,-h}, { h, h, h}, { 1, 0, 0}); // Right
	pushFace(v, i, {-h, h, h}, { h, h, h}, { h, h,-h}, {-h, h,-h}, { 0, 1, 0}); // Top
	pushFace(v, i, {-h,-h,-h}, { h,-h,-h}, { h,-h, h}, {-h,-h, h}, { 0,-1, 0}); // Bottom
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

class ShadowMappingApp : public VkAppBase
{
private:
	// -- Shadow map (one per frame-in-flight to avoid write/read hazard) ------
	static constexpr uint32_t SHADOW_MAP_SIZE = 2048;

	std::vector<VkImage>        shadowImages;
	std::vector<VkDeviceMemory> shadowMemories;
	std::vector<VkImageView>    shadowImageViews;
	VkSampler                   shadowSampler = VK_NULL_HANDLE; // shared compare sampler

	// -- Screen depth buffer --------------------------------------------------
	VkImage        depthImage     = VK_NULL_HANDLE;
	VkDeviceMemory depthMemory    = VK_NULL_HANDLE;
	VkImageView    depthImageView = VK_NULL_HANDLE;
	VkFormat       depthFormat    = VK_FORMAT_D32_SFLOAT;

	// -- Diffuse texture ------------------------------------------------------
	VkImage        diffuseImage     = VK_NULL_HANDLE;
	VkDeviceMemory diffuseMemory    = VK_NULL_HANDLE;
	VkImageView    diffuseImageView = VK_NULL_HANDLE;
	VkSampler      diffuseSampler   = VK_NULL_HANDLE;

	// -- Shadow pipeline (depth-only) -----------------------------------------
	VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
	VkPipeline       shadowPipeline       = VK_NULL_HANDLE;

	// -- Main pipeline --------------------------------------------------------
	VkDescriptorSetLayout        mainDescriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool             descriptorPool          = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> descriptorSets;

	VkPipelineLayout mainPipelineLayout = VK_NULL_HANDLE;
	VkPipeline       mainPipeline       = VK_NULL_HANDLE;

	// -- Uniform buffers (per frame-in-flight) --------------------------------
	std::vector<VkBuffer>       uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*>          uniformBuffersMapped;

	// -- Ground geometry ------------------------------------------------------
	VkBuffer       groundVertexBuffer       = VK_NULL_HANDLE;
	VkDeviceMemory groundVertexBufferMemory = VK_NULL_HANDLE;
	VkBuffer       groundIndexBuffer        = VK_NULL_HANDLE;
	VkDeviceMemory groundIndexBufferMemory  = VK_NULL_HANDLE;

	// -- Cube geometry --------------------------------------------------------
	std::vector<Vertex>   cubeVertices;
	std::vector<uint16_t> cubeIndices;
	VkBuffer       cubeVertexBuffer       = VK_NULL_HANDLE;
	VkDeviceMemory cubeVertexBufferMemory = VK_NULL_HANDLE;
	VkBuffer       cubeIndexBuffer        = VK_NULL_HANDLE;
	VkDeviceMemory cubeIndexBufferMemory  = VK_NULL_HANDLE;

	// -- Light marker pipeline (unlit, flat yellow cube) ----------------------
	VkPipelineLayout markerPipelineLayout = VK_NULL_HANDLE;
	VkPipeline       markerPipeline       = VK_NULL_HANDLE;

	// -- Per-frame light state (computed in onBeforeRecord, used in record) ---
	glm::mat4 lightSpaceMatrix{1.0f};
	glm::vec3 lightDir{0.0f};
	glm::vec3 lightPos{0.0f};

	// === DEPTH / SHADOW RESOURCES ============================================

	void createDepthResources()
	{
		VkImageCreateInfo ci{};
		ci.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ci.imageType   = VK_IMAGE_TYPE_2D;
		ci.format      = depthFormat;
		ci.extent      = {swapChainExtent.width, swapChainExtent.height, 1};
		ci.mipLevels   = 1;
		ci.arrayLayers = 1;
		ci.samples     = VK_SAMPLE_COUNT_1_BIT;
		ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
		ci.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		if (vkCreateImage(device, &ci, nullptr, &depthImage) != VK_SUCCESS)
			throw std::runtime_error("failed to create depth image!");

		VkMemoryRequirements req;
		vkGetImageMemoryRequirements(device, depthImage, &req);
		VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		ai.allocationSize  = req.size;
		ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (vkAllocateMemory(device, &ai, nullptr, &depthMemory) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate depth memory!");
		vkBindImageMemory(device, depthImage, depthMemory, 0);

		VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
		vi.image            = depthImage;
		vi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
		vi.format           = depthFormat;
		vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
		if (vkCreateImageView(device, &vi, nullptr, &depthImageView) != VK_SUCCESS)
			throw std::runtime_error("failed to create depth image view!");
	}

	void cleanupDepthResources()
	{
		vkDestroyImageView(device, depthImageView, nullptr); depthImageView = VK_NULL_HANDLE;
		vkDestroyImage(device, depthImage, nullptr);         depthImage     = VK_NULL_HANDLE;
		vkFreeMemory(device, depthMemory, nullptr);          depthMemory    = VK_NULL_HANDLE;
	}

	// The shadow map is a depth image that is ALSO sampled as a texture.
	// One image per frame-in-flight so Frame N+1 cannot clobber Frame N's
	// shadow map while the GPU is still reading it in the main pass.
	void createShadowResources()
	{
		shadowImages.resize(MAX_FRAMES_IN_FLIGHT);
		shadowMemories.resize(MAX_FRAMES_IN_FLIGHT);
		shadowImageViews.resize(MAX_FRAMES_IN_FLIGHT);

		VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		ci.imageType   = VK_IMAGE_TYPE_2D;
		ci.format      = depthFormat; // D32_SFLOAT
		ci.extent      = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
		ci.mipLevels   = 1;
		ci.arrayLayers = 1;
		ci.samples     = VK_SAMPLE_COUNT_1_BIT;
		ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
		ci.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
		                 VK_IMAGE_USAGE_SAMPLED_BIT;

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			if (vkCreateImage(device, &ci, nullptr, &shadowImages[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create shadow image!");

			VkMemoryRequirements req;
			vkGetImageMemoryRequirements(device, shadowImages[i], &req);
			VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
			ai.allocationSize  = req.size;
			ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			if (vkAllocateMemory(device, &ai, nullptr, &shadowMemories[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to allocate shadow memory!");
			vkBindImageMemory(device, shadowImages[i], shadowMemories[i], 0);

			VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
			vi.image            = shadowImages[i];
			vi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
			vi.format           = depthFormat;
			vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
			if (vkCreateImageView(device, &vi, nullptr, &shadowImageViews[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create shadow image view!");
		}

		// Comparison sampler — hardware does depth test per texel.
		// CLAMP_TO_BORDER + white border → fragments outside the shadow frustum
		// always receive shadowFactor = 1.0 (fully lit).
		// The sampler is shared across all frames (it has no per-frame state).
		VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
		si.magFilter        = VK_FILTER_LINEAR;
		si.minFilter        = VK_FILTER_LINEAR;
		si.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		si.addressModeU     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		si.addressModeV     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		si.addressModeW     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		si.borderColor      = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; // depth = 1.0 = not in shadow
		si.compareEnable    = VK_TRUE;
		si.compareOp        = VK_COMPARE_OP_LESS_OR_EQUAL;
		if (vkCreateSampler(device, &si, nullptr, &shadowSampler) != VK_SUCCESS)
			throw std::runtime_error("failed to create shadow sampler!");
	}

	void destroyShadowResources()
	{
		vkDestroySampler(device, shadowSampler, nullptr); shadowSampler = VK_NULL_HANDLE;
		for (int i = 0; i < static_cast<int>(shadowImageViews.size()); ++i)
		{
			vkDestroyImageView(device, shadowImageViews[i], nullptr);
			vkDestroyImage(device, shadowImages[i], nullptr);
			vkFreeMemory(device, shadowMemories[i], nullptr);
		}
		shadowImageViews.clear();
		shadowImages.clear();
		shadowMemories.clear();
	}

	// === DIFFUSE TEXTURE =====================================================

	void uploadImage(const std::string& path, VkFormat format,
		VkImage& image, VkDeviceMemory& memory, VkImageView& view)
	{
		int w, h, ch;
		stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &ch, STBI_rgb_alpha);
		if (!pixels)
			throw std::runtime_error("stbi_load failed (" + path + "): " + stbi_failure_reason());

		VkDeviceSize size = static_cast<VkDeviceSize>(w) * h * 4;

		VkBuffer staging; VkDeviceMemory stagingMem;
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			staging, stagingMem);

		void* data;
		vkMapMemory(device, stagingMem, 0, size, 0, &data);
		std::memcpy(data, pixels, size);
		vkUnmapMemory(device, stagingMem);
		stbi_image_free(pixels);

		// Create device-local image
		VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		ci.imageType     = VK_IMAGE_TYPE_2D;
		ci.format        = format;
		ci.extent        = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
		ci.mipLevels     = 1;
		ci.arrayLayers   = 1;
		ci.samples       = VK_SAMPLE_COUNT_1_BIT;
		ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
		ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		ci.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
		vkCreateImage(device, &ci, nullptr, &image);

		VkMemoryRequirements req;
		vkGetImageMemoryRequirements(device, image, &req);
		VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		ai.allocationSize  = req.size;
		ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vkAllocateMemory(device, &ai, nullptr, &memory);
		vkBindImageMemory(device, image, memory, 0);

		// Transition → TRANSFER_DST
		auto oneShot = [&](auto fn) {
			VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
			cai.commandPool = commandPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
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
		};

		oneShot([&](VkCommandBuffer cmd) {
			transitionImageLayout(cmd, image,
				VK_IMAGE_LAYOUT_UNDEFINED,           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
				VK_PIPELINE_STAGE_2_TRANSFER_BIT,    VK_ACCESS_2_TRANSFER_WRITE_BIT);
		});

		oneShot([&](VkCommandBuffer cmd) {
			VkBufferImageCopy region{};
			region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
			region.imageExtent      = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
			vkCmdCopyBufferToImage(cmd, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
		});

		oneShot([&](VkCommandBuffer cmd) {
			transitionImageLayout(cmd, image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_2_TRANSFER_BIT,        VK_ACCESS_2_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
		});

		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);

		VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
		vi.image            = image;
		vi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
		vi.format           = format;
		vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		vkCreateImageView(device, &vi, nullptr, &view);
	}

	void createDiffuseTexture()
	{
		uploadImage(std::string(DATA_DIR) + "/brick/short_bricks_floor_diff_1k.jpg",
			VK_FORMAT_R8G8B8A8_SRGB, diffuseImage, diffuseMemory, diffuseImageView);

		VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
		si.magFilter    = VK_FILTER_LINEAR;
		si.minFilter    = VK_FILTER_LINEAR;
		si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		si.maxAnisotropy = 1.0f;
		vkCreateSampler(device, &si, nullptr, &diffuseSampler);
	}

	void destroyDiffuseTexture()
	{
		vkDestroySampler(device, diffuseSampler, nullptr);    diffuseSampler   = VK_NULL_HANDLE;
		vkDestroyImageView(device, diffuseImageView, nullptr); diffuseImageView = VK_NULL_HANDLE;
		vkDestroyImage(device, diffuseImage, nullptr);        diffuseImage     = VK_NULL_HANDLE;
		vkFreeMemory(device, diffuseMemory, nullptr);         diffuseMemory    = VK_NULL_HANDLE;
	}

	// === DESCRIPTOR SET LAYOUT ===============================================

	void createMainDescriptorSetLayout()
	{
		std::array<VkDescriptorSetLayoutBinding, 3> b{};

		b[0].binding         = 0;
		b[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		b[0].descriptorCount = 1;
		b[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		b[1].binding         = 1;
		b[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		b[1].descriptorCount = 1;
		b[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

		b[2].binding         = 2;
		b[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		b[2].descriptorCount = 1;
		b[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		ci.bindingCount = static_cast<uint32_t>(b.size());
		ci.pBindings    = b.data();
		if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &mainDescriptorSetLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor set layout!");
	}

	// === PIPELINES ===========================================================

	// Depth-only pipeline used for the shadow pass.
	// No color attachment, only writes to the shadow depth image.
	void createShadowPipeline()
	{
		auto vertCode   = readFile(std::string(SHADER_DIR) + "/shadow.vert.spv");
		VkShaderModule vertModule = createShaderModule(vertCode);

		VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
		stage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
		stage.module = vertModule;
		stage.pName  = "main";

		// Only need position (location 0); stride covers the full Vertex struct
		VkVertexInputBindingDescription bindDesc{};
		bindDesc.binding   = 0;
		bindDesc.stride    = sizeof(Vertex);
		bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		VkVertexInputAttributeDescription attrDesc{};
		attrDesc.location = 0;
		attrDesc.binding  = 0;
		attrDesc.format   = VK_FORMAT_R32G32B32_SFLOAT;
		attrDesc.offset   = offsetof(Vertex, pos);

		VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
		vertexInput.vertexBindingDescriptionCount   = 1;
		vertexInput.pVertexBindingDescriptions      = &bindDesc;
		vertexInput.vertexAttributeDescriptionCount = 1;
		vertexInput.pVertexAttributeDescriptions    = &attrDesc;

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
		viewportState.viewportCount = 1;
		viewportState.scissorCount  = 1;

		VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
		dynState.dynamicStateCount = 2;
		dynState.pDynamicStates    = dynStates;

		VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
		raster.polygonMode            = VK_POLYGON_MODE_FILL;
		raster.cullMode               = VK_CULL_MODE_BACK_BIT;
		raster.frontFace              = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		raster.lineWidth              = 1.0f;
		raster.depthBiasEnable        = VK_FALSE;
		raster.depthBiasConstantFactor = 0.0f;
		raster.depthBiasSlopeFactor   = 0.0f;

		VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
		msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
		depthStencil.depthTestEnable  = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

		// No color attachments
		VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
		blend.attachmentCount = 0;

		VkPushConstantRange pcRange{};
		pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pcRange.offset     = 0;
		pcRange.size       = sizeof(ShadowPushConstants);

		VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		layoutCI.pushConstantRangeCount = 1;
		layoutCI.pPushConstantRanges    = &pcRange;
		if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &shadowPipelineLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create shadow pipeline layout!");

		// Dynamic rendering with depth only (no color format)
		VkPipelineRenderingCreateInfo renderingCI{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
		renderingCI.depthAttachmentFormat = depthFormat;

		VkGraphicsPipelineCreateInfo pipelineCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		pipelineCI.pNext               = &renderingCI;
		pipelineCI.stageCount          = 1;
		pipelineCI.pStages             = &stage;
		pipelineCI.pVertexInputState   = &vertexInput;
		pipelineCI.pInputAssemblyState = &inputAssembly;
		pipelineCI.pViewportState      = &viewportState;
		pipelineCI.pRasterizationState = &raster;
		pipelineCI.pMultisampleState   = &msaa;
		pipelineCI.pDepthStencilState  = &depthStencil;
		pipelineCI.pColorBlendState    = &blend;
		pipelineCI.pDynamicState       = &dynState;
		pipelineCI.layout              = shadowPipelineLayout;
		pipelineCI.renderPass          = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &shadowPipeline) != VK_SUCCESS)
			throw std::runtime_error("failed to create shadow pipeline!");

		vkDestroyShaderModule(device, vertModule, nullptr);
	}

	void createMainPipeline()
	{
		auto vertCode = readFile(std::string(SHADER_DIR) + "/scene.vert.spv");
		auto fragCode = readFile(std::string(SHADER_DIR) + "/scene.frag.spv");
		VkShaderModule vertModule = createShaderModule(vertCode);
		VkShaderModule fragModule = createShaderModule(fragCode);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vertModule;
		stages[0].pName  = "main";
		stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = fragModule;
		stages[1].pName  = "main";

		VkVertexInputBindingDescription bindDesc{};
		bindDesc.binding   = 0;
		bindDesc.stride    = sizeof(Vertex);
		bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		std::array<VkVertexInputAttributeDescription, 3> attrDescs{};
		attrDescs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
		attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
		attrDescs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)};

		VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
		vertexInput.vertexBindingDescriptionCount   = 1;
		vertexInput.pVertexBindingDescriptions      = &bindDesc;
		vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
		vertexInput.pVertexAttributeDescriptions    = attrDescs.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
		viewportState.viewportCount = 1;
		viewportState.scissorCount  = 1;

		VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
		dynState.dynamicStateCount = 2;
		dynState.pDynamicStates    = dynStates;

		VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
		raster.polygonMode = VK_POLYGON_MODE_FILL;
		raster.cullMode    = VK_CULL_MODE_BACK_BIT;
		raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		raster.lineWidth   = 1.0f;

		VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
		msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
		depthStencil.depthTestEnable  = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

		VkPipelineColorBlendAttachmentState blendAtt{};
		blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
		blend.attachmentCount = 1;
		blend.pAttachments    = &blendAtt;

		VkPushConstantRange pcRange{};
		pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pcRange.offset     = 0;
		pcRange.size       = sizeof(PushConstants); // mat4 mvp + mat4 model = 128 bytes

		VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		layoutCI.setLayoutCount         = 1;
		layoutCI.pSetLayouts            = &mainDescriptorSetLayout;
		layoutCI.pushConstantRangeCount = 1;
		layoutCI.pPushConstantRanges    = &pcRange;
		if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &mainPipelineLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create main pipeline layout!");

		VkPipelineRenderingCreateInfo renderingCI{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
		renderingCI.colorAttachmentCount    = 1;
		renderingCI.pColorAttachmentFormats = &swapChainImageFormat;
		renderingCI.depthAttachmentFormat   = depthFormat;

		VkGraphicsPipelineCreateInfo pipelineCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		pipelineCI.pNext               = &renderingCI;
		pipelineCI.stageCount          = 2;
		pipelineCI.pStages             = stages;
		pipelineCI.pVertexInputState   = &vertexInput;
		pipelineCI.pInputAssemblyState = &inputAssembly;
		pipelineCI.pViewportState      = &viewportState;
		pipelineCI.pRasterizationState = &raster;
		pipelineCI.pMultisampleState   = &msaa;
		pipelineCI.pDepthStencilState  = &depthStencil;
		pipelineCI.pColorBlendState    = &blend;
		pipelineCI.pDynamicState       = &dynState;
		pipelineCI.layout              = mainPipelineLayout;
		pipelineCI.renderPass          = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &mainPipeline) != VK_SUCCESS)
			throw std::runtime_error("failed to create main pipeline!");

		vkDestroyShaderModule(device, fragModule, nullptr);
		vkDestroyShaderModule(device, vertModule, nullptr);
	}

	// === LIGHT MARKER PIPELINE ===============================================
	// Unlit pipeline: transforms vertices, outputs flat yellow.
	// No descriptor sets — only a single mat4 push constant.

	void createMarkerPipeline()
	{
		auto vertCode = readFile(std::string(SHADER_DIR) + "/marker.vert.spv");
		auto fragCode = readFile(std::string(SHADER_DIR) + "/marker.frag.spv");
		VkShaderModule vertModule = createShaderModule(vertCode);
		VkShaderModule fragModule = createShaderModule(fragCode);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vertModule;
		stages[0].pName  = "main";
		stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = fragModule;
		stages[1].pName  = "main";

		VkVertexInputBindingDescription bindDesc{};
		bindDesc.binding   = 0;
		bindDesc.stride    = sizeof(Vertex);
		bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		VkVertexInputAttributeDescription attrDesc{};
		attrDesc.location = 0;
		attrDesc.binding  = 0;
		attrDesc.format   = VK_FORMAT_R32G32B32_SFLOAT;
		attrDesc.offset   = offsetof(Vertex, pos);

		VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
		vertexInput.vertexBindingDescriptionCount   = 1;
		vertexInput.pVertexBindingDescriptions      = &bindDesc;
		vertexInput.vertexAttributeDescriptionCount = 1;
		vertexInput.pVertexAttributeDescriptions    = &attrDesc;

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
		viewportState.viewportCount = 1;
		viewportState.scissorCount  = 1;

		VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
		dynState.dynamicStateCount = 2;
		dynState.pDynamicStates    = dynStates;

		VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
		raster.polygonMode = VK_POLYGON_MODE_FILL;
		raster.cullMode    = VK_CULL_MODE_BACK_BIT;
		raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		raster.lineWidth   = 1.0f;

		VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
		msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
		depthStencil.depthTestEnable  = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

		VkPipelineColorBlendAttachmentState blendAtt{};
		blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
		blend.attachmentCount = 1;
		blend.pAttachments    = &blendAtt;

		VkPushConstantRange pcRange{};
		pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pcRange.size       = sizeof(glm::mat4); // just MVP

		VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		layoutCI.pushConstantRangeCount = 1;
		layoutCI.pPushConstantRanges    = &pcRange;
		if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &markerPipelineLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create marker pipeline layout!");

		VkPipelineRenderingCreateInfo renderingCI{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
		renderingCI.colorAttachmentCount    = 1;
		renderingCI.pColorAttachmentFormats = &swapChainImageFormat;
		renderingCI.depthAttachmentFormat   = depthFormat;

		VkGraphicsPipelineCreateInfo pipelineCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		pipelineCI.pNext               = &renderingCI;
		pipelineCI.stageCount          = 2;
		pipelineCI.pStages             = stages;
		pipelineCI.pVertexInputState   = &vertexInput;
		pipelineCI.pInputAssemblyState = &inputAssembly;
		pipelineCI.pViewportState      = &viewportState;
		pipelineCI.pRasterizationState = &raster;
		pipelineCI.pMultisampleState   = &msaa;
		pipelineCI.pDepthStencilState  = &depthStencil;
		pipelineCI.pColorBlendState    = &blend;
		pipelineCI.pDynamicState       = &dynState;
		pipelineCI.layout              = markerPipelineLayout;
		pipelineCI.renderPass          = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &markerPipeline) != VK_SUCCESS)
			throw std::runtime_error("failed to create marker pipeline!");

		vkDestroyShaderModule(device, fragModule, nullptr);
		vkDestroyShaderModule(device, vertModule, nullptr);
	}

	// === GEOMETRY BUFFERS ====================================================

	template<typename V, typename I>
	void uploadMesh(const std::vector<V>& verts, const std::vector<I>& idxs,
		VkBuffer& vb, VkDeviceMemory& vbm, VkBuffer& ib, VkDeviceMemory& ibm)
	{
		{
			VkDeviceSize sz = sizeof(V) * verts.size();
			VkBuffer stg; VkDeviceMemory stgMem;
			createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stg, stgMem);
			void* d; vkMapMemory(device, stgMem, 0, sz, 0, &d); std::memcpy(d, verts.data(), sz); vkUnmapMemory(device, stgMem);
			createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vb, vbm);
			copyBuffer(stg, vb, sz);
			vkDestroyBuffer(device, stg, nullptr); vkFreeMemory(device, stgMem, nullptr);
		}
		{
			VkDeviceSize sz = sizeof(I) * idxs.size();
			VkBuffer stg; VkDeviceMemory stgMem;
			createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stg, stgMem);
			void* d; vkMapMemory(device, stgMem, 0, sz, 0, &d); std::memcpy(d, idxs.data(), sz); vkUnmapMemory(device, stgMem);
			createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ib, ibm);
			copyBuffer(stg, ib, sz);
			vkDestroyBuffer(device, stg, nullptr); vkFreeMemory(device, stgMem, nullptr);
		}
	}

	// === UNIFORM BUFFERS =====================================================

	void createUniformBuffers()
	{
		VkDeviceSize size = sizeof(SceneUBO);
		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				uniformBuffers[i], uniformBuffersMemory[i]);
			vkMapMemory(device, uniformBuffersMemory[i], 0, size, 0, &uniformBuffersMapped[i]);
		}
	}

	void updateUniformBuffer(uint32_t frameIndex)
	{
		float t = static_cast<float>(glfwGetTime());

		// Light orbits low (Y=4, radius=7) so the marker stays within the camera's view
		// and casts long dramatic shadows across the ground plane.
		lightPos = glm::vec3(
			7.0f * std::cos(t * 0.25f),
			4.0f,
			7.0f * std::sin(t * 0.25f));
		glm::vec3 lightTarget = glm::vec3(0.0f);

		lightDir = glm::normalize(lightPos - lightTarget); // toward-light direction

		glm::mat4 lightView = glm::lookAt(lightPos, lightTarget, glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 lightProj = glm::ortho(-5.0f, 5.0f, -5.0f, 5.0f, 0.5f, 28.0f); // tighter = better texel density
		lightProj[1][1] *= -1.0f; // Vulkan Y-flip

		lightSpaceMatrix = lightProj * lightView;

		SceneUBO ubo{};
		ubo.lightDir        = glm::vec4(lightDir, 0.0f);
		ubo.lightColor       = glm::vec4(1.0f, 0.95f, 0.85f, 2.0f); // warm, intensity 2.0
		ubo.viewPos          = glm::vec4(0.0f, 7.0f, 12.0f, 0.0f);
		ubo.lightSpaceMatrix = lightSpaceMatrix;
		ubo.ambient          = glm::vec4(0.7f, 0.75f, 0.85f, 0.40f); // enough ambient to see the cube's shadowed faces

		std::memcpy(uniformBuffersMapped[frameIndex], &ubo, sizeof(ubo));
	}

	// === DESCRIPTOR POOL & SETS ==============================================

	void createDescriptorPool()
	{
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)};
		poolSizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2};

		VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		ci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		ci.pPoolSizes    = poolSizes.data();
		ci.maxSets       = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		if (vkCreateDescriptorPool(device, &ci, nullptr, &descriptorPool) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor pool!");
	}

	void createDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, mainDescriptorSetLayout);
		VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		ai.descriptorPool     = descriptorPool;
		ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		ai.pSetLayouts        = layouts.data();
		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &ai, descriptorSets.data()) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate descriptor sets!");

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			VkDescriptorBufferInfo bufInfo{};
			bufInfo.buffer = uniformBuffers[i];
			bufInfo.range  = sizeof(SceneUBO);

			VkDescriptorImageInfo diffuseInfo{};
			diffuseInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			diffuseInfo.imageView   = diffuseImageView;
			diffuseInfo.sampler     = diffuseSampler;

			VkDescriptorImageInfo shadowInfo{};
			shadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			shadowInfo.imageView   = shadowImageViews[i];  // per-frame view
			shadowInfo.sampler     = shadowSampler;

			std::array<VkWriteDescriptorSet, 3> writes{};
			writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
			writes[0].dstSet          = descriptorSets[i];
			writes[0].dstBinding      = 0;
			writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			writes[0].descriptorCount = 1;
			writes[0].pBufferInfo     = &bufInfo;

			writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
			writes[1].dstSet          = descriptorSets[i];
			writes[1].dstBinding      = 1;
			writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			writes[1].descriptorCount = 1;
			writes[1].pImageInfo      = &diffuseInfo;

			writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
			writes[2].dstSet          = descriptorSets[i];
			writes[2].dstBinding      = 2;
			writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			writes[2].descriptorCount = 1;
			writes[2].pImageInfo      = &shadowInfo;

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		}
	}

	// === HOOKS ===============================================================

protected:
	void onInitBeforeCommandPool() override
	{
		createDepthResources();
		createShadowResources();
		createMainDescriptorSetLayout();
		createShadowPipeline();
		createMainPipeline();
		createMarkerPipeline();
	}

	void onInitAfterCommandPool() override
	{
		buildCubeMesh(cubeVertices, cubeIndices);

		uploadMesh(groundVertices, groundIndices,
			groundVertexBuffer, groundVertexBufferMemory,
			groundIndexBuffer,  groundIndexBufferMemory);
		uploadMesh(cubeVertices, cubeIndices,
			cubeVertexBuffer, cubeVertexBufferMemory,
			cubeIndexBuffer,  cubeIndexBufferMemory);

		createDiffuseTexture();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
	}

	void onBeforeRecord(uint32_t frameIndex) override
	{
		updateUniformBuffer(frameIndex);
	}

	void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) override
	{
		VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
			throw std::runtime_error("failed to begin recording command buffer!");

		// Per-object model matrices
		glm::mat4 groundModel = glm::mat4(1.0f);
		// Cube sits on the ground plane (bottom face at Y=0)
		glm::mat4 cubeModel   = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 0.0f));

		// Camera matrices (same every object, used in main pass)
		float aspect = static_cast<float>(swapChainExtent.width) /
		               static_cast<float>(swapChainExtent.height);
		glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 7.0f, 12.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
		proj[1][1] *= -1.0f;

		// =====================================================================
		// PASS 1 — Shadow pass
		// Render scene geometry from the light's POV into the shadow depth map.
		// =====================================================================

		// Transition this frame's shadow image: UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL (write)
		transitionImageLayout(cmd, shadowImages[currentFrame],
			VK_IMAGE_LAYOUT_UNDEFINED,                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,           VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			VK_IMAGE_ASPECT_DEPTH_BIT);

		VkRenderingAttachmentInfo shadowDepthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
		shadowDepthAtt.imageView              = shadowImageViews[currentFrame];
		shadowDepthAtt.imageLayout            = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
		shadowDepthAtt.loadOp                 = VK_ATTACHMENT_LOAD_OP_CLEAR;
		shadowDepthAtt.storeOp                = VK_ATTACHMENT_STORE_OP_STORE;
		shadowDepthAtt.clearValue.depthStencil = {1.0f, 0};

		VkRenderingInfo shadowRI{VK_STRUCTURE_TYPE_RENDERING_INFO};
		shadowRI.renderArea   = {{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
		shadowRI.layerCount   = 1;
		shadowRI.pDepthAttachment = &shadowDepthAtt;
		// no color attachments

		vkCmdBeginRendering(cmd, &shadowRI);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);

		VkViewport shadowVP{};
		shadowVP.width    = static_cast<float>(SHADOW_MAP_SIZE);
		shadowVP.height   = static_cast<float>(SHADOW_MAP_SIZE);
		shadowVP.minDepth = 0.0f;
		shadowVP.maxDepth = 1.0f;
		vkCmdSetViewport(cmd, 0, 1, &shadowVP);
		VkRect2D shadowScissor{{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
		vkCmdSetScissor(cmd, 0, 1, &shadowScissor);

		// Draw ground (shadow pass)
		{
			VkBuffer bufs[] = {groundVertexBuffer}; VkDeviceSize offs[] = {0};
			vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offs);
			vkCmdBindIndexBuffer(cmd, groundIndexBuffer, 0, VK_INDEX_TYPE_UINT16);
			ShadowPushConstants spc{lightSpaceMatrix * groundModel};
			vkCmdPushConstants(cmd, shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(spc), &spc);
			vkCmdDrawIndexed(cmd, static_cast<uint32_t>(groundIndices.size()), 1, 0, 0, 0);
		}
		// Draw cube (shadow pass)
		{
			VkBuffer bufs[] = {cubeVertexBuffer}; VkDeviceSize offs[] = {0};
			vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offs);
			vkCmdBindIndexBuffer(cmd, cubeIndexBuffer, 0, VK_INDEX_TYPE_UINT16);
			ShadowPushConstants spc{lightSpaceMatrix * cubeModel};
			vkCmdPushConstants(cmd, shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(spc), &spc);
			vkCmdDrawIndexed(cmd, static_cast<uint32_t>(cubeIndices.size()), 1, 0, 0, 0);
		}
		vkCmdEndRendering(cmd);

		// =====================================================================
		// Barrier: shadow map DEPTH_ATTACHMENT_WRITE → SHADER_READ
		// The fragment shader in pass 2 will sample shadowImage.
		// =====================================================================
		{
			VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
			barrier.srcStageMask        = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
			                              VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
			barrier.srcAccessMask       = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			barrier.dstStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
			barrier.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT;
			barrier.oldLayout           = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
			barrier.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.image               = shadowImages[currentFrame];
			barrier.subresourceRange    = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

			VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
			dep.imageMemoryBarrierCount = 1;
			dep.pImageMemoryBarriers    = &barrier;
			vkCmdPipelineBarrier2(cmd, &dep);
		}

		// =====================================================================
		// PASS 2 — Main pass
		// Render with full Blinn-Phong + shadow factor sampled from pass 1.
		// =====================================================================

		transitionImageLayout(cmd, swapChainImages[imageIndex],
			VK_IMAGE_LAYOUT_UNDEFINED,                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,            VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

		transitionImageLayout(cmd, depthImage,
			VK_IMAGE_LAYOUT_UNDEFINED,                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,           VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			VK_IMAGE_ASPECT_DEPTH_BIT);

		VkRenderingAttachmentInfo colorAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
		colorAtt.imageView   = swapChainImageViews[imageIndex];
		colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
		colorAtt.clearValue  = {{{0.53f, 0.71f, 0.87f, 1.0f}}}; // sky blue

		VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
		depthAtt.imageView               = depthImageView;
		depthAtt.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
		depthAtt.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAtt.storeOp                 = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAtt.clearValue.depthStencil = {1.0f, 0};

		VkRenderingInfo mainRI{VK_STRUCTURE_TYPE_RENDERING_INFO};
		mainRI.renderArea           = {{0, 0}, swapChainExtent};
		mainRI.layerCount           = 1;
		mainRI.colorAttachmentCount = 1;
		mainRI.pColorAttachments    = &colorAtt;
		mainRI.pDepthAttachment     = &depthAtt;

		vkCmdBeginRendering(cmd, &mainRI);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mainPipeline);

		VkViewport vp{};
		vp.width    = static_cast<float>(swapChainExtent.width);
		vp.height   = static_cast<float>(swapChainExtent.height);
		vp.minDepth = 0.0f;
		vp.maxDepth = 1.0f;
		vkCmdSetViewport(cmd, 0, 1, &vp);
		VkRect2D scissor{{0, 0}, swapChainExtent};
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
			mainPipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

		// Draw ground (main pass)
		{
			VkBuffer bufs[] = {groundVertexBuffer}; VkDeviceSize offs[] = {0};
			vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offs);
			vkCmdBindIndexBuffer(cmd, groundIndexBuffer, 0, VK_INDEX_TYPE_UINT16);
			PushConstants pc{proj * view * groundModel, groundModel};
			vkCmdPushConstants(cmd, mainPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
			vkCmdDrawIndexed(cmd, static_cast<uint32_t>(groundIndices.size()), 1, 0, 0, 0);
		}
		// Draw cube (main pass)
		{
			VkBuffer bufs[] = {cubeVertexBuffer}; VkDeviceSize offs[] = {0};
			vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offs);
			vkCmdBindIndexBuffer(cmd, cubeIndexBuffer, 0, VK_INDEX_TYPE_UINT16);
			PushConstants pc{proj * view * cubeModel, cubeModel};
			vkCmdPushConstants(cmd, mainPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
			vkCmdDrawIndexed(cmd, static_cast<uint32_t>(cubeIndices.size()), 1, 0, 0, 0);
		}

		// Draw light marker — small yellow cube at the light's world position.
		// Uses the unlit marker pipeline (no descriptor sets, just MVP push constant).
		{
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, markerPipeline);

			VkBuffer bufs[] = {cubeVertexBuffer}; VkDeviceSize offs[] = {0};
			vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offs);
			vkCmdBindIndexBuffer(cmd, cubeIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

			// 0.6 m bright yellow cube at the light's world position
			glm::mat4 markerModel = glm::translate(glm::mat4(1.0f), lightPos);
			markerModel = glm::scale(markerModel, glm::vec3(0.6f));

			glm::mat4 markerMVP = proj * view * markerModel;
			vkCmdPushConstants(cmd, markerPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &markerMVP);
			vkCmdDrawIndexed(cmd, static_cast<uint32_t>(cubeIndices.size()), 1, 0, 0, 0);
		}

		vkCmdEndRendering(cmd);

		transitionImageLayout(cmd, swapChainImages[imageIndex],
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,          VK_ACCESS_2_NONE);

		if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
			throw std::runtime_error("failed to record command buffer!");
	}

	void onCleanupSwapChain() override { cleanupDepthResources(); }
	void onRecreateSwapChain() override { createDepthResources(); }

	void onCleanup() override
	{
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, mainDescriptorSetLayout, nullptr);

		destroyDiffuseTexture();
		destroyShadowResources();

		vkDestroyBuffer(device, cubeIndexBuffer, nullptr);
		vkFreeMemory(device, cubeIndexBufferMemory, nullptr);
		vkDestroyBuffer(device, cubeVertexBuffer, nullptr);
		vkFreeMemory(device, cubeVertexBufferMemory, nullptr);
		vkDestroyBuffer(device, groundIndexBuffer, nullptr);
		vkFreeMemory(device, groundIndexBufferMemory, nullptr);
		vkDestroyBuffer(device, groundVertexBuffer, nullptr);
		vkFreeMemory(device, groundVertexBufferMemory, nullptr);

		vkDestroyPipeline(device, markerPipeline, nullptr);
		vkDestroyPipelineLayout(device, markerPipelineLayout, nullptr);
		vkDestroyPipeline(device, mainPipeline, nullptr);
		vkDestroyPipelineLayout(device, mainPipelineLayout, nullptr);
		vkDestroyPipeline(device, shadowPipeline, nullptr);
		vkDestroyPipelineLayout(device, shadowPipelineLayout, nullptr);
	}
};

// ===========================================================================

int main()
{
	ShadowMappingApp app;
	try
	{
		app.run(1024, 768, "vk-examples: 08 Shadow Mapping (Vulkan 1.3)");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Fatal Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
