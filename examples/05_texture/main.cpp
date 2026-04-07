
#include "../common/vk_common.h"

// ---------------------------------------------------------------------------
// 05 – Texture
//
// Builds on example 03 (point lights) and introduces GPU textures:
//   • VkImage / VkDeviceMemory / VkImageView  for a 2-D colour texture
//   • VkSampler                               for filtered sampling
//   • Combined image sampler descriptor       (set 0, binding 1)
//   • UV coordinates in the vertex layout
//
// The texture is a 256×256 Concentric Rings generated on the CPU and uploaded
// via a staging buffer — no external image-loading library is required.
// The point-light Blinn-Phong shading from example 03 is applied on top.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU data structures
// ---------------------------------------------------------------------------

struct PointLight
{
	glm::vec4 position; // xyz = pos,   w = unused
	glm::vec4 color;	// xyz = color, w = intensity
};

struct LightUBO
{
	PointLight lights[3];
	glm::vec4 viewPos; // xyz = camera
	glm::vec4 ambient; // xyz = color, w = strength
};

struct PushConstants
{
	glm::mat4 mvp;
	glm::mat4 model;
};

// ---------------------------------------------------------------------------
// Vertex data – ground plane with UV coordinates
// ---------------------------------------------------------------------------

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 uv;
};

// UVs tile 5× across the 20 m ground, giving one checker cell every ~0.5 m
// (the 256×256 texture has 32 px cells → 8 cells per tile → 40 cells total).
static const std::vector<Vertex> groundVertices = {
	{{-10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
	{{10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 0.0f}},
	{{10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 5.0f}},
	{{-10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 5.0f}},
};

static const std::vector<uint16_t> groundIndices = {
	0, 1, 2, 2, 3, 0,
};

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

class TexturedGroundApp : public VkAppBase
{
private:
	// -- Depth buffer --
	VkImage depthImage = VK_NULL_HANDLE;
	VkDeviceMemory depthMemory = VK_NULL_HANDLE;
	VkImageView depthImageView = VK_NULL_HANDLE;
	VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

	// -- Texture --
	VkImage textureImage = VK_NULL_HANDLE;
	VkDeviceMemory textureMemory = VK_NULL_HANDLE;
	VkImageView textureImageView = VK_NULL_HANDLE;
	VkSampler textureSampler = VK_NULL_HANDLE;

	// -- Pipeline --
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline graphicsPipeline = VK_NULL_HANDLE;

	// -- Descriptor system --
	VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> descriptorSets; // per frame-in-flight

	// -- Uniform buffers (per frame-in-flight, persistently mapped) --
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	// -- Geometry --
	VkBuffer vertexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
	VkBuffer indexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;

	// === DEPTH BUFFER ========================================================

	void createDepthResources()
	{
		VkImageCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ci.imageType = VK_IMAGE_TYPE_2D;
		ci.format = depthFormat;
		ci.extent = {swapChainExtent.width, swapChainExtent.height, 1};
		ci.mipLevels = 1;
		ci.arrayLayers = 1;
		ci.samples = VK_SAMPLE_COUNT_1_BIT;
		ci.tiling = VK_IMAGE_TILING_OPTIMAL;
		ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		if (vkCreateImage(device, &ci, nullptr, &depthImage) != VK_SUCCESS)
			throw std::runtime_error("failed to create depth image!");

		VkMemoryRequirements memReq;
		vkGetImageMemoryRequirements(device, depthImage, &memReq);
		VkMemoryAllocateInfo ai{};
		ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize = memReq.size;
		ai.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (vkAllocateMemory(device, &ai, nullptr, &depthMemory) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate depth memory!");
		vkBindImageMemory(device, depthImage, depthMemory, 0);

		VkImageViewCreateInfo vi{};
		vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		vi.image = depthImage;
		vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
		vi.format = depthFormat;
		vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
		if (vkCreateImageView(device, &vi, nullptr, &depthImageView) != VK_SUCCESS)
			throw std::runtime_error("failed to create depth image view!");
	}

	void cleanupDepthResources()
	{
		vkDestroyImageView(device, depthImageView, nullptr);
		depthImageView = VK_NULL_HANDLE;
		vkDestroyImage(device, depthImage, nullptr);
		depthImage = VK_NULL_HANDLE;
		vkFreeMemory(device, depthMemory, nullptr);
		depthMemory = VK_NULL_HANDLE;
	}

	// === TEXTURE =============================================================

	// Create a VkImage backed by device memory.
	void createImage(uint32_t w, uint32_t h, VkFormat format, VkImageTiling tiling,
		VkImageUsageFlags usage, VkMemoryPropertyFlags memProps,
		VkImage& image, VkDeviceMemory& memory)
	{
		VkImageCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ci.imageType = VK_IMAGE_TYPE_2D;
		ci.format = format;
		ci.extent = {w, h, 1};
		ci.mipLevels = 1;
		ci.arrayLayers = 1;
		ci.samples = VK_SAMPLE_COUNT_1_BIT;
		ci.tiling = tiling;
		ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		ci.usage = usage;
		ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (vkCreateImage(device, &ci, nullptr, &image) != VK_SUCCESS)
			throw std::runtime_error("failed to create image!");

		VkMemoryRequirements req;
		vkGetImageMemoryRequirements(device, image, &req);
		VkMemoryAllocateInfo ai{};
		ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize = req.size;
		ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, memProps);
		if (vkAllocateMemory(device, &ai, nullptr, &memory) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate image memory!");
		vkBindImageMemory(device, image, memory, 0);
	}

	// Copy a buffer into a VkImage (image must be in TRANSFER_DST_OPTIMAL layout).
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t w, uint32_t h)
	{
		VkCommandBufferAllocateInfo ai{};
		ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ai.commandPool = commandPool;
		ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ai.commandBufferCount = 1;

		VkCommandBuffer cmd;
		vkAllocateCommandBuffers(device, &ai, &cmd);

		VkCommandBufferBeginInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &bi);

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		region.imageOffset = {0, 0, 0};
		region.imageExtent = {w, h, 1};
		vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		vkEndCommandBuffer(cmd);

		VkSubmitInfo si{};
		si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		si.commandBufferCount = 1;
		si.pCommandBuffers = &cmd;
		vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);
		vkFreeCommandBuffers(device, commandPool, 1, &cmd);
	}

	// Generate a 256×256 Concentric Rings on the CPU, upload it to the GPU, and
	// create a VkImageView + VkSampler ready for use in a descriptor set.
	void createTexture()
	{
		constexpr uint32_t TEX_W = 256;
		constexpr uint32_t TEX_H = 256;
		constexpr uint32_t CELL = 32; // checker cell size in pixels

		// --- Generate pixels (RGBA8 sRGB) ---
		std::vector<uint8_t> pixels(TEX_W * TEX_H * 4);
		for (uint32_t y = 0; y < TEX_H; ++y)
		{
			for (uint32_t x = 0; x < TEX_W; ++x)
			{
				float centerX = TEX_W / 2.0f;
				float centerY = TEX_H / 2.0f;
				float dx = x - centerX;
				float dy = y - centerY;

				// Calculate distance from center
				float distance = std::sqrt(dx * dx + dy * dy);

				// Alternate colors based on distance and CELL size
				bool ring = (static_cast<int>(distance) / CELL) % 2 == 0;
				uint8_t v = ring ? 210 : 55;

				uint8_t* p = &pixels[(y * TEX_W + x) * 4];
				p[0] = v;
				p[1] = v;
				p[2] = v;
				p[3] = 255;
			}
		}

		// --- Upload via staging buffer ---
		VkDeviceSize imageSize = TEX_W * TEX_H * 4;
		VkBuffer staging;
		VkDeviceMemory stagingMem;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);

		void* data;
		vkMapMemory(device, stagingMem, 0, imageSize, 0, &data);
		std::memcpy(data, pixels.data(), imageSize);
		vkUnmapMemory(device, stagingMem);

		// --- Create the device-local VkImage ---
		createImage(TEX_W, TEX_H, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureMemory);

		// --- Transition: UNDEFINED -> TRANSFER_DST_OPTIMAL ---
		{
			VkCommandBufferAllocateInfo ai{};
			ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			ai.commandPool = commandPool;
			ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			ai.commandBufferCount = 1;
			VkCommandBuffer cmd;
			vkAllocateCommandBuffers(device, &ai, &cmd);

			VkCommandBufferBeginInfo bi{};
			bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(cmd, &bi);

			transitionImageLayout(cmd, textureImage,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
				VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

			vkEndCommandBuffer(cmd);

			VkSubmitInfo si{};
			si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			si.commandBufferCount = 1;
			si.pCommandBuffers = &cmd;
			vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
			vkQueueWaitIdle(graphicsQueue);
			vkFreeCommandBuffers(device, commandPool, 1, &cmd);
		}

		// --- Copy pixels into the image ---
		copyBufferToImage(staging, textureImage, TEX_W, TEX_H);

		// --- Transition: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL ---
		{
			VkCommandBufferAllocateInfo ai{};
			ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			ai.commandPool = commandPool;
			ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			ai.commandBufferCount = 1;
			VkCommandBuffer cmd;
			vkAllocateCommandBuffers(device, &ai, &cmd);

			VkCommandBufferBeginInfo bi{};
			bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(cmd, &bi);

			transitionImageLayout(cmd, textureImage,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

			vkEndCommandBuffer(cmd);

			VkSubmitInfo si{};
			si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			si.commandBufferCount = 1;
			si.pCommandBuffers = &cmd;
			vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
			vkQueueWaitIdle(graphicsQueue);
			vkFreeCommandBuffers(device, commandPool, 1, &cmd);
		}

		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);

		// --- VkImageView ---
		VkImageViewCreateInfo vi{};
		vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		vi.image = textureImage;
		vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
		vi.format = VK_FORMAT_R8G8B8A8_SRGB;
		vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		if (vkCreateImageView(device, &vi, nullptr, &textureImageView) != VK_SUCCESS)
			throw std::runtime_error("failed to create texture image view!");

		// --- VkSampler ---
		VkSamplerCreateInfo samplerCI{};
		samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerCI.magFilter = VK_FILTER_LINEAR;
		samplerCI.minFilter = VK_FILTER_LINEAR;
		samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCI.anisotropyEnable = VK_FALSE; // no extra feature needed
		samplerCI.maxAnisotropy = 1.0f;
		samplerCI.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerCI.unnormalizedCoordinates = VK_FALSE;
		samplerCI.compareEnable = VK_FALSE;
		samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		if (vkCreateSampler(device, &samplerCI, nullptr, &textureSampler) != VK_SUCCESS)
			throw std::runtime_error("failed to create texture sampler!");
	}

	void destroyTexture()
	{
		vkDestroySampler(device, textureSampler, nullptr);
		textureSampler = VK_NULL_HANDLE;
		vkDestroyImageView(device, textureImageView, nullptr);
		textureImageView = VK_NULL_HANDLE;
		vkDestroyImage(device, textureImage, nullptr);
		textureImage = VK_NULL_HANDLE;
		vkFreeMemory(device, textureMemory, nullptr);
		textureMemory = VK_NULL_HANDLE;
	}

	// === DESCRIPTOR SET LAYOUT ===============================================

	void createDescriptorSetLayout()
	{
		// Binding 0 – uniform buffer (lights + camera)
		VkDescriptorSetLayoutBinding uboBinding{};
		uboBinding.binding = 0;
		uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboBinding.descriptorCount = 1;
		uboBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		// Binding 1 – combined image sampler (diffuse texture)
		VkDescriptorSetLayoutBinding samplerBinding{};
		samplerBinding.binding = 1;
		samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerBinding.descriptorCount = 1;
		samplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboBinding, samplerBinding};

		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = static_cast<uint32_t>(bindings.size());
		ci.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &descriptorSetLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor set layout!");
	}

	// === GRAPHICS PIPELINE ===================================================

	void createGraphicsPipeline()
	{
		auto vertCode = readFile(std::string(SHADER_DIR) + "/tex.vert.spv");
		auto fragCode = readFile(std::string(SHADER_DIR) + "/tex.frag.spv");
		VkShaderModule vertModule = createShaderModule(vertCode);
		VkShaderModule fragModule = createShaderModule(fragCode);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vertModule;
		stages[0].pName = "main";
		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = fragModule;
		stages[1].pName = "main";

		// Vertex input: position + normal + uv
		VkVertexInputBindingDescription bindDesc{};
		bindDesc.binding = 0;
		bindDesc.stride = sizeof(Vertex);
		bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		std::array<VkVertexInputAttributeDescription, 3> attrDescs{};
		attrDescs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
		attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
		attrDescs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)};

		VkPipelineVertexInputStateCreateInfo vertexInput{};
		vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInput.vertexBindingDescriptionCount = 1;
		vertexInput.pVertexBindingDescriptions = &bindDesc;
		vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
		vertexInput.pVertexAttributeDescriptions = attrDescs.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynState{};
		dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynState.dynamicStateCount = 2;
		dynState.pDynamicStates = dynStates;

		VkPipelineRasterizationStateCreateInfo raster{};
		raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		raster.polygonMode = VK_POLYGON_MODE_FILL;
		raster.cullMode = VK_CULL_MODE_BACK_BIT;
		raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		raster.lineWidth = 1.0f;

		VkPipelineMultisampleStateCreateInfo msaa{};
		msaa.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

		VkPipelineColorBlendAttachmentState blendAtt{};
		blendAtt.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo blend{};
		blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blend.attachmentCount = 1;
		blend.pAttachments = &blendAtt;

		VkPushConstantRange pcRange{};
		pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pcRange.offset = 0;
		pcRange.size = sizeof(PushConstants);

		VkPipelineLayoutCreateInfo layoutCI{};
		layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutCI.setLayoutCount = 1;
		layoutCI.pSetLayouts = &descriptorSetLayout;
		layoutCI.pushConstantRangeCount = 1;
		layoutCI.pPushConstantRanges = &pcRange;
		if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &pipelineLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout!");

		VkPipelineRenderingCreateInfo renderingCI{};
		renderingCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
		renderingCI.colorAttachmentCount = 1;
		renderingCI.pColorAttachmentFormats = &swapChainImageFormat;
		renderingCI.depthAttachmentFormat = depthFormat;

		VkGraphicsPipelineCreateInfo pipelineCI{};
		pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineCI.pNext = &renderingCI;
		pipelineCI.stageCount = 2;
		pipelineCI.pStages = stages;
		pipelineCI.pVertexInputState = &vertexInput;
		pipelineCI.pInputAssemblyState = &inputAssembly;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pRasterizationState = &raster;
		pipelineCI.pMultisampleState = &msaa;
		pipelineCI.pDepthStencilState = &depthStencil;
		pipelineCI.pColorBlendState = &blend;
		pipelineCI.pDynamicState = &dynState;
		pipelineCI.layout = pipelineLayout;
		pipelineCI.renderPass = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &graphicsPipeline) != VK_SUCCESS)
			throw std::runtime_error("failed to create graphics pipeline!");

		vkDestroyShaderModule(device, fragModule, nullptr);
		vkDestroyShaderModule(device, vertModule, nullptr);
	}

	// === VERTEX & INDEX BUFFERS ==============================================

	void createVertexBuffer()
	{
		VkDeviceSize size = sizeof(groundVertices[0]) * groundVertices.size();
		VkBuffer staging;
		VkDeviceMemory stagingMem;
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
		void* data;
		vkMapMemory(device, stagingMem, 0, size, 0, &data);
		std::memcpy(data, groundVertices.data(), size);
		vkUnmapMemory(device, stagingMem);
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
		copyBuffer(staging, vertexBuffer, size);
		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);
	}

	void createIndexBuffer()
	{
		VkDeviceSize size = sizeof(groundIndices[0]) * groundIndices.size();
		VkBuffer staging;
		VkDeviceMemory stagingMem;
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
		void* data;
		vkMapMemory(device, stagingMem, 0, size, 0, &data);
		std::memcpy(data, groundIndices.data(), size);
		vkUnmapMemory(device, stagingMem);
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
		copyBuffer(staging, indexBuffer, size);
		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);
	}

	// === UNIFORM BUFFERS =====================================================

	void createUniformBuffers()
	{
		VkDeviceSize size = sizeof(LightUBO);
		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i],
				uniformBuffersMemory[i]);
			vkMapMemory(device, uniformBuffersMemory[i], 0, size, 0, &uniformBuffersMapped[i]);
		}
	}

	void updateUniformBuffer(uint32_t frameIndex)
	{
		float t = static_cast<float>(glfwGetTime());

		LightUBO ubo{};

		float radius = 4.0f;
		float height = 1.5f;

		// Red light
		ubo.lights[0].position = glm::vec4(radius * std::cos(t * 0.7f), height, radius * std::sin(t * 0.7f), 0.0f);
		ubo.lights[0].color = glm::vec4(1.0f, 0.1f, 0.1f, 5.0f);

		// Green light (120 degrees offset)
		float offset1 = glm::radians(120.0f);
		ubo.lights[1].position =
			glm::vec4(radius * std::cos(t * 0.7f + offset1), height, radius * std::sin(t * 0.7f + offset1), 0.0f);
		ubo.lights[1].color = glm::vec4(0.1f, 1.0f, 0.1f, 5.0f);

		// Blue light (240 degrees offset)
		float offset2 = glm::radians(240.0f);
		ubo.lights[2].position =
			glm::vec4(radius * std::cos(t * 0.7f + offset2), height, radius * std::sin(t * 0.7f + offset2), 0.0f);
		ubo.lights[2].color = glm::vec4(0.1f, 0.2f, 1.0f, 5.0f);

		ubo.viewPos = glm::vec4(0.0f, 8.0f, 12.0f, 0.0f);
		ubo.ambient = glm::vec4(0.8f, 0.8f, 0.9f, 0.05f);

		std::memcpy(uniformBuffersMapped[frameIndex], &ubo, sizeof(ubo));
	}

	// === DESCRIPTOR POOL & SETS ==============================================

	void createDescriptorPool()
	{
		// One UBO and one combined image sampler per frame-in-flight
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		VkDescriptorPoolCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		ci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		ci.pPoolSizes = poolSizes.data();
		ci.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(device, &ci, nullptr, &descriptorPool) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor pool!");
	}

	void createDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);

		VkDescriptorSetAllocateInfo ai{};
		ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		ai.descriptorPool = descriptorPool;
		ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		ai.pSetLayouts = layouts.data();

		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &ai, descriptorSets.data()) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate descriptor sets!");

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			// Write 0: UBO
			VkDescriptorBufferInfo bufInfo{};
			bufInfo.buffer = uniformBuffers[i];
			bufInfo.offset = 0;
			bufInfo.range = sizeof(LightUBO);

			VkWriteDescriptorSet uboWrite{};
			uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			uboWrite.dstSet = descriptorSets[i];
			uboWrite.dstBinding = 0;
			uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			uboWrite.descriptorCount = 1;
			uboWrite.pBufferInfo = &bufInfo;

			// Write 1: combined image sampler (same texture for all frames)
			VkDescriptorImageInfo imgInfo{};
			imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imgInfo.imageView = textureImageView;
			imgInfo.sampler = textureSampler;

			VkWriteDescriptorSet samplerWrite{};
			samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			samplerWrite.dstSet = descriptorSets[i];
			samplerWrite.dstBinding = 1;
			samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			samplerWrite.descriptorCount = 1;
			samplerWrite.pImageInfo = &imgInfo;

			std::array<VkWriteDescriptorSet, 2> writes = {uboWrite, samplerWrite};
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		}
	}

	// -----------------------------------------------------------------------
	// Virtual hook implementations
	// -----------------------------------------------------------------------

protected:
	void onInitBeforeCommandPool() override
	{
		createDepthResources();
		createDescriptorSetLayout();
		createGraphicsPipeline();
	}

	void onInitAfterCommandPool() override
	{
		createVertexBuffer();
		createIndexBuffer();
		createTexture(); // must come before descriptor sets (imageView/sampler needed for writes)
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
		VkCommandBufferBeginInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
			throw std::runtime_error("failed to begin recording command buffer!");

		transitionImageLayout(cmd, swapChainImages[imageIndex],
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

		transitionImageLayout(cmd, depthImage,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			VK_IMAGE_ASPECT_DEPTH_BIT);

		VkRenderingAttachmentInfo colorAtt{};
		colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		colorAtt.imageView = swapChainImageViews[imageIndex];
		colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAtt.clearValue = {{{0.02f, 0.02f, 0.04f, 1.0f}}};

		VkRenderingAttachmentInfo depthAtt{};
		depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		depthAtt.imageView = depthImageView;
		depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
		depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAtt.clearValue.depthStencil = {1.0f, 0};

		VkRenderingInfo ri{};
		ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		ri.renderArea = {{0, 0}, swapChainExtent};
		ri.layerCount = 1;
		ri.colorAttachmentCount = 1;
		ri.pColorAttachments = &colorAtt;
		ri.pDepthAttachment = &depthAtt;

		vkCmdBeginRendering(cmd, &ri);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		VkViewport vp{};
		vp.width = static_cast<float>(swapChainExtent.width);
		vp.height = static_cast<float>(swapChainExtent.height);
		vp.minDepth = 0.0f;
		vp.maxDepth = 1.0f;
		vkCmdSetViewport(cmd, 0, 1, &vp);
		VkRect2D scissor{{0, 0}, swapChainExtent};
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		// Bind descriptor set (UBO + texture sampler)
		vkCmdBindDescriptorSets(
			cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

		VkBuffer buffers[] = {vertexBuffer};
		VkDeviceSize offsets[] = {0};
		vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
		vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

		float aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);

		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view =
			glm::lookAt(glm::vec3(0.0f, 8.0f, 12.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
		proj[1][1] *= -1.0f;

		PushConstants pc{};
		pc.mvp = proj * view * model;
		pc.model = model;

		vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pc);
		vkCmdDrawIndexed(cmd, static_cast<uint32_t>(groundIndices.size()), 1, 0, 0, 0);

		vkCmdEndRendering(cmd);

		transitionImageLayout(cmd, swapChainImages[imageIndex],
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE);

		if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
			throw std::runtime_error("failed to record command buffer!");
	}

	void onCleanupSwapChain() override
	{
		cleanupDepthResources();
	}

	void onRecreateSwapChain() override
	{
		createDepthResources();
	}

	void onCleanup() override
	{
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		destroyTexture();

		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);
		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
	}
};

// ===========================================================================

int main()
{
	TexturedGroundApp app;
	try
	{
		app.run(1024, 768, "vk-examples: 05 Texture (Vulkan 1.3)");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Fatal Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
