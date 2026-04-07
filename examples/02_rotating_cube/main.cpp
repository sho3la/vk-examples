
#include "../common/vk_common.h"

// ---------------------------------------------------------------------------
// 02 – Rotating Cube
//
// Extends the common base to add:
//   • Depth buffer
//   • Graphics pipeline (vertex + fragment shader, push constants)
//   • Vertex and index buffers (staged upload)
//   • MVP matrix via push constants
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Vertex data – a unit cube with per-face colours
// ---------------------------------------------------------------------------

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
};

static const std::vector<Vertex> cubeVertices = {
	// Front  (Z+) – red
	{{-0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
	{{-0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
	// Back   (Z-) – green
	{{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{-0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	// Top    (Y+) – blue
	{{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
	{{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
	{{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}},
	// Bottom (Y-) – yellow
	{{-0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}},
	{{0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}},
	{{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 0.0f}},
	{{-0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 0.0f}},
	// Right  (X+) – cyan
	{{0.5f, -0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}},
	{{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 1.0f}},
	{{0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 1.0f}},
	{{0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}},
	// Left   (X-) – magenta
	{{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}},
	{{-0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}},
};

static const std::vector<uint16_t> cubeIndices = {
	0, 1, 2, 2, 3, 0,
	4, 5, 6, 6, 7, 4,
	8, 9, 10, 10, 11, 8,
	12, 13, 14, 14, 15, 12,
	16, 17, 18, 18, 19, 16,
	20, 21, 22, 22, 23, 20,
};

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

class RotatingCubeApp : public VkAppBase
{
private:
	// -- Depth buffer --
	VkImage depthImage = VK_NULL_HANDLE;
	VkDeviceMemory depthMemory = VK_NULL_HANDLE;
	VkImageView depthImageView = VK_NULL_HANDLE;
	VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

	// -- Pipeline --
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline graphicsPipeline = VK_NULL_HANDLE;

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

	// === GRAPHICS PIPELINE ===================================================

	void createGraphicsPipeline()
	{
		auto vertCode = readFile(std::string(SHADER_DIR) + "/cube.vert.spv");
		auto fragCode = readFile(std::string(SHADER_DIR) + "/cube.frag.spv");
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

		VkVertexInputBindingDescription bindDesc{};
		bindDesc.binding = 0;
		bindDesc.stride = sizeof(Vertex);
		bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		std::array<VkVertexInputAttributeDescription, 2> attrDescs{};
		attrDescs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
		attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};

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
		pcRange.size = sizeof(glm::mat4);

		VkPipelineLayoutCreateInfo layoutCI{};
		layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutCI.pushConstantRangeCount = 1;
		layoutCI.pPushConstantRanges = &pcRange;
		if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &pipelineLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout!");

		// Dynamic rendering – replaces VkRenderPass (Vulkan 1.3 core)
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
		VkDeviceSize size = sizeof(cubeVertices[0]) * cubeVertices.size();
		VkBuffer staging;
		VkDeviceMemory stagingMem;
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
		void* data;
		vkMapMemory(device, stagingMem, 0, size, 0, &data);
		std::memcpy(data, cubeVertices.data(), size);
		vkUnmapMemory(device, stagingMem);
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
		copyBuffer(staging, vertexBuffer, size);
		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);
	}

	void createIndexBuffer()
	{
		VkDeviceSize size = sizeof(cubeIndices[0]) * cubeIndices.size();
		VkBuffer staging;
		VkDeviceMemory stagingMem;
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
		void* data;
		vkMapMemory(device, stagingMem, 0, size, 0, &data);
		std::memcpy(data, cubeIndices.data(), size);
		vkUnmapMemory(device, stagingMem);
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
		copyBuffer(staging, indexBuffer, size);
		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);
	}

	// -----------------------------------------------------------------------
	// Virtual hook implementations
	// -----------------------------------------------------------------------

protected:
	void onInitBeforeCommandPool() override
	{
		createDepthResources();
		createGraphicsPipeline();
	}

	void onInitAfterCommandPool() override
	{
		createVertexBuffer();
		createIndexBuffer();
	}

	void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) override
	{
		VkCommandBufferBeginInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
			throw std::runtime_error("failed to begin recording command buffer!");

		// Transition colour: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
		transitionImageLayout(cmd, swapChainImages[imageIndex],
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

		// Transition depth: UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL
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
		colorAtt.clearValue = {{{0.1f, 0.1f, 0.12f, 1.0f}}};

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

		VkBuffer buffers[] = {vertexBuffer};
		VkDeviceSize offsets[] = {0};
		vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
		vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

		// MVP using GLM – guaranteed correct column-major layout for GLSL
		float time = static_cast<float>(glfwGetTime());
		float aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);

		glm::mat4 model = glm::rotate(glm::mat4(1.0f), time, glm::vec3(0.0f, 1.0f, 0.0f)) *
						  glm::rotate(glm::mat4(1.0f), time * 0.5f, glm::vec3(1.0f, 0.0f, 0.0f));
		glm::mat4 view =
			glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
		proj[1][1] *= -1.0f; // Vulkan Y-flip

		glm::mat4 mvp = proj * view * model;

		vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mvp);
		vkCmdDrawIndexed(cmd, static_cast<uint32_t>(cubeIndices.size()), 1, 0, 0, 0);

		vkCmdEndRendering(cmd);

		// Transition colour: COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR
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
	RotatingCubeApp app;
	try
	{
		app.run(800, 600, "vk-examples: 02 Rotating Cube");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Fatal Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
