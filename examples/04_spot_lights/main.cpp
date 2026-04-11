
#include "../common/vk_common.h"
#include "../common/vk_pipelines.h"
#include "../common/vk_descriptors.h"

// ---------------------------------------------------------------------------
// 04 – Spot Lights
//
// Extends the common base to add:
//   • Depth buffer
//   • Descriptor set layout + pool + sets (UBO binding)
//   • Uniform buffer objects (per frame-in-flight, persistently mapped)
//   • Graphics pipeline (vertex + fragment shader, push constants + UBO)
//   • Vertex and index buffers (ground plane)
//   • Blinn-Phong shading with 3 spot lights (inner/outer cone angles)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU data structures (must match shader layout exactly)
// ---------------------------------------------------------------------------

struct SpotLight
{
	glm::vec4 position;	 // xyz = pos,       w = unused
	glm::vec4 direction; // xyz = direction,  w = unused
	glm::vec4 color;	 // xyz = color,      w = intensity
	glm::vec4 cutoffs;	 // x = cos(inner),   y = cos(outer), zw = unused
};

struct LightUBO
{
	SpotLight lights[3];
	glm::vec4 viewPos; // xyz = camera
	glm::vec4 ambient; // xyz = color, w = strength
};

struct PushConstants
{
	glm::mat4 mvp;
	glm::mat4 model;
};

// ---------------------------------------------------------------------------
// Vertex data – ground plane on XZ, facing Y+
// ---------------------------------------------------------------------------

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 normal;
};

static const std::vector<Vertex> groundVertices = {
	{{-10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}},
	{{10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}},
	{{10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}},
	{{-10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}},
};

static const std::vector<uint16_t> groundIndices = {
	0, 1, 2, 2, 3, 0,
};

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

class SpotLightsApp : public VkAppBase
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

	// === DESCRIPTOR SET LAYOUT ===============================================

	void createDescriptorSetLayout()
	{
		descriptorSetLayout = DescriptorLayoutBuilder()
			.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(device);
	}

	// === GRAPHICS PIPELINE ===================================================

	void createGraphicsPipeline()
	{
		VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/spot.vert.spv"));
		VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/spot.frag.spv"));

		auto [pipeline, layout] = GraphicsPipelineBuilder(device)
			.vertShader(vert)
			.fragShader(frag)
			.vertexBinding<Vertex>()
			.vertexAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
			.vertexAttribute(1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
			.pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(PushConstants))
			.descriptorSetLayout(descriptorSetLayout)
			.colorFormat(swapChainImageFormat)
			.depthFormat(depthFormat)
			.build();

		graphicsPipeline = pipeline;
		pipelineLayout   = layout;
		vkDestroyShaderModule(device, frag, nullptr);
		vkDestroyShaderModule(device, vert, nullptr);
	}

	void updateUniformBuffer(uint32_t frameIndex)
	{
		float t = static_cast<float>(glfwGetTime());

		LightUBO ubo{};

		float height = 5.0f;
		float spacing = 4.0f; // distance between lights on X axis
		float innerAngle = glm::radians(18.0f);
		float outerAngle = glm::radians(25.0f);

		// All three lights point straight down (perpendicular to the ground)
		glm::vec3 downDir(0.0f, -1.0f, 0.0f);

		// Slow drift on Z so the cones sweep gently but stay separated
		float drift = std::sin(t * 0.4f) * 2.0f;

		// Red – left
		ubo.lights[0].position = glm::vec4(-spacing, height, drift, 0.0f);
		ubo.lights[0].direction = glm::vec4(downDir, 0.0f);
		ubo.lights[0].color = glm::vec4(1.0f, 0.1f, 0.1f, 10.0f);
		ubo.lights[0].cutoffs = glm::vec4(std::cos(innerAngle), std::cos(outerAngle), 0.0f, 0.0f);

		// Green – center
		ubo.lights[1].position = glm::vec4(0.0f, height, -drift, 0.0f);
		ubo.lights[1].direction = glm::vec4(downDir, 0.0f);
		ubo.lights[1].color = glm::vec4(0.1f, 1.0f, 0.1f, 10.0f);
		ubo.lights[1].cutoffs = glm::vec4(std::cos(innerAngle), std::cos(outerAngle), 0.0f, 0.0f);

		// Blue – right
		ubo.lights[2].position = glm::vec4(spacing, height, drift * 0.5f, 0.0f);
		ubo.lights[2].direction = glm::vec4(downDir, 0.0f);
		ubo.lights[2].color = glm::vec4(0.2f, 0.3f, 1.0f, 10.0f);
		ubo.lights[2].cutoffs = glm::vec4(std::cos(innerAngle), std::cos(outerAngle), 0.0f, 0.0f);

		ubo.viewPos = glm::vec4(0.0f, 8.0f, 14.0f, 0.0f);
		ubo.ambient = glm::vec4(0.7f, 0.7f, 0.8f, 0.03f);

		std::memcpy(uniformBuffersMapped[frameIndex], &ubo, sizeof(ubo));
	}

	// === DESCRIPTOR POOL & SETS ==============================================

	void createDescriptorPool()
	{
		descriptorPool = DescriptorPoolBuilder()
			.addSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES_IN_FLIGHT)
			.build(device, MAX_FRAMES_IN_FLIGHT);
	}

	void createDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		VkDescriptorSetAllocateInfo ai{};
		ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		ai.descriptorPool     = descriptorPool;
		ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
		ai.pSetLayouts        = layouts.data();
		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		vkAllocateDescriptorSets(device, &ai, descriptorSets.data());

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
			DescriptorWriter()
				.writeBuffer(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					uniformBuffers[i], 0, sizeof(LightUBO))
				.update(device, descriptorSets[i]);
	}

	// -----------------------------------------------------------------------
	// Virtual hook implementations
	// -----------------------------------------------------------------------

protected:
	void onInitBeforeCommandPool() override
	{
		createDepthResources(depthFormat, depthImage, depthMemory, depthImageView);
		createDescriptorSetLayout();
		createGraphicsPipeline();
	}

	void onInitAfterCommandPool() override
	{
		uploadStagedBuffer(groundVertices.data(),
			sizeof(groundVertices[0]) * groundVertices.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexBufferMemory);
		uploadStagedBuffer(groundIndices.data(),
			sizeof(groundIndices[0]) * groundIndices.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexBufferMemory);
		createPersistentUBOs(sizeof(LightUBO), MAX_FRAMES_IN_FLIGHT,
			uniformBuffers, uniformBuffersMemory, uniformBuffersMapped);
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

		vkCmdBindDescriptorSets(
			cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

		VkBuffer buffers[] = {vertexBuffer};
		VkDeviceSize offsets[] = {0};
		vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
		vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

		float aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);

		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view =
			glm::lookAt(glm::vec3(0.0f, 8.0f, 14.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
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
		destroyDepthResources(depthImage, depthMemory, depthImageView);
	}

	void onRecreateSwapChain() override
	{
		createDepthResources(depthFormat, depthImage, depthMemory, depthImageView);
	}

	void onCleanup() override
	{
		destroyUBOs(uniformBuffers, uniformBuffersMemory);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

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
	SpotLightsApp app;
	try
	{
		app.run(1024, 768, "vk-examples: 04 Spot Lights (Vulkan 1.3)");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Fatal Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
