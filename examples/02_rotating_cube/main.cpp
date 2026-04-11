
#include "../common/vk_common.h"
#include "../common/vk_pipelines.h"

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

	// === GRAPHICS PIPELINE ===================================================

	void createGraphicsPipeline()
	{
		VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/cube.vert.spv"));
		VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/cube.frag.spv"));

		auto [pipeline, layout] = GraphicsPipelineBuilder(device)
			.vertShader(vert)
			.fragShader(frag)
			.vertexBinding<Vertex>()
			.vertexAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
			.vertexAttribute(1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
			.pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(glm::mat4))
			.colorFormat(swapChainImageFormat)
			.depthFormat(depthFormat)
			.build();

		graphicsPipeline = pipeline;
		pipelineLayout   = layout;
		vkDestroyShaderModule(device, frag, nullptr);
		vkDestroyShaderModule(device, vert, nullptr);
	}

	// -----------------------------------------------------------------------
	// Virtual hook implementations
	// -----------------------------------------------------------------------

protected:
	void onInitBeforeCommandPool() override
	{
		createDepthResources(depthFormat, depthImage, depthMemory, depthImageView);
		createGraphicsPipeline();
	}

	void onInitAfterCommandPool() override
	{
		uploadStagedBuffer(cubeVertices.data(),
			sizeof(cubeVertices[0]) * cubeVertices.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			vertexBuffer, vertexBufferMemory);
		uploadStagedBuffer(cubeIndices.data(),
			sizeof(cubeIndices[0]) * cubeIndices.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			indexBuffer, indexBufferMemory);
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
		destroyDepthResources(depthImage, depthMemory, depthImageView);
	}

	void onRecreateSwapChain() override
	{
		createDepthResources(depthFormat, depthImage, depthMemory, depthImageView);
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
