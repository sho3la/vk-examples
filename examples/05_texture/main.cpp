
#include "../common/vk_common.h"
#include "../common/vk_pipelines.h"
#include "../common/vk_descriptors.h"

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

	// === TEXTURE =============================================================

	// Generate a 256×256 Concentric Rings on the CPU, upload it to the GPU, and
	// create a VkImageView + VkSampler ready for use in a descriptor set.
	void createTexture()
	{
		constexpr uint32_t TEX_W = 256;
		constexpr uint32_t TEX_H = 256;
		constexpr uint32_t CELL  = 32;

		// --- Generate pixels (RGBA8 sRGB) ---
		std::vector<uint8_t> pixels(TEX_W * TEX_H * 4);
		for (uint32_t y = 0; y < TEX_H; ++y)
			for (uint32_t x = 0; x < TEX_W; ++x)
			{
				float dx  = x - TEX_W / 2.0f, dy = y - TEX_H / 2.0f;
				bool  ring = (static_cast<int>(std::sqrt(dx * dx + dy * dy)) / CELL) % 2 == 0;
				uint8_t v = ring ? 210 : 55;
				uint8_t* p = &pixels[(y * TEX_W + x) * 4];
				p[0] = p[1] = p[2] = v; p[3] = 255;
			}

		// --- Upload via staging buffer ---
		VkDeviceSize imageSize = TEX_W * TEX_H * 4;
		VkBuffer staging; VkDeviceMemory stagingMem;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
		void* ptr;
		vkMapMemory(device, stagingMem, 0, imageSize, 0, &ptr);
		std::memcpy(ptr, pixels.data(), imageSize);
		vkUnmapMemory(device, stagingMem);

		// --- Create device-local image ---
		createImage(TEX_W, TEX_H, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureMemory);

		// --- Transition UNDEFINED -> TRANSFER_DST, copy, TRANSFER_DST -> SHADER_READ ---
		{ auto cmd = beginOneTimeCommands();
		  transitionImageLayout(cmd, textureImage,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
		  endOneTimeCommands(cmd); }

		copyBufferToImage(staging, textureImage, TEX_W, TEX_H);

		{ auto cmd = beginOneTimeCommands();
		  transitionImageLayout(cmd, textureImage,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
		  endOneTimeCommands(cmd); }

		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);

		textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB);
		textureSampler   = createSampler();
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
		descriptorSetLayout = DescriptorLayoutBuilder()
			.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         VK_SHADER_STAGE_FRAGMENT_BIT)
			.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
			.build(device);
	}

	// === GRAPHICS PIPELINE ===================================================

	void createGraphicsPipeline()
	{
		VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/tex.vert.spv"));
		VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/tex.frag.spv"));

		auto [pipeline, layout] = GraphicsPipelineBuilder(device)
			.vertShader(vert)
			.fragShader(frag)
			.vertexBinding<Vertex>()
			.vertexAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
			.vertexAttribute(1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
			.vertexAttribute(2, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv))
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
		descriptorPool = DescriptorPoolBuilder()
			.addSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAX_FRAMES_IN_FLIGHT)
			.addSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES_IN_FLIGHT)
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
				.writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					textureImageView, textureSampler,
					VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
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
		createTexture(); // must come before descriptor sets
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
