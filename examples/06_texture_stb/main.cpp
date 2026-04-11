
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "../common/vk_common.h"
#include "../common/vk_pipelines.h"
#include "../common/vk_descriptors.h"

// ---------------------------------------------------------------------------
// 06 – Texture with stb_image + Cube on Ground Plane
//
// Builds on example 05 and introduces:
//   • stb_image  – loads brick_pavement_02_diffuse_1k.jpg from examples/data
//   • Two draw calls – a ground plane and a cube rotating around the Y axis
//   • Push constants  – different MVP+model per object
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU data structures
// ---------------------------------------------------------------------------

struct PointLight
{
	glm::vec4 position; // xyz = pos,   w = unused
	glm::vec4 color;    // xyz = color, w = intensity
};

struct LightUBO
{
	PointLight lights[3];
	glm::vec4  viewPos; // xyz = camera
	glm::vec4  ambient; // xyz = color, w = strength
};

struct PushConstants
{
	glm::mat4 mvp;
	glm::mat4 model;
};

// ---------------------------------------------------------------------------
// Vertex layout: position + normal + UV
// ---------------------------------------------------------------------------

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 uv;
};

// ---------------------------------------------------------------------------
// Ground plane – 20 m × 20 m, UV tiles 5×
// ---------------------------------------------------------------------------

static const std::vector<Vertex> groundVertices = {
	{{-10.0f, 0.0f,  10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
	{{ 10.0f, 0.0f,  10.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 0.0f}},
	{{ 10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 5.0f}},
	{{-10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 5.0f}},
};

static const std::vector<uint16_t> groundIndices = {0, 1, 2, 2, 3, 0};

// ---------------------------------------------------------------------------
// Unit cube centered at origin (half-extents 0.5).
// 24 vertices – 4 per face, each with a unique per-face normal.
// CCW winding viewed from outside. Translate +0.5 Y via model matrix to
// place it on the ground plane.
// ---------------------------------------------------------------------------

// Helper: build one quad face: bl, br, tr, tl (bottom-left → CCW from outside)
static void pushFace(std::vector<Vertex>& verts, std::vector<uint16_t>& idxs,
	glm::vec3 bl, glm::vec3 br, glm::vec3 tr, glm::vec3 tl, glm::vec3 n)
{
	uint16_t base = static_cast<uint16_t>(verts.size());
	verts.push_back({bl, n, {0.0f, 1.0f}});
	verts.push_back({br, n, {1.0f, 1.0f}});
	verts.push_back({tr, n, {1.0f, 0.0f}});
	verts.push_back({tl, n, {0.0f, 0.0f}});
	// Two triangles: 0-1-2 and 2-3-0
	idxs.insert(idxs.end(), {base, uint16_t(base + 1), uint16_t(base + 2),
	                          uint16_t(base + 2), uint16_t(base + 3), base});
}

static void buildCubeMesh(std::vector<Vertex>& v, std::vector<uint16_t>& i)
{
	const float h = 0.5f;
	// Front  (+Z)
	pushFace(v, i, {-h,-h, h}, { h,-h, h}, { h, h, h}, {-h, h, h}, { 0, 0, 1});
	// Back   (-Z)
	pushFace(v, i, { h,-h,-h}, {-h,-h,-h}, {-h, h,-h}, { h, h,-h}, { 0, 0,-1});
	// Left   (-X)
	pushFace(v, i, {-h,-h,-h}, {-h,-h, h}, {-h, h, h}, {-h, h,-h}, {-1, 0, 0});
	// Right  (+X)
	pushFace(v, i, { h,-h, h}, { h,-h,-h}, { h, h,-h}, { h, h, h}, { 1, 0, 0});
	// Top    (+Y)
	pushFace(v, i, {-h, h, h}, { h, h, h}, { h, h,-h}, {-h, h,-h}, { 0, 1, 0});
	// Bottom (-Y)
	pushFace(v, i, {-h,-h,-h}, { h,-h,-h}, { h,-h, h}, {-h,-h, h}, { 0,-1, 0});
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

class TexturedSceneApp : public VkAppBase
{
private:
	// -- Depth buffer --
	VkImage        depthImage     = VK_NULL_HANDLE;
	VkDeviceMemory depthMemory    = VK_NULL_HANDLE;
	VkImageView    depthImageView = VK_NULL_HANDLE;
	VkFormat       depthFormat    = VK_FORMAT_D32_SFLOAT;

	// -- Texture --
	VkImage        textureImage     = VK_NULL_HANDLE;
	VkDeviceMemory textureMemory    = VK_NULL_HANDLE;
	VkImageView    textureImageView = VK_NULL_HANDLE;
	VkSampler      textureSampler   = VK_NULL_HANDLE;

	// -- Pipeline --
	VkPipelineLayout pipelineLayout  = VK_NULL_HANDLE;
	VkPipeline       graphicsPipeline = VK_NULL_HANDLE;

	// -- Descriptor system --
	VkDescriptorSetLayout            descriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool                 descriptorPool      = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet>     descriptorSets; // per frame-in-flight

	// -- Uniform buffers (per frame-in-flight, persistently mapped) --
	std::vector<VkBuffer>       uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*>          uniformBuffersMapped;

	// -- Ground geometry --
	VkBuffer       groundVertexBuffer       = VK_NULL_HANDLE;
	VkDeviceMemory groundVertexBufferMemory = VK_NULL_HANDLE;
	VkBuffer       groundIndexBuffer        = VK_NULL_HANDLE;
	VkDeviceMemory groundIndexBufferMemory  = VK_NULL_HANDLE;

	// -- Cube geometry --
	std::vector<Vertex>   cubeVertices;
	std::vector<uint16_t> cubeIndices;
	VkBuffer       cubeVertexBuffer        = VK_NULL_HANDLE;
	VkDeviceMemory cubeVertexBufferMemory  = VK_NULL_HANDLE;
	VkBuffer       cubeIndexBuffer         = VK_NULL_HANDLE;
	VkDeviceMemory cubeIndexBufferMemory   = VK_NULL_HANDLE;

	// === TEXTURE (stb_image) =================================================

	// Load a texture from disk, upload to GPU, create VkImageView + VkSampler.
	void createTexture()
	{
		const std::string texPath = std::string(DATA_DIR) + "/brick/short_bricks_floor_diff_1k.jpg";
		int loadW, loadH, ch;
		stbi_uc* pixels = stbi_load(texPath.c_str(), &loadW, &loadH, &ch, STBI_rgb_alpha);
		if (!pixels)
			throw std::runtime_error("stbi_load failed (" + texPath + "): " + stbi_failure_reason());

		VkDeviceSize imageSize = static_cast<VkDeviceSize>(loadW) * loadH * 4;
		VkBuffer staging; VkDeviceMemory stagingMem;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
		void* ptr;
		vkMapMemory(device, stagingMem, 0, imageSize, 0, &ptr);
		std::memcpy(ptr, pixels, imageSize);
		vkUnmapMemory(device, stagingMem);
		stbi_image_free(pixels);

		uint32_t w = static_cast<uint32_t>(loadW), h = static_cast<uint32_t>(loadH);
		createImage(w, h, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureMemory);

		{ auto cmd = beginOneTimeCommands();
		  transitionImageLayout(cmd, textureImage,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
		  endOneTimeCommands(cmd); }

		copyBufferToImage(staging, textureImage, w, h);

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
		vkDestroySampler(device, textureSampler, nullptr);     textureSampler   = VK_NULL_HANDLE;
		vkDestroyImageView(device, textureImageView, nullptr); textureImageView = VK_NULL_HANDLE;
		vkDestroyImage(device, textureImage, nullptr);         textureImage     = VK_NULL_HANDLE;
		vkFreeMemory(device, textureMemory, nullptr);          textureMemory    = VK_NULL_HANDLE;
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
		VkShaderModule vert = createShaderModule(readFile(std::string(SHADER_DIR) + "/scene.vert.spv"));
		VkShaderModule frag = createShaderModule(readFile(std::string(SHADER_DIR) + "/scene.frag.spv"));

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

	// === GEOMETRY BUFFERS ====================================================

	template<typename V, typename I>
	void uploadMesh(const std::vector<V>& verts, const std::vector<I>& idxs,
		VkBuffer& vb, VkDeviceMemory& vbm, VkBuffer& ib, VkDeviceMemory& ibm)
	{
		uploadStagedBuffer(verts.data(), sizeof(V) * verts.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vb, vbm);
		uploadStagedBuffer(idxs.data(),  sizeof(I) * idxs.size(),  VK_BUFFER_USAGE_INDEX_BUFFER_BIT,  ib, ibm);
	}

	void updateUniformBuffer(uint32_t frameIndex)
	{
		float t = static_cast<float>(glfwGetTime());

		LightUBO ubo{};

		float radius = 5.0f;
		float height = 2.0f;

		// Red light
		ubo.lights[0].position = glm::vec4(radius * std::cos(t * 0.7f), height, radius * std::sin(t * 0.7f), 0.0f);
		ubo.lights[0].color    = glm::vec4(1.0f, 0.15f, 0.05f, 6.0f);

		// Green light (120° offset)
		float off1 = glm::radians(120.0f);
		ubo.lights[1].position = glm::vec4(radius * std::cos(t * 0.7f + off1), height, radius * std::sin(t * 0.7f + off1), 0.0f);
		ubo.lights[1].color    = glm::vec4(0.1f, 1.0f, 0.1f, 6.0f);

		// Blue light (240° offset)
		float off2 = glm::radians(240.0f);
		ubo.lights[2].position = glm::vec4(radius * std::cos(t * 0.7f + off2), height, radius * std::sin(t * 0.7f + off2), 0.0f);
		ubo.lights[2].color    = glm::vec4(0.1f, 0.2f, 1.0f, 6.0f);

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
					textureImageView, textureSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
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
		buildCubeMesh(cubeVertices, cubeIndices);

		uploadMesh(groundVertices, groundIndices,
			groundVertexBuffer, groundVertexBufferMemory,
			groundIndexBuffer,  groundIndexBufferMemory);
		uploadMesh(cubeVertices, cubeIndices,
			cubeVertexBuffer, cubeVertexBufferMemory,
			cubeIndexBuffer,  cubeIndexBufferMemory);

		createTexture();
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
			VK_IMAGE_LAYOUT_UNDEFINED,                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,        VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

		transitionImageLayout(cmd, depthImage,
			VK_IMAGE_LAYOUT_UNDEFINED,                  VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,        VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			VK_IMAGE_ASPECT_DEPTH_BIT);

		VkRenderingAttachmentInfo colorAtt{};
		colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		colorAtt.imageView   = swapChainImageViews[imageIndex];
		colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
		colorAtt.clearValue  = {{{0.05f, 0.05f, 0.08f, 1.0f}}};

		VkRenderingAttachmentInfo depthAtt{};
		depthAtt.sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		depthAtt.imageView          = depthImageView;
		depthAtt.imageLayout        = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
		depthAtt.loadOp             = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAtt.storeOp            = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAtt.clearValue.depthStencil = {1.0f, 0};

		VkRenderingInfo ri{};
		ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
		ri.renderArea           = {{0, 0}, swapChainExtent};
		ri.layerCount           = 1;
		ri.colorAttachmentCount = 1;
		ri.pColorAttachments    = &colorAtt;
		ri.pDepthAttachment     = &depthAtt;

		vkCmdBeginRendering(cmd, &ri);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		VkViewport vp{};
		vp.width    = static_cast<float>(swapChainExtent.width);
		vp.height   = static_cast<float>(swapChainExtent.height);
		vp.minDepth = 0.0f;
		vp.maxDepth = 1.0f;
		vkCmdSetViewport(cmd, 0, 1, &vp);
		VkRect2D scissor{{0, 0}, swapChainExtent};
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

		float aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
		glm::mat4 view = glm::lookAt(
			glm::vec3(0.0f, 8.0f, 12.0f),
			glm::vec3(0.0f, 0.0f, 0.0f),
			glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
		proj[1][1] *= -1.0f;

		// --- Draw ground plane ---
		{
			VkBuffer     bufs[]     = {groundVertexBuffer};
			VkDeviceSize offsets[]  = {0};
			vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offsets);
			vkCmdBindIndexBuffer(cmd, groundIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

			glm::mat4 model = glm::mat4(1.0f);
			PushConstants pc{};
			pc.mvp   = proj * view * model;
			pc.model = model;
			vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
			vkCmdDrawIndexed(cmd, static_cast<uint32_t>(groundIndices.size()), 1, 0, 0, 0);
		}

		// --- Draw rotating cube (centered on origin, sitting on the ground) ---
		{
			VkBuffer     bufs[]    = {cubeVertexBuffer};
			VkDeviceSize offsets[] = {0};
			vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offsets);
			vkCmdBindIndexBuffer(cmd, cubeIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

			float t = static_cast<float>(glfwGetTime());
			// Translate +0.5 Y so the cube's bottom sits exactly on the ground (Y=0)
			glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 0.0f));
			model = glm::rotate(model, t * glm::radians(60.0f), glm::vec3(0.0f, 1.0f, 0.0f));

			PushConstants pc{};
			pc.mvp   = proj * view * model;
			pc.model = model;
			vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
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

		vkDestroyBuffer(device, cubeIndexBuffer, nullptr);
		vkFreeMemory(device, cubeIndexBufferMemory, nullptr);
		vkDestroyBuffer(device, cubeVertexBuffer, nullptr);
		vkFreeMemory(device, cubeVertexBufferMemory, nullptr);

		vkDestroyBuffer(device, groundIndexBuffer, nullptr);
		vkFreeMemory(device, groundIndexBufferMemory, nullptr);
		vkDestroyBuffer(device, groundVertexBuffer, nullptr);
		vkFreeMemory(device, groundVertexBufferMemory, nullptr);

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
	}
};

// ===========================================================================

int main()
{
	TexturedSceneApp app;
	try
	{
		app.run(1024, 768, "vk-examples: 06 Texture (stb_image) + Rotating Cube (Vulkan 1.3)");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Fatal Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
