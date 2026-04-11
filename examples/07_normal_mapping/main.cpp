
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "../common/vk_common.h"
#include "../common/vk_pipelines.h"
#include "../common/vk_descriptors.h"

// ---------------------------------------------------------------------------
// 07 – Normal Mapping
//
// Builds on example 06 and introduces:
//   • Per-vertex tangent vectors stored in the vertex buffer
//   • TBN matrix constructed in the vertex shader (T, B=cross(N,T), N)
//   • Normal map sampled in the fragment shader and rotated into world space
//   • Two descriptors for textures: diffuse (sRGB) + normal map (UNORM)
//
// Scene: a large brick ground plane + rotating brick cube, both lit by 3
// orbiting point lights.  Moving lights clearly reveal the surface detail
// stored in the normal map.
//
// Textures (from examples/data/brick/):
//   short_bricks_floor_diff_1k.jpg  — loaded as VK_FORMAT_R8G8B8A8_SRGB
//   short_bricks_floor_nor_gl_1k.jpg — loaded as VK_FORMAT_R8G8B8A8_UNORM
//     (OpenGL convention: green channel = +bitangent = increasing-V direction)
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
	glm::vec4  viewPos;
	glm::vec4  ambient;
};

struct PushConstants
{
	glm::mat4 mvp;
	glm::mat4 model;
};

// ---------------------------------------------------------------------------
// Vertex layout: position + normal + UV + tangent
// ---------------------------------------------------------------------------

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 uv;
	glm::vec3 tangent;
};

// ---------------------------------------------------------------------------
// Ground plane — 20 m × 20 m, UV tiles 4×, tangent along +X
// ---------------------------------------------------------------------------

// Ground tangent: U increases in +X, V increases in -Z.
// cross(N=(0,1,0), T=(1,0,0)) = (0,0,-1) — matches V direction. ✓
static const glm::vec3 groundNormal  = {0.0f, 1.0f, 0.0f};
static const glm::vec3 groundTangent = {1.0f, 0.0f, 0.0f};

static const std::vector<Vertex> groundVertices = {
	{{-10.0f, 0.0f,  10.0f}, groundNormal, {0.0f, 0.0f}, groundTangent},
	{{ 10.0f, 0.0f,  10.0f}, groundNormal, {4.0f, 0.0f}, groundTangent},
	{{ 10.0f, 0.0f, -10.0f}, groundNormal, {4.0f, 4.0f}, groundTangent},
	{{-10.0f, 0.0f, -10.0f}, groundNormal, {0.0f, 4.0f}, groundTangent},
};

static const std::vector<uint16_t> groundIndices = {0, 1, 2, 2, 3, 0};

// ---------------------------------------------------------------------------
// Unit cube — 6 faces, 4 vertices each, per-face normals and tangents
//
// Tangent convention: T = direction of increasing U in world space.
// Bitangent is recomputed in the shader as cross(N, T).
// For faces where V increases upward (+Y) in world, cross(N,T) agrees with
// the OpenGL normal-map green-channel convention.
// ---------------------------------------------------------------------------

static void pushFace(std::vector<Vertex>& verts, std::vector<uint16_t>& idxs,
	glm::vec3 bl, glm::vec3 br, glm::vec3 tr, glm::vec3 tl,
	glm::vec3 n, glm::vec3 t)
{
	uint16_t base = static_cast<uint16_t>(verts.size());
	verts.push_back({bl, n, {0.0f, 1.0f}, t});
	verts.push_back({br, n, {1.0f, 1.0f}, t});
	verts.push_back({tr, n, {1.0f, 0.0f}, t});
	verts.push_back({tl, n, {0.0f, 0.0f}, t});
	idxs.insert(idxs.end(), {base, uint16_t(base + 1), uint16_t(base + 2),
	                          uint16_t(base + 2), uint16_t(base + 3), base});
}

static void buildCubeMesh(std::vector<Vertex>& v, std::vector<uint16_t>& i)
{
	const float h = 0.5f;
	//                   bl              br              tr              tl              normal          tangent
	pushFace(v, i, {-h,-h, h}, { h,-h, h}, { h, h, h}, {-h, h, h}, { 0, 0, 1}, { 1, 0, 0}); // Front  +Z
	pushFace(v, i, { h,-h,-h}, {-h,-h,-h}, {-h, h,-h}, { h, h,-h}, { 0, 0,-1}, {-1, 0, 0}); // Back   -Z
	pushFace(v, i, {-h,-h,-h}, {-h,-h, h}, {-h, h, h}, {-h, h,-h}, {-1, 0, 0}, { 0, 0, 1}); // Left   -X
	pushFace(v, i, { h,-h, h}, { h,-h,-h}, { h, h,-h}, { h, h, h}, { 1, 0, 0}, { 0, 0,-1}); // Right  +X
	pushFace(v, i, {-h, h, h}, { h, h, h}, { h, h,-h}, {-h, h,-h}, { 0, 1, 0}, { 1, 0, 0}); // Top    +Y
	pushFace(v, i, {-h,-h,-h}, { h,-h,-h}, { h,-h, h}, {-h,-h, h}, { 0,-1, 0}, { 1, 0, 0}); // Bottom -Y
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

class NormalMappingApp : public VkAppBase
{
private:
	// -- Depth buffer --
	VkImage        depthImage     = VK_NULL_HANDLE;
	VkDeviceMemory depthMemory    = VK_NULL_HANDLE;
	VkImageView    depthImageView = VK_NULL_HANDLE;
	VkFormat       depthFormat    = VK_FORMAT_D32_SFLOAT;

	// -- Textures --
	VkImage        diffuseImage     = VK_NULL_HANDLE;
	VkDeviceMemory diffuseMemory    = VK_NULL_HANDLE;
	VkImageView    diffuseImageView = VK_NULL_HANDLE;

	VkImage        normalImage     = VK_NULL_HANDLE;
	VkDeviceMemory normalMemory    = VK_NULL_HANDLE;
	VkImageView    normalImageView = VK_NULL_HANDLE;

	VkSampler textureSampler = VK_NULL_HANDLE; // shared sampler

	// -- Pipeline --
	VkPipelineLayout pipelineLayout   = VK_NULL_HANDLE;
	VkPipeline       graphicsPipeline = VK_NULL_HANDLE;

	// -- Descriptor system --
	VkDescriptorSetLayout        descriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool             descriptorPool      = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> descriptorSets;

	// -- Uniform buffers --
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
	VkBuffer       cubeVertexBuffer       = VK_NULL_HANDLE;
	VkDeviceMemory cubeVertexBufferMemory = VK_NULL_HANDLE;
	VkBuffer       cubeIndexBuffer        = VK_NULL_HANDLE;
	VkDeviceMemory cubeIndexBufferMemory  = VK_NULL_HANDLE;

	// === TEXTURE HELPERS =====================================================

	// Load a JPEG/PNG from disk, upload to a device-local VkImage, return its view.
	VkImageView uploadTexture(const std::string& path, VkFormat format,
		VkImage& image, VkDeviceMemory& memory)
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

		createImage(static_cast<uint32_t>(w), static_cast<uint32_t>(h),
			format, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, memory);

		{ auto cmd = beginOneTimeCommands();
		  transitionImageLayout(cmd, image,
			VK_IMAGE_LAYOUT_UNDEFINED,           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_TRANSFER_BIT,    VK_ACCESS_2_TRANSFER_WRITE_BIT);
		  endOneTimeCommands(cmd); }

		copyBufferToImage(staging, image, static_cast<uint32_t>(w), static_cast<uint32_t>(h));

		{ auto cmd = beginOneTimeCommands();
		  transitionImageLayout(cmd, image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_2_TRANSFER_BIT,        VK_ACCESS_2_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
		  endOneTimeCommands(cmd); }

		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);

		return createImageView(image, format);
	}

	void createTextures()
	{
		const std::string base = std::string(DATA_DIR) + "/brick/";
		diffuseImageView = uploadTexture(base + "short_bricks_floor_diff_1k.jpg",
			VK_FORMAT_R8G8B8A8_SRGB,  diffuseImage, diffuseMemory);
		normalImageView  = uploadTexture(base + "short_bricks_floor_nor_gl_1k.jpg",
			VK_FORMAT_R8G8B8A8_UNORM, normalImage,  normalMemory);
		textureSampler = createSampler();
	}

	void destroyTextures()
	{
		vkDestroySampler(device, textureSampler, nullptr);   textureSampler   = VK_NULL_HANDLE;
		vkDestroyImageView(device, normalImageView, nullptr); normalImageView  = VK_NULL_HANDLE;
		vkDestroyImage(device, normalImage, nullptr);         normalImage      = VK_NULL_HANDLE;
		vkFreeMemory(device, normalMemory, nullptr);          normalMemory     = VK_NULL_HANDLE;
		vkDestroyImageView(device, diffuseImageView, nullptr); diffuseImageView = VK_NULL_HANDLE;
		vkDestroyImage(device, diffuseImage, nullptr);        diffuseImage     = VK_NULL_HANDLE;
		vkFreeMemory(device, diffuseMemory, nullptr);         diffuseMemory    = VK_NULL_HANDLE;
	}

	// === DESCRIPTOR SET LAYOUT ===============================================

	void createDescriptorSetLayout()
	{
		descriptorSetLayout = DescriptorLayoutBuilder()
			.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         VK_SHADER_STAGE_FRAGMENT_BIT)
			.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
			.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
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
			.vertexAttribute(3, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent))
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

		// Lights orbit low and close so the normal-map shading effect is pronounced
		float radius = 4.0f;
		float height = 1.2f;

		float off0 = 0.0f;
		float off1 = glm::radians(120.0f);
		float off2 = glm::radians(240.0f);

		ubo.lights[0].position = glm::vec4(radius * std::cos(t * 0.6f + off0), height, radius * std::sin(t * 0.6f + off0), 0.0f);
		ubo.lights[0].color    = glm::vec4(1.0f, 0.3f, 0.1f, 8.0f);

		ubo.lights[1].position = glm::vec4(radius * std::cos(t * 0.6f + off1), height, radius * std::sin(t * 0.6f + off1), 0.0f);
		ubo.lights[1].color    = glm::vec4(0.2f, 0.6f, 1.0f, 8.0f);

		ubo.lights[2].position = glm::vec4(radius * std::cos(t * 0.6f + off2), height, radius * std::sin(t * 0.6f + off2), 0.0f);
		ubo.lights[2].color    = glm::vec4(0.4f, 1.0f, 0.3f, 8.0f);

		ubo.viewPos = glm::vec4(0.0f, 7.0f, 10.0f, 0.0f);
		ubo.ambient = glm::vec4(1.0f, 1.0f, 1.0f, 0.04f); // very dark ambient — lights stand out

		std::memcpy(uniformBuffersMapped[frameIndex], &ubo, sizeof(ubo));
	}

	// === DESCRIPTOR POOL & SETS ==============================================

	void createDescriptorPool()
	{
		descriptorPool = DescriptorPoolBuilder()
			.addSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAX_FRAMES_IN_FLIGHT)
			.addSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES_IN_FLIGHT * 2)
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
					diffuseImageView, textureSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
				.writeImage(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					normalImageView,  textureSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
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

		createTextures();
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
			VK_IMAGE_LAYOUT_UNDEFINED,                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,          VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

		transitionImageLayout(cmd, depthImage,
			VK_IMAGE_LAYOUT_UNDEFINED,                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,          VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			VK_IMAGE_ASPECT_DEPTH_BIT);

		VkRenderingAttachmentInfo colorAtt{};
		colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		colorAtt.imageView   = swapChainImageViews[imageIndex];
		colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
		colorAtt.clearValue  = {{{0.04f, 0.04f, 0.06f, 1.0f}}};

		VkRenderingAttachmentInfo depthAtt{};
		depthAtt.sType                   = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		depthAtt.imageView               = depthImageView;
		depthAtt.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
		depthAtt.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAtt.storeOp                 = VK_ATTACHMENT_STORE_OP_DONT_CARE;
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
			glm::vec3(0.0f, 7.0f, 10.0f),
			glm::vec3(0.0f, 0.0f,  0.0f),
			glm::vec3(0.0f, 1.0f,  0.0f));
		glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
		proj[1][1] *= -1.0f;

		// --- Ground plane ---
		{
			VkBuffer     bufs[]    = {groundVertexBuffer};
			VkDeviceSize offsets[] = {0};
			vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offsets);
			vkCmdBindIndexBuffer(cmd, groundIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

			glm::mat4 model = glm::mat4(1.0f);
			PushConstants pc{proj * view * model, model};
			vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
			vkCmdDrawIndexed(cmd, static_cast<uint32_t>(groundIndices.size()), 1, 0, 0, 0);
		}

		// --- Rotating cube (sits on ground, Y offset = 0.5) ---
		{
			VkBuffer     bufs[]    = {cubeVertexBuffer};
			VkDeviceSize offsets[] = {0};
			vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offsets);
			vkCmdBindIndexBuffer(cmd, cubeIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

			float t = static_cast<float>(glfwGetTime());
			glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 0.0f));
			model = glm::rotate(model, t * glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));

			PushConstants pc{proj * view * model, model};
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

	void onCleanupSwapChain() override { destroyDepthResources(depthImage, depthMemory, depthImageView); }
	void onRecreateSwapChain() override { createDepthResources(depthFormat, depthImage, depthMemory, depthImageView); }

	void onCleanup() override
	{
		destroyUBOs(uniformBuffers, uniformBuffersMemory);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		destroyTextures();

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
	NormalMappingApp app;
	try
	{
		app.run(1024, 768, "vk-examples: 07 Normal Mapping (Vulkan 1.3)");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Fatal Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
