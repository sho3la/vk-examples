
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "../common/vk_common.h"

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

	// === DEPTH BUFFER ========================================================

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

		VkMemoryRequirements memReq;
		vkGetImageMemoryRequirements(device, depthImage, &memReq);
		VkMemoryAllocateInfo ai{};
		ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize  = memReq.size;
		ai.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (vkAllocateMemory(device, &ai, nullptr, &depthMemory) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate depth memory!");
		vkBindImageMemory(device, depthImage, depthMemory, 0);

		VkImageViewCreateInfo vi{};
		vi.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
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

	// === TEXTURE HELPERS =====================================================

	void createImage(uint32_t w, uint32_t h, VkFormat format, VkImageTiling tiling,
		VkImageUsageFlags usage, VkMemoryPropertyFlags memProps,
		VkImage& image, VkDeviceMemory& memory)
	{
		VkImageCreateInfo ci{};
		ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ci.imageType     = VK_IMAGE_TYPE_2D;
		ci.format        = format;
		ci.extent        = {w, h, 1};
		ci.mipLevels     = 1;
		ci.arrayLayers   = 1;
		ci.samples       = VK_SAMPLE_COUNT_1_BIT;
		ci.tiling        = tiling;
		ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		ci.usage         = usage;
		ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
		if (vkCreateImage(device, &ci, nullptr, &image) != VK_SUCCESS)
			throw std::runtime_error("failed to create image!");

		VkMemoryRequirements req;
		vkGetImageMemoryRequirements(device, image, &req);
		VkMemoryAllocateInfo ai{};
		ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize  = req.size;
		ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, memProps);
		if (vkAllocateMemory(device, &ai, nullptr, &memory) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate image memory!");
		vkBindImageMemory(device, image, memory, 0);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t w, uint32_t h)
	{
		VkCommandBufferAllocateInfo ai{};
		ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ai.commandPool        = commandPool;
		ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ai.commandBufferCount = 1;
		VkCommandBuffer cmd;
		vkAllocateCommandBuffers(device, &ai, &cmd);

		VkCommandBufferBeginInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &bi);

		VkBufferImageCopy region{};
		region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		region.imageExtent      = {w, h, 1};
		vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		vkEndCommandBuffer(cmd);
		VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		si.commandBufferCount = 1;
		si.pCommandBuffers    = &cmd;
		vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);
		vkFreeCommandBuffers(device, commandPool, 1, &cmd);
	}

	// Load a JPEG/PNG from disk with stbi_load, upload to a device-local VkImage,
	// and return a VkImageView. format must match the pixel data (sRGB or UNORM).
	VkImageView uploadTexture(const std::string& path, VkFormat format,
		VkImage& image, VkDeviceMemory& memory)
	{
		int w, h, ch;
		stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &ch, STBI_rgb_alpha);
		if (!pixels)
			throw std::runtime_error("stbi_load failed (" + path + "): " + stbi_failure_reason());

		VkDeviceSize size = static_cast<VkDeviceSize>(w) * h * 4;

		VkBuffer       staging;
		VkDeviceMemory stagingMem;
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

		// Transition → TRANSFER_DST
		{
			VkCommandBufferAllocateInfo ai{};
			ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			ai.commandPool = commandPool;
			ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			ai.commandBufferCount = 1;
			VkCommandBuffer cmd;
			vkAllocateCommandBuffers(device, &ai, &cmd);
			VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
			bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(cmd, &bi);
			transitionImageLayout(cmd, image,
				VK_IMAGE_LAYOUT_UNDEFINED,           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
				VK_PIPELINE_STAGE_2_TRANSFER_BIT,    VK_ACCESS_2_TRANSFER_WRITE_BIT);
			vkEndCommandBuffer(cmd);
			VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
			si.commandBufferCount = 1;
			si.pCommandBuffers    = &cmd;
			vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
			vkQueueWaitIdle(graphicsQueue);
			vkFreeCommandBuffers(device, commandPool, 1, &cmd);
		}

		copyBufferToImage(staging, image, static_cast<uint32_t>(w), static_cast<uint32_t>(h));

		// Transition → SHADER_READ_ONLY
		{
			VkCommandBufferAllocateInfo ai{};
			ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			ai.commandPool = commandPool;
			ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			ai.commandBufferCount = 1;
			VkCommandBuffer cmd;
			vkAllocateCommandBuffers(device, &ai, &cmd);
			VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
			bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(cmd, &bi);
			transitionImageLayout(cmd, image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_2_TRANSFER_BIT,        VK_ACCESS_2_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
			vkEndCommandBuffer(cmd);
			VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
			si.commandBufferCount = 1;
			si.pCommandBuffers    = &cmd;
			vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
			vkQueueWaitIdle(graphicsQueue);
			vkFreeCommandBuffers(device, commandPool, 1, &cmd);
		}

		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);

		VkImageViewCreateInfo vi{};
		vi.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		vi.image            = image;
		vi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
		vi.format           = format;
		vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		VkImageView view;
		if (vkCreateImageView(device, &vi, nullptr, &view) != VK_SUCCESS)
			throw std::runtime_error("failed to create image view!");
		return view;
	}

	void createTextures()
	{
		const std::string base = std::string(DATA_DIR) + "/brick/";

		// Diffuse — sRGB (gamma-corrected colour data)
		diffuseImageView = uploadTexture(base + "short_bricks_floor_diff_1k.jpg",
			VK_FORMAT_R8G8B8A8_SRGB, diffuseImage, diffuseMemory);

		// Normal map — UNORM (linear; must NOT be gamma-corrected)
		normalImageView = uploadTexture(base + "short_bricks_floor_nor_gl_1k.jpg",
			VK_FORMAT_R8G8B8A8_UNORM, normalImage, normalMemory);

		// Shared sampler (repeat + linear filtering for both maps)
		VkSamplerCreateInfo samplerCI{};
		samplerCI.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerCI.magFilter     = VK_FILTER_LINEAR;
		samplerCI.minFilter     = VK_FILTER_LINEAR;
		samplerCI.addressModeU  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCI.addressModeV  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCI.addressModeW  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCI.maxAnisotropy = 1.0f;
		samplerCI.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		if (vkCreateSampler(device, &samplerCI, nullptr, &textureSampler) != VK_SUCCESS)
			throw std::runtime_error("failed to create texture sampler!");
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
		// Binding 0: lights UBO
		VkDescriptorSetLayoutBinding uboBinding{};
		uboBinding.binding         = 0;
		uboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboBinding.descriptorCount = 1;
		uboBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

		// Binding 1: diffuse texture
		VkDescriptorSetLayoutBinding diffuseBinding{};
		diffuseBinding.binding         = 1;
		diffuseBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		diffuseBinding.descriptorCount = 1;
		diffuseBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

		// Binding 2: normal map
		VkDescriptorSetLayoutBinding normalBinding{};
		normalBinding.binding         = 2;
		normalBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		normalBinding.descriptorCount = 1;
		normalBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 3> bindings = {uboBinding, diffuseBinding, normalBinding};
		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = static_cast<uint32_t>(bindings.size());
		ci.pBindings    = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &descriptorSetLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor set layout!");
	}

	// === GRAPHICS PIPELINE ===================================================

	void createGraphicsPipeline()
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

		// Vertex input: pos + normal + uv + tangent (4 attributes)
		VkVertexInputBindingDescription bindDesc{};
		bindDesc.binding   = 0;
		bindDesc.stride    = sizeof(Vertex);
		bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		std::array<VkVertexInputAttributeDescription, 4> attrDescs{};
		attrDescs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
		attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
		attrDescs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)};
		attrDescs[3] = {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent)};

		VkPipelineVertexInputStateCreateInfo vertexInput{};
		vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInput.vertexBindingDescriptionCount   = 1;
		vertexInput.pVertexBindingDescriptions      = &bindDesc;
		vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
		vertexInput.pVertexAttributeDescriptions    = attrDescs.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount  = 1;

		VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynState{};
		dynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynState.dynamicStateCount = 2;
		dynState.pDynamicStates    = dynStates;

		VkPipelineRasterizationStateCreateInfo raster{};
		raster.sType     = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		raster.polygonMode = VK_POLYGON_MODE_FILL;
		raster.cullMode  = VK_CULL_MODE_BACK_BIT;
		raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		raster.lineWidth = 1.0f;

		VkPipelineMultisampleStateCreateInfo msaa{};
		msaa.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable  = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

		VkPipelineColorBlendAttachmentState blendAtt{};
		blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo blend{};
		blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blend.attachmentCount = 1;
		blend.pAttachments    = &blendAtt;

		VkPushConstantRange pcRange{};
		pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pcRange.offset     = 0;
		pcRange.size       = sizeof(PushConstants);

		VkPipelineLayoutCreateInfo layoutCI{};
		layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutCI.setLayoutCount         = 1;
		layoutCI.pSetLayouts            = &descriptorSetLayout;
		layoutCI.pushConstantRangeCount = 1;
		layoutCI.pPushConstantRanges    = &pcRange;
		if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &pipelineLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout!");

		VkPipelineRenderingCreateInfo renderingCI{};
		renderingCI.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
		renderingCI.colorAttachmentCount    = 1;
		renderingCI.pColorAttachmentFormats = &swapChainImageFormat;
		renderingCI.depthAttachmentFormat   = depthFormat;

		VkGraphicsPipelineCreateInfo pipelineCI{};
		pipelineCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
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
		pipelineCI.layout              = pipelineLayout;
		pipelineCI.renderPass          = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &graphicsPipeline) != VK_SUCCESS)
			throw std::runtime_error("failed to create graphics pipeline!");

		vkDestroyShaderModule(device, fragModule, nullptr);
		vkDestroyShaderModule(device, vertModule, nullptr);
	}

	// === GEOMETRY BUFFERS ====================================================

	template<typename V, typename I>
	void uploadMesh(const std::vector<V>& verts, const std::vector<I>& idxs,
		VkBuffer& vb, VkDeviceMemory& vbm, VkBuffer& ib, VkDeviceMemory& ibm)
	{
		{
			VkDeviceSize size = sizeof(V) * verts.size();
			VkBuffer stg; VkDeviceMemory stgMem;
			createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stg, stgMem);
			void* data;
			vkMapMemory(device, stgMem, 0, size, 0, &data);
			std::memcpy(data, verts.data(), size);
			vkUnmapMemory(device, stgMem);
			createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vb, vbm);
			copyBuffer(stg, vb, size);
			vkDestroyBuffer(device, stg, nullptr);
			vkFreeMemory(device, stgMem, nullptr);
		}
		{
			VkDeviceSize size = sizeof(I) * idxs.size();
			VkBuffer stg; VkDeviceMemory stgMem;
			createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stg, stgMem);
			void* data;
			vkMapMemory(device, stgMem, 0, size, 0, &data);
			std::memcpy(data, idxs.data(), size);
			vkUnmapMemory(device, stgMem);
			createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ib, ibm);
			copyBuffer(stg, ib, size);
			vkDestroyBuffer(device, stg, nullptr);
			vkFreeMemory(device, stgMem, nullptr);
		}
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
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				uniformBuffers[i], uniformBuffersMemory[i]);
			vkMapMemory(device, uniformBuffersMemory[i], 0, size, 0, &uniformBuffersMapped[i]);
		}
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
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2; // diffuse + normal

		VkDescriptorPoolCreateInfo ci{};
		ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		ci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		ci.pPoolSizes    = poolSizes.data();
		ci.maxSets       = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(device, &ci, nullptr, &descriptorPool) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor pool!");
	}

	void createDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		VkDescriptorSetAllocateInfo ai{};
		ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
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
			bufInfo.range  = sizeof(LightUBO);

			VkDescriptorImageInfo diffuseInfo{};
			diffuseInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			diffuseInfo.imageView   = diffuseImageView;
			diffuseInfo.sampler     = textureSampler;

			VkDescriptorImageInfo normalInfo{};
			normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			normalInfo.imageView   = normalImageView;
			normalInfo.sampler     = textureSampler;

			std::array<VkWriteDescriptorSet, 3> writes{};

			writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[0].dstSet          = descriptorSets[i];
			writes[0].dstBinding      = 0;
			writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			writes[0].descriptorCount = 1;
			writes[0].pBufferInfo     = &bufInfo;

			writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[1].dstSet          = descriptorSets[i];
			writes[1].dstBinding      = 1;
			writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			writes[1].descriptorCount = 1;
			writes[1].pImageInfo      = &diffuseInfo;

			writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[2].dstSet          = descriptorSets[i];
			writes[2].dstBinding      = 2;
			writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			writes[2].descriptorCount = 1;
			writes[2].pImageInfo      = &normalInfo;

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
		buildCubeMesh(cubeVertices, cubeIndices);

		uploadMesh(groundVertices, groundIndices,
			groundVertexBuffer, groundVertexBufferMemory,
			groundIndexBuffer,  groundIndexBufferMemory);
		uploadMesh(cubeVertices, cubeIndices,
			cubeVertexBuffer, cubeVertexBufferMemory,
			cubeIndexBuffer,  cubeIndexBufferMemory);

		createTextures(); // must come before descriptor sets
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
