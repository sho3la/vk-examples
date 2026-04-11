
#pragma once

// ---------------------------------------------------------------------------
// vk_pipelines.h  –  GraphicsPipelineBuilder
//
// Reduces the 80-120 line Vulkan graphics pipeline creation boilerplate to a
// short fluent chain.  Sensible defaults match the conventions used across all
// vk-examples:
//   • TRIANGLE_LIST topology
//   • Dynamic viewport + scissor
//   • CCW front face, back-face culling
//   • Depth test + write enabled, LESS compare op
//   • No blending, single colour attachment
//
// Usage:
//
//   auto [pipeline, layout] = GraphicsPipelineBuilder(device)
//       .vertShader(vertModule)
//       .fragShader(fragModule)
//       .vertexBinding<Vertex>()
//       .vertexAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
//       .vertexAttribute(1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
//       .pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(PushConstants))
//       .descriptorSetLayout(myLayout)
//       .colorFormat(swapChainImageFormat)
//       .depthFormat(VK_FORMAT_D32_SFLOAT)
//       .build();
//
// The caller is responsible for destroying the returned VkPipeline and
// VkPipelineLayout when no longer needed.
// ---------------------------------------------------------------------------

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <vector>

class GraphicsPipelineBuilder
{
public:
	explicit GraphicsPipelineBuilder(VkDevice device) : device_(device) {}

	// -----------------------------------------------------------------------
	// Shader stages
	// -----------------------------------------------------------------------

	GraphicsPipelineBuilder& vertShader(VkShaderModule module)
	{
		VkPipelineShaderStageCreateInfo s{};
		s.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		s.stage  = VK_SHADER_STAGE_VERTEX_BIT;
		s.module = module;
		s.pName  = "main";
		stages_.push_back(s);
		return *this;
	}

	GraphicsPipelineBuilder& fragShader(VkShaderModule module)
	{
		VkPipelineShaderStageCreateInfo s{};
		s.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		s.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
		s.module = module;
		s.pName  = "main";
		stages_.push_back(s);
		return *this;
	}

	// -----------------------------------------------------------------------
	// Vertex input
	// -----------------------------------------------------------------------

	// Set the per-vertex binding stride.  Call once per binding (binding 0 is
	// assumed; multi-binding setups should call setVertexBinding() directly).
	GraphicsPipelineBuilder& vertexBinding(uint32_t stride,
		VkVertexInputRate rate = VK_VERTEX_INPUT_RATE_VERTEX)
	{
		VkVertexInputBindingDescription b{};
		b.binding   = static_cast<uint32_t>(bindings_.size());
		b.stride    = stride;
		b.inputRate = rate;
		bindings_.push_back(b);
		return *this;
	}

	// Convenience overload: infers stride from a vertex type.
	template<typename VertexT>
	GraphicsPipelineBuilder& vertexBinding(
		VkVertexInputRate rate = VK_VERTEX_INPUT_RATE_VERTEX)
	{
		return vertexBinding(static_cast<uint32_t>(sizeof(VertexT)), rate);
	}

	// Add a per-vertex attribute.  binding defaults to the last binding added.
	GraphicsPipelineBuilder& vertexAttribute(uint32_t location, VkFormat format,
		uint32_t offset, uint32_t binding = 0)
	{
		VkVertexInputAttributeDescription a{};
		a.location = location;
		a.binding  = binding;
		a.format   = format;
		a.offset   = offset;
		attributes_.push_back(a);
		return *this;
	}

	// No vertex input (e.g. fullscreen triangle driven by gl_VertexIndex).
	GraphicsPipelineBuilder& noVertexInput()
	{
		noVertex_ = true;
		return *this;
	}

	// -----------------------------------------------------------------------
	// Pipeline layout
	// -----------------------------------------------------------------------

	GraphicsPipelineBuilder& pushConstantRange(VkShaderStageFlags stages,
		uint32_t size, uint32_t offset = 0)
	{
		VkPushConstantRange r{};
		r.stageFlags = stages;
		r.offset     = offset;
		r.size       = size;
		pushConstants_.push_back(r);
		return *this;
	}

	GraphicsPipelineBuilder& descriptorSetLayout(VkDescriptorSetLayout layout)
	{
		setLayouts_.push_back(layout);
		return *this;
	}

	// -----------------------------------------------------------------------
	// Dynamic rendering (Vulkan 1.3, replaces VkRenderPass)
	// -----------------------------------------------------------------------

	// Add one colour attachment format.  Call multiple times for MRT.
	GraphicsPipelineBuilder& colorFormat(VkFormat format)
	{
		colorFormats_.push_back(format);
		return *this;
	}

	// Set the depth attachment format.  Pass VK_FORMAT_UNDEFINED to omit depth.
	GraphicsPipelineBuilder& depthFormat(VkFormat format)
	{
		depthFormat_ = format;
		return *this;
	}

	// No colour output (e.g. shadow / depth-only pass).
	GraphicsPipelineBuilder& noColorOutput()
	{
		colorFormats_.clear();
		return *this;
	}

	// -----------------------------------------------------------------------
	// Rasterisation & depth/stencil
	// -----------------------------------------------------------------------

	GraphicsPipelineBuilder& cullMode(VkCullModeFlags mode)
	{
		cullMode_ = mode;
		return *this;
	}

	GraphicsPipelineBuilder& frontFace(VkFrontFace ff)
	{
		frontFace_ = ff;
		return *this;
	}

	GraphicsPipelineBuilder& depthTest(bool testEnable, bool writeEnable = true,
		VkCompareOp op = VK_COMPARE_OP_LESS)
	{
		depthTestEnable_  = testEnable  ? VK_TRUE : VK_FALSE;
		depthWriteEnable_ = writeEnable ? VK_TRUE : VK_FALSE;
		depthCompareOp_   = op;
		return *this;
	}

	// Disable depth testing entirely (e.g. fullscreen post-process passes).
	GraphicsPipelineBuilder& noDepth()
	{
		return depthTest(false, false);
	}

	// -----------------------------------------------------------------------
	// Colour blending
	// -----------------------------------------------------------------------

	// Enable standard src-alpha blending on all colour attachments.
	GraphicsPipelineBuilder& alphaBlend(bool enable = true)
	{
		alphaBlend_ = enable;
		return *this;
	}

	// -----------------------------------------------------------------------
	// Build
	// -----------------------------------------------------------------------

	// Returns {pipeline, pipelineLayout}.  The caller must destroy both.
	std::pair<VkPipeline, VkPipelineLayout> build() const
	{
		// Pipeline layout
		VkPipelineLayoutCreateInfo layoutCI{};
		layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutCI.setLayoutCount         = static_cast<uint32_t>(setLayouts_.size());
		layoutCI.pSetLayouts            = setLayouts_.empty() ? nullptr : setLayouts_.data();
		layoutCI.pushConstantRangeCount = static_cast<uint32_t>(pushConstants_.size());
		layoutCI.pPushConstantRanges    = pushConstants_.empty() ? nullptr : pushConstants_.data();

		VkPipelineLayout layout;
		if (vkCreatePipelineLayout(device_, &layoutCI, nullptr, &layout) != VK_SUCCESS)
			throw std::runtime_error("GraphicsPipelineBuilder: failed to create pipeline layout!");

		// Vertex input
		VkPipelineVertexInputStateCreateInfo vertexInput{};
		vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		if (!noVertex_)
		{
			vertexInput.vertexBindingDescriptionCount   = static_cast<uint32_t>(bindings_.size());
			vertexInput.pVertexBindingDescriptions      = bindings_.empty()    ? nullptr : bindings_.data();
			vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes_.size());
			vertexInput.pVertexAttributeDescriptions    = attributes_.empty()  ? nullptr : attributes_.data();
		}

		// Input assembly
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		// Viewport (dynamic)
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount  = 1;

		VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynState{};
		dynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynState.dynamicStateCount = 2;
		dynState.pDynamicStates    = dynStates;

		// Rasterisation
		VkPipelineRasterizationStateCreateInfo raster{};
		raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		raster.polygonMode = VK_POLYGON_MODE_FILL;
		raster.cullMode    = cullMode_;
		raster.frontFace   = frontFace_;
		raster.lineWidth   = 1.0f;

		// Multisample
		VkPipelineMultisampleStateCreateInfo msaa{};
		msaa.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Depth / stencil
		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable  = depthTestEnable_;
		depthStencil.depthWriteEnable = depthWriteEnable_;
		depthStencil.depthCompareOp   = depthCompareOp_;

		// Colour blend (one state per attachment)
		uint32_t numColorAtts = colorFormats_.empty() ? 0
		                      : static_cast<uint32_t>(colorFormats_.size());
		std::vector<VkPipelineColorBlendAttachmentState> blendAtts(numColorAtts);
		for (auto& att : blendAtts)
		{
			att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			if (alphaBlend_)
			{
				att.blendEnable         = VK_TRUE;
				att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
				att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
				att.colorBlendOp        = VK_BLEND_OP_ADD;
				att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
				att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
				att.alphaBlendOp        = VK_BLEND_OP_ADD;
			}
		}
		VkPipelineColorBlendStateCreateInfo blend{};
		blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blend.attachmentCount = static_cast<uint32_t>(blendAtts.size());
		blend.pAttachments    = blendAtts.empty() ? nullptr : blendAtts.data();

		// Dynamic rendering (Vulkan 1.3)
		VkPipelineRenderingCreateInfo renderingCI{};
		renderingCI.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
		renderingCI.colorAttachmentCount    = static_cast<uint32_t>(colorFormats_.size());
		renderingCI.pColorAttachmentFormats = colorFormats_.empty() ? nullptr : colorFormats_.data();
		renderingCI.depthAttachmentFormat   = depthFormat_;

		// Final pipeline create info
		VkGraphicsPipelineCreateInfo pipelineCI{};
		pipelineCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineCI.pNext               = &renderingCI;
		pipelineCI.stageCount          = static_cast<uint32_t>(stages_.size());
		pipelineCI.pStages             = stages_.data();
		pipelineCI.pVertexInputState   = &vertexInput;
		pipelineCI.pInputAssemblyState = &inputAssembly;
		pipelineCI.pViewportState      = &viewportState;
		pipelineCI.pRasterizationState = &raster;
		pipelineCI.pMultisampleState   = &msaa;
		pipelineCI.pDepthStencilState  = &depthStencil;
		pipelineCI.pColorBlendState    = &blend;
		pipelineCI.pDynamicState       = &dynState;
		pipelineCI.layout              = layout;
		pipelineCI.renderPass          = VK_NULL_HANDLE;

		VkPipeline pipeline;
		if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline) != VK_SUCCESS)
		{
			vkDestroyPipelineLayout(device_, layout, nullptr);
			throw std::runtime_error("GraphicsPipelineBuilder: failed to create graphics pipeline!");
		}

		return {pipeline, layout};
	}

private:
	VkDevice device_;

	std::vector<VkPipelineShaderStageCreateInfo>      stages_;
	std::vector<VkVertexInputBindingDescription>      bindings_;
	std::vector<VkVertexInputAttributeDescription>    attributes_;
	std::vector<VkPushConstantRange>                  pushConstants_;
	std::vector<VkDescriptorSetLayout>                setLayouts_;
	std::vector<VkFormat>                             colorFormats_;

	VkFormat        depthFormat_      = VK_FORMAT_UNDEFINED;
	VkCullModeFlags cullMode_         = VK_CULL_MODE_BACK_BIT;
	VkFrontFace     frontFace_        = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	VkBool32        depthTestEnable_  = VK_TRUE;
	VkBool32        depthWriteEnable_ = VK_TRUE;
	VkCompareOp     depthCompareOp_   = VK_COMPARE_OP_LESS;
	bool            alphaBlend_       = false;
	bool            noVertex_         = false;
};
