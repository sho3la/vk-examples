
#include "../common/vk_common.h"

// ---------------------------------------------------------------------------
// 01 – Clear Screen
//
// Demonstrates the minimal Vulkan 1.3 frame loop:
//   • Dynamic rendering  (vkCmdBeginRendering – no VkRenderPass)
//   • Synchronization2   (vkQueueSubmit2)
//   • Image layout transitions via vkCmdPipelineBarrier2
//
// The window clears to an animated RGB colour; no geometry is drawn.
// ---------------------------------------------------------------------------

class ClearScreenApp : public VkAppBase
{
protected:
	// No extra resources needed before the command pool
	void onInitBeforeCommandPool() override {}

	// No buffers or descriptors to upload
	void onInitAfterCommandPool() override {}

	// -----------------------------------------------------------------------
	// Command buffer recording
	// -----------------------------------------------------------------------

	void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) override
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
			throw std::runtime_error("failed to begin recording command buffer!");

		// 1. UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
		transitionImageLayout(cmd, swapChainImages[imageIndex],
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

		// Animated clear colour
		const auto t = static_cast<float>(glfwGetTime());
		float r = (std::sin(t * 1.0f) * 0.5f) + 0.5f;
		float g = (std::sin(t * 1.5f) * 0.5f) + 0.5f;
		float b = (std::sin(t * 2.0f) * 0.5f) + 0.5f;

		VkClearValue clearColor = {{{r, g, b, 1.0f}}};

		// 2. Dynamic Rendering (Vulkan 1.3 core)
		VkRenderingAttachmentInfo colorAttachment{};
		colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		colorAttachment.imageView = swapChainImageViews[imageIndex];
		colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.clearValue = clearColor;

		VkRenderingInfo renderingInfo{};
		renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		renderingInfo.renderArea.offset = {0, 0};
		renderingInfo.renderArea.extent = swapChainExtent;
		renderingInfo.layerCount = 1;
		renderingInfo.colorAttachmentCount = 1;
		renderingInfo.pColorAttachments = &colorAttachment;

		vkCmdBeginRendering(cmd, &renderingInfo);
		// (Draw calls would go here)
		vkCmdEndRendering(cmd);

		// 3. COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR
		transitionImageLayout(cmd, swapChainImages[imageIndex],
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE);

		if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
			throw std::runtime_error("failed to record command buffer!");
	}

	// No extra resources to destroy
	void onCleanup() override {}
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main()
{
	ClearScreenApp app;

	try
	{
		app.run(800, 600, "vk-examples: 01 Clear Screen");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Fatal Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
