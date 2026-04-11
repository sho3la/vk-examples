
#pragma once

// ---------------------------------------------------------------------------
// vk_common.h  –  Shared Vulkan 1.3 boilerplate for vk-examples
//
// Provides VkAppBase: a base class that handles all common Vulkan setup so
// each example only needs to implement its unique rendering logic.
//
// Virtual hooks for derived classes:
//   onInitBeforeCommandPool() – create depth resources, descriptor set
//                               layout, graphics pipeline
//   onInitAfterCommandPool()  – upload vertex/index buffers, create uniform
//                               buffers, allocate descriptor sets
//   recordCommandBuffer()     – record all draw calls for one frame (pure virtual)
//   onBeforeRecord()          – called each frame before recordCommandBuffer;
//                               use to update per-frame uniform buffers
//   onCleanupSwapChain()      – destroy depth resources (swapchain-sized)
//   onRecreateSwapChain()     – recreate depth resources after resize
//   onCleanup()               – destroy pipelines, buffers, descriptor objects
// ---------------------------------------------------------------------------

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation",
};

// ---------------------------------------------------------------------------
// Helper structs
// ---------------------------------------------------------------------------

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() const { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities{};
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

// ---------------------------------------------------------------------------
// VkAppBase
// ---------------------------------------------------------------------------

class VkAppBase
{
public:
	void run(uint32_t width, uint32_t height, const char* title)
	{
		initWindow(width, height, title);
		initVulkan();
		mainLoop();
		cleanup();
	}

protected:
	// -- Window --
	GLFWwindow* window = nullptr;
	bool framebufferResized = false;

	// -- Core Vulkan --
	VkInstance instance = VK_NULL_HANDLE;
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue graphicsQueue = VK_NULL_HANDLE;
	VkQueue presentQueue = VK_NULL_HANDLE;
	VkPhysicalDeviceMemoryProperties memProperties{};

	// -- Swapchain --
	VkSwapchainKHR swapChain = VK_NULL_HANDLE;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat{};
	VkExtent2D swapChainExtent{};
	std::vector<VkImageView> swapChainImageViews;

	// -- Commands --
	VkCommandPool commandPool = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> commandBuffers;

	// -- Synchronisation --
	std::vector<VkSemaphore> imageAvailableSemaphores; // per frame-in-flight
	std::vector<VkSemaphore> renderFinishedSemaphores; // per swapchain image
	std::vector<VkFence> inFlightFences;			   // per frame-in-flight
	uint32_t currentFrame = 0;

	// -----------------------------------------------------------------------
	// Virtual hooks – override in derived classes
	// -----------------------------------------------------------------------

	// Called between createImageViews() and createCommandPool().
	// Create depth resources, descriptor set layout, graphics pipeline here.
	virtual void onInitBeforeCommandPool() {}

	// Called between createCommandPool() and createCommandBuffers().
	// Upload vertex/index buffers and create uniform buffers / descriptor sets here.
	virtual void onInitAfterCommandPool() {}

	// Record all draw calls for one frame into cmd.
	virtual void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) = 0;

	// Called each frame in drawFrame() after the fence is reset, before recording.
	// Use this to update per-frame uniform buffers.
	virtual void onBeforeRecord(uint32_t /*frameIndex*/) {}

	// Called at the start of cleanupSwapChain(). Destroy depth resources here.
	virtual void onCleanupSwapChain() {}

	// Called in recreateSwapChain() after createImageViews(). Recreate depth resources here.
	virtual void onRecreateSwapChain() {}

	// Called in cleanup() after the command pool is destroyed.
	// Destroy pipelines, buffers, descriptor sets, etc. here.
	virtual void onCleanup() {}

	// -----------------------------------------------------------------------
	// Common helpers available to derived classes
	// -----------------------------------------------------------------------

	// Find a memory type index satisfying typeBits and required properties.
	uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const
	{
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
			if ((typeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & props) == props)
				return i;
		throw std::runtime_error("failed to find suitable memory type!");
	}

	// Allocate a VkBuffer with its own VkDeviceMemory.
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps,
		VkBuffer& buffer, VkDeviceMemory& memory)
	{
		VkBufferCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		ci.size = size;
		ci.usage = usage;
		ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (vkCreateBuffer(device, &ci, nullptr, &buffer) != VK_SUCCESS)
			throw std::runtime_error("failed to create buffer!");

		VkMemoryRequirements req;
		vkGetBufferMemoryRequirements(device, buffer, &req);
		VkMemoryAllocateInfo ai{};
		ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize = req.size;
		ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, memProps);
		if (vkAllocateMemory(device, &ai, nullptr, &memory) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate buffer memory!");
		vkBindBufferMemory(device, buffer, memory, 0);
	}

	// Allocate and begin a one-shot command buffer for transfer / upload operations.
	VkCommandBuffer beginOneTimeCommands()
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
		return cmd;
	}

	// Submit and free a one-shot command buffer created by beginOneTimeCommands().
	void endOneTimeCommands(VkCommandBuffer cmd)
	{
		vkEndCommandBuffer(cmd);
		VkSubmitInfo si{};
		si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		si.commandBufferCount = 1;
		si.pCommandBuffers    = &cmd;
		vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);
		vkFreeCommandBuffers(device, commandPool, 1, &cmd);
	}

	// Copy size bytes from src to dst using a one-shot command buffer.
	void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
	{
		VkCommandBuffer cmd = beginOneTimeCommands();
		VkBufferCopy region{};
		region.size = size;
		vkCmdCopyBuffer(cmd, src, dst, 1, &region);
		endOneTimeCommands(cmd);
	}

	// Read a binary file into a byte vector (for loading SPIR-V).
	static std::vector<char> readFile(const std::string& path)
	{
		std::ifstream file(path, std::ios::ate | std::ios::binary);
		if (!file.is_open())
			throw std::runtime_error("failed to open file: " + path);
		size_t sz = static_cast<size_t>(file.tellg());
		std::vector<char> buf(sz);
		file.seekg(0);
		file.read(buf.data(), static_cast<std::streamsize>(sz));
		return buf;
	}

	// Wrap SPIR-V bytecode in a VkShaderModule.
	VkShaderModule createShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		ci.codeSize = code.size();
		ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
		VkShaderModule sm;
		if (vkCreateShaderModule(device, &ci, nullptr, &sm) != VK_SUCCESS)
			throw std::runtime_error("failed to create shader module!");
		return sm;
	}

	// Image layout transition using Vulkan 1.3 vkCmdPipelineBarrier2.
	// aspect defaults to VK_IMAGE_ASPECT_COLOR_BIT for colour images.
	static void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
		VkImageLayout oldLayout, VkImageLayout newLayout,
		VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
		VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
		VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT)
	{
		VkImageMemoryBarrier2 barrier{};
		barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		barrier.srcStageMask        = srcStage;
		barrier.srcAccessMask       = srcAccess;
		barrier.dstStageMask        = dstStage;
		barrier.dstAccessMask       = dstAccess;
		barrier.oldLayout           = oldLayout;
		barrier.newLayout           = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = image;
		barrier.subresourceRange    = {aspect, 0, 1, 0, 1};

		VkDependencyInfo dep{};
		dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dep.imageMemoryBarrierCount = 1;
		dep.pImageMemoryBarriers    = &barrier;
		vkCmdPipelineBarrier2(cmd, &dep);
	}

	// -----------------------------------------------------------------------
	// Image helpers
	// -----------------------------------------------------------------------

	// Create a 2-D VkImage backed by device memory.
	void createImage(uint32_t w, uint32_t h, VkFormat format,
		VkImageTiling tiling, VkImageUsageFlags usage,
		VkMemoryPropertyFlags memProps,
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

	// Create a 2-D VkImageView.
	VkImageView createImageView(VkImage image, VkFormat format,
		VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT)
	{
		VkImageViewCreateInfo ci{};
		ci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		ci.image            = image;
		ci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
		ci.format           = format;
		ci.subresourceRange = {aspect, 0, 1, 0, 1};
		VkImageView view;
		if (vkCreateImageView(device, &ci, nullptr, &view) != VK_SUCCESS)
			throw std::runtime_error("failed to create image view!");
		return view;
	}

	// Copy a buffer into a VkImage (image must be in TRANSFER_DST_OPTIMAL layout).
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t w, uint32_t h)
	{
		VkCommandBuffer cmd = beginOneTimeCommands();
		VkBufferImageCopy region{};
		region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		region.imageExtent      = {w, h, 1};
		vkCmdCopyBufferToImage(cmd, buffer, image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
		endOneTimeCommands(cmd);
	}

	// -----------------------------------------------------------------------
	// Buffer helpers
	// -----------------------------------------------------------------------

	// Upload arbitrary data to a device-local buffer via a temporary staging buffer.
	// outBuffer / outMemory receive the final device-local buffer.
	// usage should be the intended usage flag (e.g. VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	// VK_BUFFER_USAGE_TRANSFER_DST_BIT is added automatically.
	void uploadStagedBuffer(const void* data, VkDeviceSize size,
		VkBufferUsageFlags usage,
		VkBuffer& outBuffer, VkDeviceMemory& outMemory)
	{
		VkBuffer staging; VkDeviceMemory stagingMem;
		createBuffer(size,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			staging, stagingMem);
		void* ptr;
		vkMapMemory(device, stagingMem, 0, size, 0, &ptr);
		std::memcpy(ptr, data, static_cast<size_t>(size));
		vkUnmapMemory(device, stagingMem);

		createBuffer(size,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			outBuffer, outMemory);
		copyBuffer(staging, outBuffer, size);
		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);
	}

	// -----------------------------------------------------------------------
	// Depth resource helpers
	// -----------------------------------------------------------------------

	// Create a depth image sized to the current swapchain extent.
	// Typical format: VK_FORMAT_D32_SFLOAT
	void createDepthResources(VkFormat format,
		VkImage& image, VkDeviceMemory& memory, VkImageView& view)
	{
		createImage(swapChainExtent.width, swapChainExtent.height, format,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			image, memory);
		view = createImageView(image, format, VK_IMAGE_ASPECT_DEPTH_BIT);
	}

	// Destroy depth resources created by createDepthResources() and null out handles.
	void destroyDepthResources(VkImage& image, VkDeviceMemory& memory, VkImageView& view)
	{
		vkDestroyImageView(device, view,   nullptr); view   = VK_NULL_HANDLE;
		vkDestroyImage    (device, image,  nullptr); image  = VK_NULL_HANDLE;
		vkFreeMemory      (device, memory, nullptr); memory = VK_NULL_HANDLE;
	}

	// -----------------------------------------------------------------------
	// Uniform buffer helpers
	// -----------------------------------------------------------------------

	// Create 'count' persistently-mapped uniform buffers of the given size.
	// All three output vectors are resized and filled.
	// Typical usage: createPersistentUBOs(sizeof(MyUBO), MAX_FRAMES_IN_FLIGHT, ...)
	void createPersistentUBOs(VkDeviceSize uboSize, uint32_t count,
		std::vector<VkBuffer>&       buffers,
		std::vector<VkDeviceMemory>& memories,
		std::vector<void*>&          mapped)
	{
		buffers .resize(count);
		memories.resize(count);
		mapped  .resize(count);
		for (uint32_t i = 0; i < count; ++i)
		{
			createBuffer(uboSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				buffers[i], memories[i]);
			vkMapMemory(device, memories[i], 0, uboSize, 0, &mapped[i]);
		}
	}

	// Destroy uniform buffers created by createPersistentUBOs() and clear vectors.
	void destroyUBOs(std::vector<VkBuffer>& buffers, std::vector<VkDeviceMemory>& memories)
	{
		for (size_t i = 0; i < buffers.size(); ++i)
		{
			vkDestroyBuffer(device, buffers[i],  nullptr);
			vkFreeMemory   (device, memories[i], nullptr);
		}
		buffers .clear();
		memories.clear();
	}

	// -----------------------------------------------------------------------
	// Sampler helper
	// -----------------------------------------------------------------------

	// Create a VkSampler with common parameters.
	// Pass enableCompare = true and a compareOp for shadow / depth sampler usage.
	VkSampler createSampler(
		VkFilter              filter      = VK_FILTER_LINEAR,
		VkSamplerAddressMode  addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		bool                  enableCompare = false,
		VkCompareOp           compareOp   = VK_COMPARE_OP_LESS)
	{
		VkSamplerCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		ci.magFilter    = filter;
		ci.minFilter    = filter;
		ci.addressModeU = addressMode;
		ci.addressModeV = addressMode;
		ci.addressModeW = addressMode;
		ci.compareEnable = enableCompare ? VK_TRUE : VK_FALSE;
		ci.compareOp     = compareOp;
		ci.borderColor   = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VkSampler sampler;
		if (vkCreateSampler(device, &ci, nullptr, &sampler) != VK_SUCCESS)
			throw std::runtime_error("failed to create sampler!");
		return sampler;
	}

private:
	// -----------------------------------------------------------------------
	// Window
	// -----------------------------------------------------------------------

	static void framebufferResizeCallback(GLFWwindow* w, int, int)
	{
		auto* app = reinterpret_cast<VkAppBase*>(glfwGetWindowUserPointer(w));
		app->framebufferResized = true;
	}

	void initWindow(uint32_t width, uint32_t height, const char* title)
	{
		if (!glfwInit())
			throw std::runtime_error("failed to initialise GLFW!");
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
		window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), title, nullptr, nullptr);
		if (!window)
			throw std::runtime_error("failed to create GLFW window!");
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	// -----------------------------------------------------------------------
	// Vulkan initialisation
	// -----------------------------------------------------------------------

	void initVulkan()
	{
		createInstance();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		onInitBeforeCommandPool();
		createCommandPool();
		onInitAfterCommandPool();
		createCommandBuffers();
		createSyncObjects();
	}

	// -----------------------------------------------------------------------
	// Instance
	// -----------------------------------------------------------------------

	void createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport())
			throw std::runtime_error("validation layers requested but not available!");

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "vk-example";
		appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_3;

		uint32_t glfwExtCount = 0;
		const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
		if (!glfwExts)
			throw std::runtime_error("GLFW: Vulkan not supported!");

		VkInstanceCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		ci.pApplicationInfo = &appInfo;
		ci.enabledExtensionCount = glfwExtCount;
		ci.ppEnabledExtensionNames = glfwExts;
		if (enableValidationLayers)
		{
			ci.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			ci.ppEnabledLayerNames = validationLayers.data();
		}
		if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
			throw std::runtime_error("failed to create Vulkan instance!");
	}

	bool checkValidationLayerSupport() const
	{
		uint32_t count = 0;
		vkEnumerateInstanceLayerProperties(&count, nullptr);
		std::vector<VkLayerProperties> available(count);
		vkEnumerateInstanceLayerProperties(&count, available.data());
		for (const char* name : validationLayers)
		{
			bool found = false;
			for (auto& lp : available)
				if (std::strcmp(name, lp.layerName) == 0)
				{
					found = true;
					break;
				}
			if (!found)
				return false;
		}
		return true;
	}

	// -----------------------------------------------------------------------
	// Surface
	// -----------------------------------------------------------------------

	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
			throw std::runtime_error("failed to create window surface!");
	}

	// -----------------------------------------------------------------------
	// Physical device selection
	// -----------------------------------------------------------------------

	void pickPhysicalDevice()
	{
		uint32_t count = 0;
		vkEnumeratePhysicalDevices(instance, &count, nullptr);
		if (count == 0)
			throw std::runtime_error("no Vulkan GPUs found!");

		std::vector<VkPhysicalDevice> devices(count);
		vkEnumeratePhysicalDevices(instance, &count, devices.data());
		for (auto& d : devices)
			if (isDeviceSuitable(d))
			{
				physicalDevice = d;
				break;
			}

		if (physicalDevice == VK_NULL_HANDLE)
			throw std::runtime_error("no suitable GPU found!");

		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
	}

	bool isDeviceSuitable(VkPhysicalDevice d) const
	{
		// Require Vulkan 1.3
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(d, &props);
		if (props.apiVersion < VK_API_VERSION_1_3)
			return false;

		// Require dynamic rendering & synchronisation2 (both Vulkan 1.3 core)
		VkPhysicalDeviceVulkan13Features f13{};
		f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
		VkPhysicalDeviceFeatures2 f2{};
		f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		f2.pNext = &f13;
		vkGetPhysicalDeviceFeatures2(d, &f2);
		if (!f13.dynamicRendering || !f13.synchronization2)
			return false;

		auto qi = findQueueFamilies(d);
		if (!qi.isComplete())
			return false;

		if (!checkDeviceExtensionSupport(d))
			return false;

		auto ss = querySwapChainSupport(d);
		return !ss.formats.empty() && !ss.presentModes.empty();
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice d) const
	{
		uint32_t c = 0;
		vkEnumerateDeviceExtensionProperties(d, nullptr, &c, nullptr);
		std::vector<VkExtensionProperties> avail(c);
		vkEnumerateDeviceExtensionProperties(d, nullptr, &c, avail.data());
		std::set<std::string> req(deviceExtensions.begin(), deviceExtensions.end());
		for (auto& e : avail)
			req.erase(e.extensionName);
		return req.empty();
	}

	// -----------------------------------------------------------------------
	// Queue families
	// -----------------------------------------------------------------------

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice d) const
	{
		QueueFamilyIndices idx;
		uint32_t c = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(d, &c, nullptr);
		std::vector<VkQueueFamilyProperties> fams(c);
		vkGetPhysicalDeviceQueueFamilyProperties(d, &c, fams.data());
		for (uint32_t i = 0; i < c; ++i)
		{
			if (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
				idx.graphicsFamily = i;
			VkBool32 ps = VK_FALSE;
			vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface, &ps);
			if (ps)
				idx.presentFamily = i;
			if (idx.isComplete())
				break;
		}
		return idx;
	}

	// -----------------------------------------------------------------------
	// Logical device
	// -----------------------------------------------------------------------

	void createLogicalDevice()
	{
		auto idx = findQueueFamilies(physicalDevice);
		std::vector<VkDeviceQueueCreateInfo> qcis;
		std::set<uint32_t> unique = {idx.graphicsFamily.value(), idx.presentFamily.value()};
		float prio = 1.0f;
		for (uint32_t f : unique)
		{
			VkDeviceQueueCreateInfo qi{};
			qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			qi.queueFamilyIndex = f;
			qi.queueCount = 1;
			qi.pQueuePriorities = &prio;
			qcis.push_back(qi);
		}

		// Enable Vulkan 1.3 core features
		VkPhysicalDeviceVulkan13Features f13{};
		f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
		f13.dynamicRendering = VK_TRUE;
		f13.synchronization2 = VK_TRUE;

		VkPhysicalDeviceFeatures2 f2{};
		f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		f2.pNext = &f13;

		VkDeviceCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		ci.pNext = &f2;
		ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
		ci.pQueueCreateInfos = qcis.data();
		ci.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		ci.ppEnabledExtensionNames = deviceExtensions.data();
		if (enableValidationLayers)
		{
			ci.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			ci.ppEnabledLayerNames = validationLayers.data();
		}
		if (vkCreateDevice(physicalDevice, &ci, nullptr, &device) != VK_SUCCESS)
			throw std::runtime_error("failed to create logical device!");

		vkGetDeviceQueue(device, idx.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, idx.presentFamily.value(), 0, &presentQueue);
	}

	// -----------------------------------------------------------------------
	// Swapchain
	// -----------------------------------------------------------------------

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice d) const
	{
		SwapChainSupportDetails det;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d, surface, &det.capabilities);

		uint32_t fc = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(d, surface, &fc, nullptr);
		if (fc)
		{
			det.formats.resize(fc);
			vkGetPhysicalDeviceSurfaceFormatsKHR(d, surface, &fc, det.formats.data());
		}

		uint32_t mc = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(d, surface, &mc, nullptr);
		if (mc)
		{
			det.presentModes.resize(mc);
			vkGetPhysicalDeviceSurfacePresentModesKHR(d, surface, &mc, det.presentModes.data());
		}
		return det;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& av) const
	{
		for (auto& f : av)
			if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				return f;
		return av[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& av) const
	{
		for (auto& m : av)
			if (m == VK_PRESENT_MODE_MAILBOX_KHR)
				return m;
		return VK_PRESENT_MODE_FIFO_KHR; // guaranteed available
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps) const
	{
		if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
			return caps.currentExtent;
		int w = 0, h = 0;
		glfwGetFramebufferSize(window, &w, &h);
		return {
			std::clamp(static_cast<uint32_t>(w), caps.minImageExtent.width, caps.maxImageExtent.width),
			std::clamp(static_cast<uint32_t>(h), caps.minImageExtent.height, caps.maxImageExtent.height),
		};
	}

	void createSwapChain()
	{
		auto sup = querySwapChainSupport(physicalDevice);
		auto fmt = chooseSwapSurfaceFormat(sup.formats);
		auto pm = chooseSwapPresentMode(sup.presentModes);
		auto ext = chooseSwapExtent(sup.capabilities);

		uint32_t ic = sup.capabilities.minImageCount + 1;
		if (sup.capabilities.maxImageCount > 0 && ic > sup.capabilities.maxImageCount)
			ic = sup.capabilities.maxImageCount;

		VkSwapchainCreateInfoKHR ci{};
		ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		ci.surface = surface;
		ci.minImageCount = ic;
		ci.imageFormat = fmt.format;
		ci.imageColorSpace = fmt.colorSpace;
		ci.imageExtent = ext;
		ci.imageArrayLayers = 1;
		ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		auto idx = findQueueFamilies(physicalDevice);
		uint32_t fi[] = {idx.graphicsFamily.value(), idx.presentFamily.value()};
		if (idx.graphicsFamily != idx.presentFamily)
		{
			ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			ci.queueFamilyIndexCount = 2;
			ci.pQueueFamilyIndices = fi;
		}
		else
		{
			ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		ci.preTransform = sup.capabilities.currentTransform;
		ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		ci.presentMode = pm;
		ci.clipped = VK_TRUE;
		ci.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &ci, nullptr, &swapChain) != VK_SUCCESS)
			throw std::runtime_error("failed to create swap chain!");

		vkGetSwapchainImagesKHR(device, swapChain, &ic, nullptr);
		swapChainImages.resize(ic);
		vkGetSwapchainImagesKHR(device, swapChain, &ic, swapChainImages.data());
		swapChainImageFormat = fmt.format;
		swapChainExtent = ext;
	}

	void createImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); ++i)
		{
			VkImageViewCreateInfo ci{};
			ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			ci.image = swapChainImages[i];
			ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
			ci.format = swapChainImageFormat;
			ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			if (vkCreateImageView(device, &ci, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create image view!");
		}
	}

	// -----------------------------------------------------------------------
	// Swapchain recreation (resize handling)
	// -----------------------------------------------------------------------

	void cleanupSwapChain()
	{
		onCleanupSwapChain(); // derived class destroys depth / other swapchain-sized resources

		for (auto sem : renderFinishedSemaphores)
			vkDestroySemaphore(device, sem, nullptr);
		renderFinishedSemaphores.clear();

		for (auto view : swapChainImageViews)
			vkDestroyImageView(device, view, nullptr);
		swapChainImageViews.clear();

		vkDestroySwapchainKHR(device, swapChain, nullptr);
		swapChain = VK_NULL_HANDLE;
	}

	void recreateSwapChain()
	{
		// Wait until the window has a non-zero size (handles minimisation)
		int w = 0, h = 0;
		glfwGetFramebufferSize(window, &w, &h);
		while (w == 0 || h == 0)
		{
			glfwGetFramebufferSize(window, &w, &h);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);
		cleanupSwapChain();
		createSwapChain();
		createImageViews();
		onRecreateSwapChain(); // derived class recreates depth / other swapchain-sized resources

		// Recreate per-swapchain-image semaphores (image count may have changed)
		VkSemaphoreCreateInfo si{};
		si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		renderFinishedSemaphores.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); ++i)
			if (vkCreateSemaphore(device, &si, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to recreate per-image semaphore!");
	}

	// -----------------------------------------------------------------------
	// Commands
	// -----------------------------------------------------------------------

	void createCommandPool()
	{
		auto idx = findQueueFamilies(physicalDevice);
		VkCommandPoolCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		ci.queueFamilyIndex = idx.graphicsFamily.value();
		if (vkCreateCommandPool(device, &ci, nullptr, &commandPool) != VK_SUCCESS)
			throw std::runtime_error("failed to create command pool!");
	}

	void createCommandBuffers()
	{
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo ai{};
		ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ai.commandPool = commandPool;
		ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ai.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
		if (vkAllocateCommandBuffers(device, &ai, commandBuffers.data()) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate command buffers!");
	}

	// -----------------------------------------------------------------------
	// Synchronisation objects
	// -----------------------------------------------------------------------

	void createSyncObjects()
	{
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo si{};
		si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		VkFenceCreateInfo fi{};
		fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
			if (vkCreateSemaphore(device, &si, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fi, nullptr, &inFlightFences[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create per-frame sync objects!");

		// One render-finished semaphore per swapchain image.
		// The presentation engine holds a reference to this semaphore until the
		// corresponding image is re-acquired, so we cannot safely reuse it
		// based on frame-in-flight index alone (the swapchain may have more
		// images than MAX_FRAMES_IN_FLIGHT).  Indexing by imageIndex guarantees
		// the semaphore is only re-signalled after its image has been acquired
		// again, which implicitly means the previous present has consumed it.
		renderFinishedSemaphores.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); ++i)
			if (vkCreateSemaphore(device, &si, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create per-image semaphore!");
	}

	// -----------------------------------------------------------------------
	// Frame submission  (Vulkan 1.3 – Synchronization2 / VkSubmitInfo2)
	// -----------------------------------------------------------------------

	void drawFrame()
	{
		// Wait for the previous frame using this slot to finish
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		// Acquire next image – do NOT reset the fence yet
		uint32_t imageIndex = 0;
		VkResult result = vkAcquireNextImageKHR(
			device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapChain();
			return;
		}
		if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
			throw std::runtime_error("failed to acquire swap chain image!");

		// Only reset the fence after we know we will submit work
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		// Allow derived class to update per-frame data (e.g. uniform buffers)
		onBeforeRecord(currentFrame);

		// Record
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		// Submit  (Synchronization2 – VkSubmitInfo2 / vkQueueSubmit2)
		VkSemaphoreSubmitInfo waitSem{};
		waitSem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
		waitSem.semaphore = imageAvailableSemaphores[currentFrame];
		waitSem.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkCommandBufferSubmitInfo cmdSub{};
		cmdSub.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
		cmdSub.commandBuffer = commandBuffers[currentFrame];

		VkSemaphoreSubmitInfo sigSem{};
		sigSem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
		sigSem.semaphore = renderFinishedSemaphores[imageIndex];
		sigSem.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

		VkSubmitInfo2 submit{};
		submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
		submit.waitSemaphoreInfoCount = 1;
		submit.pWaitSemaphoreInfos = &waitSem;
		submit.commandBufferInfoCount = 1;
		submit.pCommandBufferInfos = &cmdSub;
		submit.signalSemaphoreInfoCount = 1;
		submit.pSignalSemaphoreInfos = &sigSem;

		if (vkQueueSubmit2(graphicsQueue, 1, &submit, inFlightFences[currentFrame]) != VK_SUCCESS)
			throw std::runtime_error("failed to submit draw command buffer!");

		// Present
		VkSwapchainKHR chains[] = {swapChain};
		VkPresentInfoKHR pi{};
		pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		pi.waitSemaphoreCount = 1;
		pi.pWaitSemaphores = &renderFinishedSemaphores[imageIndex];
		pi.swapchainCount = 1;
		pi.pSwapchains = chains;
		pi.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &pi);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS)
			throw std::runtime_error("failed to present!");

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	// -----------------------------------------------------------------------
	// Main loop
	// -----------------------------------------------------------------------

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}
		vkDeviceWaitIdle(device);
	}

	// -----------------------------------------------------------------------
	// Cleanup
	// -----------------------------------------------------------------------

	void cleanup()
	{
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		onCleanup(); // derived class destroys pipelines, buffers, descriptors, etc.

		cleanupSwapChain(); // destroys depth (via onCleanupSwapChain), semaphores, views, swapchain

		vkDestroyDevice(device, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);
		glfwTerminate();
	}
};
