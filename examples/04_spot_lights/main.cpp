
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

constexpr uint32_t INITIAL_WIDTH = 1024;
constexpr uint32_t INITIAL_HEIGHT = 768;
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
// Vertex data — ground plane on XZ, facing Y+
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
	0,
	1,
	2,
	2,
	3,
	0,
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
// Application
// ---------------------------------------------------------------------------

class SpotLightsApp
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window = nullptr;
	bool framebufferResized = false;

	VkInstance instance = VK_NULL_HANDLE;
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue graphicsQueue = VK_NULL_HANDLE;
	VkQueue presentQueue = VK_NULL_HANDLE;
	VkPhysicalDeviceMemoryProperties memProperties{};

	VkSwapchainKHR swapChain = VK_NULL_HANDLE;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat{};
	VkExtent2D swapChainExtent{};
	std::vector<VkImageView> swapChainImageViews;

	VkImage depthImage = VK_NULL_HANDLE;
	VkDeviceMemory depthMemory = VK_NULL_HANDLE;
	VkImageView depthImageView = VK_NULL_HANDLE;
	VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline graphicsPipeline = VK_NULL_HANDLE;

	VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> descriptorSets;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	VkBuffer vertexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
	VkBuffer indexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;

	VkCommandPool commandPool = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> commandBuffers;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	uint32_t currentFrame = 0;

	// === WINDOW ===========================================================

	static void framebufferResizeCallback(GLFWwindow* w, int, int)
	{
		auto* app = reinterpret_cast<SpotLightsApp*>(glfwGetWindowUserPointer(w));
		app->framebufferResized = true;
	}

	void initWindow()
	{
		if (!glfwInit())
			throw std::runtime_error("failed to initialise GLFW!");
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
		window = glfwCreateWindow(
			INITIAL_WIDTH, INITIAL_HEIGHT, "vk-examples: 04 Spot Lights", nullptr, nullptr);
		if (!window)
			throw std::runtime_error("failed to create GLFW window!");
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	// === VULKAN INIT ======================================================

	void initVulkan()
	{
		createInstance();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createDepthResources();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
	}

	// === INSTANCE =========================================================

	void createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport())
			throw std::runtime_error("validation layers requested but not available!");

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "04 Spot Lights";
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

	// === SURFACE ==========================================================

	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
			throw std::runtime_error("failed to create window surface!");
	}

	// === PHYSICAL DEVICE ==================================================

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
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(d, &props);
		if (props.apiVersion < VK_API_VERSION_1_3)
			return false;
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

	// === QUEUE FAMILIES ===================================================

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

	// === LOGICAL DEVICE ===================================================

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

	// === SWAPCHAIN ========================================================

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
		return VK_PRESENT_MODE_FIFO_KHR;
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

	// === DEPTH BUFFER =====================================================

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

	// === MEMORY HELPERS ===================================================

	uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const
	{
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
			if ((typeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & props) == props)
				return i;
		throw std::runtime_error("failed to find suitable memory type!");
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps, VkBuffer& buffer,
		VkDeviceMemory& memory)
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

	void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
	{
		VkCommandBufferAllocateInfo ai{};
		ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ai.commandPool = commandPool;
		ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ai.commandBufferCount = 1;
		VkCommandBuffer cmd;
		vkAllocateCommandBuffers(device, &ai, &cmd);
		VkCommandBufferBeginInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &bi);
		VkBufferCopy region{};
		region.size = size;
		vkCmdCopyBuffer(cmd, src, dst, 1, &region);
		vkEndCommandBuffer(cmd);
		VkSubmitInfo si{};
		si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		si.commandBufferCount = 1;
		si.pCommandBuffers = &cmd;
		vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);
		vkFreeCommandBuffers(device, commandPool, 1, &cmd);
	}

	// === VERTEX & INDEX BUFFERS ===========================================

	void createVertexBuffer()
	{
		VkDeviceSize size = sizeof(groundVertices[0]) * groundVertices.size();
		VkBuffer staging;
		VkDeviceMemory stagingMem;
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
		void* data;
		vkMapMemory(device, stagingMem, 0, size, 0, &data);
		std::memcpy(data, groundVertices.data(), size);
		vkUnmapMemory(device, stagingMem);
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
		copyBuffer(staging, vertexBuffer, size);
		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);
	}

	void createIndexBuffer()
	{
		VkDeviceSize size = sizeof(groundIndices[0]) * groundIndices.size();
		VkBuffer staging;
		VkDeviceMemory stagingMem;
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
		void* data;
		vkMapMemory(device, stagingMem, 0, size, 0, &data);
		std::memcpy(data, groundIndices.data(), size);
		vkUnmapMemory(device, stagingMem);
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
		copyBuffer(staging, indexBuffer, size);
		vkDestroyBuffer(device, staging, nullptr);
		vkFreeMemory(device, stagingMem, nullptr);
	}

	// === UNIFORM BUFFERS ==================================================

	void createUniformBuffers()
	{
		VkDeviceSize size = sizeof(LightUBO);
		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i],
				uniformBuffersMemory[i]);
			vkMapMemory(device, uniformBuffersMemory[i], 0, size, 0, &uniformBuffersMapped[i]);
		}
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

		// Red — left
		ubo.lights[0].position = glm::vec4(-spacing, height, drift, 0.0f);
		ubo.lights[0].direction = glm::vec4(downDir, 0.0f);
		ubo.lights[0].color = glm::vec4(1.0f, 0.1f, 0.1f, 10.0f);
		ubo.lights[0].cutoffs = glm::vec4(std::cos(innerAngle), std::cos(outerAngle), 0.0f, 0.0f);

		// Green — center
		ubo.lights[1].position = glm::vec4(0.0f, height, -drift, 0.0f);
		ubo.lights[1].direction = glm::vec4(downDir, 0.0f);
		ubo.lights[1].color = glm::vec4(0.1f, 1.0f, 0.1f, 10.0f);
		ubo.lights[1].cutoffs = glm::vec4(std::cos(innerAngle), std::cos(outerAngle), 0.0f, 0.0f);

		// Blue — right
		ubo.lights[2].position = glm::vec4(spacing, height, drift * 0.5f, 0.0f);
		ubo.lights[2].direction = glm::vec4(downDir, 0.0f);
		ubo.lights[2].color = glm::vec4(0.2f, 0.3f, 1.0f, 10.0f);
		ubo.lights[2].cutoffs = glm::vec4(std::cos(innerAngle), std::cos(outerAngle), 0.0f, 0.0f);

		ubo.viewPos = glm::vec4(0.0f, 8.0f, 14.0f, 0.0f);
		ubo.ambient = glm::vec4(0.7f, 0.7f, 0.8f, 0.03f);

		std::memcpy(uniformBuffersMapped[frameIndex], &ubo, sizeof(ubo));
	}

	// === DESCRIPTORS ======================================================

	void createDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding uboBinding{};
		uboBinding.binding = 0;
		uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboBinding.descriptorCount = 1;
		uboBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = 1;
		ci.pBindings = &uboBinding;
		if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &descriptorSetLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor set layout!");
	}

	void createDescriptorPool()
	{
		VkDescriptorPoolSize poolSize{};
		poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		VkDescriptorPoolCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		ci.poolSizeCount = 1;
		ci.pPoolSizes = &poolSize;
		ci.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		if (vkCreateDescriptorPool(device, &ci, nullptr, &descriptorPool) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor pool!");
	}

	void createDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		VkDescriptorSetAllocateInfo ai{};
		ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		ai.descriptorPool = descriptorPool;
		ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		ai.pSetLayouts = layouts.data();

		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &ai, descriptorSets.data()) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate descriptor sets!");

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			VkDescriptorBufferInfo bufInfo{};
			bufInfo.buffer = uniformBuffers[i];
			bufInfo.offset = 0;
			bufInfo.range = sizeof(LightUBO);

			VkWriteDescriptorSet write{};
			write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet = descriptorSets[i];
			write.dstBinding = 0;
			write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			write.descriptorCount = 1;
			write.pBufferInfo = &bufInfo;
			vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
		}
	}

	// === SHADER LOADING ===================================================

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

	// === GRAPHICS PIPELINE ================================================

	void createGraphicsPipeline()
	{
		auto vertCode = readFile(std::string(SHADER_DIR) + "/spot.vert.spv");
		auto fragCode = readFile(std::string(SHADER_DIR) + "/spot.frag.spv");
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
		attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};

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
		pcRange.size = sizeof(PushConstants);

		VkPipelineLayoutCreateInfo layoutCI{};
		layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutCI.setLayoutCount = 1;
		layoutCI.pSetLayouts = &descriptorSetLayout;
		layoutCI.pushConstantRangeCount = 1;
		layoutCI.pPushConstantRanges = &pcRange;
		if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &pipelineLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout!");

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

	// === SWAPCHAIN RECREATION =============================================

	void cleanupSwapChain()
	{
		cleanupDepthResources();
		for (auto sem : renderFinishedSemaphores)
			vkDestroySemaphore(device, sem, nullptr);
		renderFinishedSemaphores.clear();
		for (auto v : swapChainImageViews)
			vkDestroyImageView(device, v, nullptr);
		swapChainImageViews.clear();
		vkDestroySwapchainKHR(device, swapChain, nullptr);
		swapChain = VK_NULL_HANDLE;
	}

	void recreateSwapChain()
	{
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
		createDepthResources();
		VkSemaphoreCreateInfo si{};
		si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		renderFinishedSemaphores.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); ++i)
			if (vkCreateSemaphore(device, &si, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to recreate per-image semaphore!");
	}

	// === COMMANDS =========================================================

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

	// === SYNC OBJECTS =====================================================

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
		renderFinishedSemaphores.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); ++i)
			if (vkCreateSemaphore(device, &si, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create per-image semaphore!");
	}

	// === IMAGE TRANSITIONS ================================================

	static void transitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
		VkImageLayout newLayout, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
		VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT)
	{
		VkImageMemoryBarrier2 barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		barrier.srcStageMask = srcStage;
		barrier.srcAccessMask = srcAccess;
		barrier.dstStageMask = dstStage;
		barrier.dstAccessMask = dstAccess;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange = {aspect, 0, 1, 0, 1};
		VkDependencyInfo dep{};
		dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dep.imageMemoryBarrierCount = 1;
		dep.pImageMemoryBarriers = &barrier;
		vkCmdPipelineBarrier2(cmd, &dep);
	}

	// === COMMAND BUFFER RECORDING =========================================

	void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
	{
		VkCommandBufferBeginInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
			throw std::runtime_error("failed to begin recording command buffer!");

		transitionImageLayout(cmd, swapChainImages[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

		transitionImageLayout(cmd, depthImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
			VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

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

		transitionImageLayout(cmd, swapChainImages[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE);

		if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
			throw std::runtime_error("failed to record command buffer!");
	}

	// === FRAME SUBMISSION =================================================

	void drawFrame()
	{
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

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

		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		updateUniformBuffer(currentFrame);

		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

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

	// === MAIN LOOP ========================================================

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}
		vkDeviceWaitIdle(device);
	}

	// === CLEANUP ==========================================================

	void cleanup()
	{
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}
		vkDestroyCommandPool(device, commandPool, nullptr);

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);
		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

		cleanupSwapChain();
		vkDestroyDevice(device, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}
};

// ===========================================================================

int main()
{
	SpotLightsApp app;
	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << "Fatal Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}