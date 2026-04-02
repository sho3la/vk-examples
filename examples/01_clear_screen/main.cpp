
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr uint32_t INITIAL_WIDTH = 800;
constexpr uint32_t INITIAL_HEIGHT = 600;
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
// Application
// ---------------------------------------------------------------------------

class ClearScreenApp
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
	// Window ---
	GLFWwindow* window = nullptr;
	bool framebufferResized = false;

	// Core Vulkan ---
	VkInstance instance = VK_NULL_HANDLE;
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue graphicsQueue = VK_NULL_HANDLE;
	VkQueue presentQueue = VK_NULL_HANDLE;

	// Swapchain ---
	VkSwapchainKHR swapChain = VK_NULL_HANDLE;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat{};
	VkExtent2D swapChainExtent{};
	std::vector<VkImageView> swapChainImageViews;

	// Commands ---
	VkCommandPool commandPool = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> commandBuffers;

	// Synchronisation ---
	std::vector<VkSemaphore> imageAvailableSemaphores; // per frame-in-flight
	std::vector<VkSemaphore> renderFinishedSemaphores; // per swapchain image
	std::vector<VkFence> inFlightFences;			   // per frame-in-flight

	uint32_t currentFrame = 0;

	// -----------------------------------------------------------------------
	// Window
	// -----------------------------------------------------------------------

	static void framebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/)
	{
		auto* app = reinterpret_cast<ClearScreenApp*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initWindow()
	{
		if (!glfwInit())
			throw std::runtime_error("failed to initialise GLFW!");

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(
			INITIAL_WIDTH, INITIAL_HEIGHT, "vk-examples: 01 Clear Screen", nullptr, nullptr);

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
		createCommandPool();
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
		appInfo.pApplicationName = "01 Clear Screen";
		appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_3;

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		if (!glfwExtensions)
			throw std::runtime_error("GLFW: Vulkan not supported (glfwGetRequiredInstanceExtensions returned NULL)!");

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledExtensionCount = glfwExtensionCount;
		createInfo.ppEnabledExtensionNames = glfwExtensions;

		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
			throw std::runtime_error("failed to create Vulkan instance!");
	}

	bool checkValidationLayerSupport() const
	{
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> available(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, available.data());

		for (const char* name : validationLayers)
		{
			bool found = false;
			for (const auto& layer : available)
			{
				if (std::strcmp(name, layer.layerName) == 0)
				{
					found = true;
					break;
				}
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
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if (deviceCount == 0)
			throw std::runtime_error("failed to find GPUs with Vulkan support!");

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& candidate : devices)
		{
			if (isDeviceSuitable(candidate))
			{
				physicalDevice = candidate;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
			throw std::runtime_error("failed to find a suitable GPU!");
	}

	bool isDeviceSuitable(VkPhysicalDevice candidate) const
	{
		// ---- Require Vulkan 1.3 support ----
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(candidate, &props);
		if (props.apiVersion < VK_API_VERSION_1_3)
			return false;

		// ---- Require dynamic rendering & synchronisation2 ----
		VkPhysicalDeviceVulkan13Features features13{};
		features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

		VkPhysicalDeviceFeatures2 features2{};
		features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		features2.pNext = &features13;
		vkGetPhysicalDeviceFeatures2(candidate, &features2);

		if (!features13.dynamicRendering || !features13.synchronization2)
			return false;

		// ---- Queue families ----
		QueueFamilyIndices indices = findQueueFamilies(candidate);
		if (!indices.isComplete())
			return false;

		// ---- Swapchain extension + adequacy ----
		if (!checkDeviceExtensionSupport(candidate))
			return false;

		SwapChainSupportDetails swapSupport = querySwapChainSupport(candidate);
		return !swapSupport.formats.empty() && !swapSupport.presentModes.empty();
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice candidate) const
	{
		uint32_t count = 0;
		vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr);
		std::vector<VkExtensionProperties> available(count);
		vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, available.data());

		std::set<std::string> required(deviceExtensions.begin(), deviceExtensions.end());
		for (const auto& ext : available)
			required.erase(ext.extensionName);

		return required.empty();
	}

	// -----------------------------------------------------------------------
	// Queue families
	// -----------------------------------------------------------------------

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice candidate) const
	{
		QueueFamilyIndices indices;

		uint32_t count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
		std::vector<VkQueueFamilyProperties> families(count);
		vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, families.data());

		for (uint32_t i = 0; i < count; ++i)
		{
			if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
				indices.graphicsFamily = i;

			VkBool32 presentSupport = VK_FALSE;
			vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface, &presentSupport);
			if (presentSupport)
				indices.presentFamily = i;

			if (indices.isComplete())
				break;
		}
		return indices;
	}

	// -----------------------------------------------------------------------
	// Logical device
	// -----------------------------------------------------------------------

	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = {
			indices.graphicsFamily.value(),
			indices.presentFamily.value(),
		};

		float queuePriority = 1.0f;
		for (uint32_t family : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			info.queueFamilyIndex = family;
			info.queueCount = 1;
			info.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(info);
		}

		// Enable Vulkan 1.3 core features we need
		VkPhysicalDeviceVulkan13Features features13{};
		features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
		features13.dynamicRendering = VK_TRUE;
		features13.synchronization2 = VK_TRUE;

		VkPhysicalDeviceFeatures2 features2{};
		features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		features2.pNext = &features13;

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pNext = &features2;
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		// Device-level validation layers are deprecated but still set for back-compat
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
			throw std::runtime_error("failed to create logical device!");

		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	// -----------------------------------------------------------------------
	// Swapchain
	// -----------------------------------------------------------------------

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice candidate) const
	{
		SwapChainSupportDetails details;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(candidate, surface, &details.capabilities);

		uint32_t formatCount = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface, &formatCount, nullptr);
		if (formatCount)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface, &formatCount, details.formats.data());
		}

		uint32_t modeCount = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface, &modeCount, nullptr);
		if (modeCount)
		{
			details.presentModes.resize(modeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface, &modeCount, details.presentModes.data());
		}
		return details;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available) const
	{
		for (const auto& fmt : available)
		{
			if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				return fmt;
		}
		return available[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available) const
	{
		for (const auto& mode : available)
		{
			if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
				return mode;
		}
		return VK_PRESENT_MODE_FIFO_KHR; // guaranteed available
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps) const
	{
		if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
			return caps.currentExtent;

		int w = 0, h = 0;
		glfwGetFramebufferSize(window, &w, &h);

		VkExtent2D extent = {
			static_cast<uint32_t>(w),
			static_cast<uint32_t>(h),
		};
		extent.width = std::clamp(extent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
		extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
		return extent;
	}

	void createSwapChain()
	{
		SwapChainSupportDetails support = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(support.presentModes);
		VkExtent2D extent = chooseSwapExtent(support.capabilities);

		uint32_t imageCount = support.capabilities.minImageCount + 1;
		if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
		{
			imageCount = support.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t familyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

		if (indices.graphicsFamily != indices.presentFamily)
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = familyIndices;
		}
		else
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = support.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
			throw std::runtime_error("failed to create swap chain!");

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); ++i)
		{
			VkImageViewCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			info.image = swapChainImages[i];
			info.viewType = VK_IMAGE_VIEW_TYPE_2D;
			info.format = swapChainImageFormat;
			info.components = {
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
			};
			info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			info.subresourceRange.baseMipLevel = 0;
			info.subresourceRange.levelCount = 1;
			info.subresourceRange.baseArrayLayer = 0;
			info.subresourceRange.layerCount = 1;

			if (vkCreateImageView(device, &info, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create image view!");
		}
	}

	// -----------------------------------------------------------------------
	// Swapchain recreation (resize handling)
	// -----------------------------------------------------------------------

	void cleanupSwapChain()
	{
		// Destroy per-swapchain-image semaphores
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
		// Handle minimisation — wait until the window has a non-zero size
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

		// Recreate per-swapchain-image semaphores (image count may have changed)
		VkSemaphoreCreateInfo semInfo{};
		semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		renderFinishedSemaphores.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); ++i)
		{
			if (vkCreateSemaphore(device, &semInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to recreate per-image semaphore!");
		}
	}

	// -----------------------------------------------------------------------
	// Commands
	// -----------------------------------------------------------------------

	void createCommandPool()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = indices.graphicsFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
			throw std::runtime_error("failed to create command pool!");
	}

	void createCommandBuffers()
	{
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate command buffers!");
	}

	// -----------------------------------------------------------------------
	// Synchronisation objects
	// -----------------------------------------------------------------------

	void createSyncObjects()
	{
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semInfo{};
		semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			if (vkCreateSemaphore(device, &semInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create per-frame synchronisation objects!");
			}
		}

		// One render-finished semaphore per swapchain image.
		// The presentation engine holds a reference to this semaphore until the
		// corresponding image is re-acquired, so we cannot safely reuse it
		// based on frame-in-flight index alone (the swapchain may have more
		// images than MAX_FRAMES_IN_FLIGHT).  Indexing by imageIndex guarantees
		// the semaphore is only re-signalled after its image has been acquired
		// again, which implicitly means the previous present has consumed it.
		renderFinishedSemaphores.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); ++i)
		{
			if (vkCreateSemaphore(device, &semInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create per-image semaphore!");
		}
	}

	// -----------------------------------------------------------------------
	// Image layout transitions  (Vulkan 1.3 — vkCmdPipelineBarrier2)
	// -----------------------------------------------------------------------

	static void transitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
		VkImageLayout newLayout, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
		VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess)
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
		barrier.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		};

		VkDependencyInfo dep{};
		dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dep.imageMemoryBarrierCount = 1;
		dep.pImageMemoryBarriers = &barrier;

		vkCmdPipelineBarrier2(cmd, &dep);
	}

	// -----------------------------------------------------------------------
	// Command buffer recording
	// -----------------------------------------------------------------------

	void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
			throw std::runtime_error("failed to begin recording command buffer!");

		// 1. UNDEFINED  ->  COLOR_ATTACHMENT_OPTIMAL
		transitionImageLayout(cmd, swapChainImages[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
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

		// 3. COLOR_ATTACHMENT_OPTIMAL  ->  PRESENT_SRC_KHR
		transitionImageLayout(cmd, swapChainImages[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE);

		if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
			throw std::runtime_error("failed to record command buffer!");
	}

	// -----------------------------------------------------------------------
	// Frame submission  (Vulkan 1.3 — Synchronization2 / VkSubmitInfo2)
	// -----------------------------------------------------------------------

	void drawFrame()
	{
		// Wait for the previous frame using this slot to finish
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		// Acquire next image — do NOT reset the fence yet
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

		// Record
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		// Submit  (Synchronization2 — VkSubmitInfo2 / vkQueueSubmit2)
		VkSemaphoreSubmitInfo waitSemaphore{};
		waitSemaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
		waitSemaphore.semaphore = imageAvailableSemaphores[currentFrame];
		waitSemaphore.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkCommandBufferSubmitInfo cmdSubmit{};
		cmdSubmit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
		cmdSubmit.commandBuffer = commandBuffers[currentFrame];

		VkSemaphoreSubmitInfo signalSemaphore{};
		signalSemaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
		signalSemaphore.semaphore = renderFinishedSemaphores[imageIndex];
		signalSemaphore.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

		VkSubmitInfo2 submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
		submitInfo.waitSemaphoreInfoCount = 1;
		submitInfo.pWaitSemaphoreInfos = &waitSemaphore;
		submitInfo.commandBufferInfoCount = 1;
		submitInfo.pCommandBufferInfos = &cmdSubmit;
		submitInfo.signalSemaphoreInfoCount = 1;
		submitInfo.pSignalSemaphoreInfos = &signalSemaphore;

		if (vkQueueSubmit2(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
			throw std::runtime_error("failed to submit draw command buffer!");

		// Present
		VkSwapchainKHR swapChains[] = {swapChain};

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinishedSemaphores[imageIndex];
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image!");
		}

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
		// Per-frame objects
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		// Destroys image views, swapchain, and per-image semaphores
		cleanupSwapChain();

		vkDestroyDevice(device, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);
		glfwTerminate();
	}
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main()
{
	ClearScreenApp app;

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