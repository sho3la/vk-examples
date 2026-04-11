# ---------------------------------------------------------------------------
# vk_common – header-only INTERFACE library
#
# Provides shared Vulkan 1.3 boilerplate for all examples.
# Link every example against this target instead of linking Vulkan, GLFW,
# and GLM individually.
#
# Headers available after linking:
#   vk_common.h      – VkAppBase: window, device, swapchain, frame loop,
#                      and common helpers (createImage, uploadStagedBuffer,
#                      createDepthResources, createPersistentUBOs, createSampler, …)
#   vk_pipelines.h   – GraphicsPipelineBuilder: fluent graphics pipeline creation
#   vk_descriptors.h – DescriptorLayoutBuilder, DescriptorPoolBuilder,
#                      DescriptorWriter: eliminate descriptor boilerplate
# ---------------------------------------------------------------------------

add_library(vk_common INTERFACE)

target_include_directories(vk_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(vk_common INTERFACE
    Vulkan::Vulkan
    glfw
    glm::glm
)
