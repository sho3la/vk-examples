# ---------------------------------------------------------------------------
# vk_common – header-only INTERFACE library
#
# Provides VkAppBase and all shared Vulkan 1.3 boilerplate.
# Link every example against this target instead of linking Vulkan, GLFW,
# and GLM individually.
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
