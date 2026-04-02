add_executable(01_clear_screen main.cpp)

# Link Vulkan and our fetched GLFW library
target_link_libraries(01_clear_screen PRIVATE
    Vulkan::Vulkan
    glfw
)