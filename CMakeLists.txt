cmake_minimum_required(VERSION 3.15)
project(vk-examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------
# DEPENDENCIES via FetchContent
# ---------------------------------------------------------
include(FetchContent)

# Fetch GLFW
message(STATUS "Fetching GLFW...")
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)

# Disable GLFW extras to keep the build fast and clean
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(glfw)

# Find Vulkan (User must have the Vulkan SDK installed)
find_package(Vulkan REQUIRED)

# ---------------------------------------------------------
# EXAMPLES
# ---------------------------------------------------------
add_subdirectory(examples/01_clear_screen)