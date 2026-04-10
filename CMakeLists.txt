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

# Fetch GLM (header-only math library)
message(STATUS "Fetching GLM...")
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# Fetch stb (single-header image loading / writing)
message(STATUS "Fetching stb...")
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(stb)
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

# Find Vulkan (User must have the Vulkan SDK installed)
find_package(Vulkan REQUIRED)

# ---------------------------------------------------------
# EXAMPLES
# ---------------------------------------------------------
add_subdirectory(examples/common)
add_subdirectory(examples/01_clear_screen)
add_subdirectory(examples/02_rotating_cube)
add_subdirectory(examples/03_point_lights)
add_subdirectory(examples/04_spot_lights)
add_subdirectory(examples/05_texture)
add_subdirectory(examples/06_texture_stb)
add_subdirectory(examples/07_normal_mapping)
add_subdirectory(examples/08_shadow_mapping)
add_subdirectory(examples/09_bloom_hdr)