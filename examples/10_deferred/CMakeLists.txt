add_executable(10_deferred main.cpp)

target_link_libraries(10_deferred PRIVATE vk_common)

find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/bin")
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found — install the Vulkan SDK or add glslc to PATH")
endif()

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SPV_DIR    ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPV_DIR})

set(SHADERS
    ${SHADER_DIR}/geometry.vert
    ${SHADER_DIR}/geometry.frag
    ${SHADER_DIR}/fullscreen.vert
    ${SHADER_DIR}/lighting.frag
)

foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV_FILE ${SPV_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT  ${SPV_FILE}
        COMMAND ${GLSLC} ${SHADER} -o ${SPV_FILE}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} -> SPIR-V"
    )
    list(APPEND SPV_FILES ${SPV_FILE})
endforeach()

add_custom_target(10_shaders DEPENDS ${SPV_FILES})
add_dependencies(10_deferred 10_shaders)

target_compile_definitions(10_deferred PRIVATE
    SHADER_DIR="${SPV_DIR}"
)
