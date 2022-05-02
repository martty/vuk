cmake_minimum_required(VERSION 3.7)
project(vuk-examples)

if(NOT VUK_USE_SHADERC)
    message(FATAL_ERROR "Building vuk examples require shaderc for building shaders, enable VUK_USE_SHADERC")
endif()

FetchContent_Declare(
  vk-bootstrap 
  GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
  GIT_TAG        cf8df11a0a071463009031cb474099dacffe90ed
)
FetchContent_MakeAvailable(vk-bootstrap)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  glfw 
  GIT_REPOSITORY https://github.com/glfw/glfw
  GIT_TAG        3.3.6
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm
  GIT_TAG        0.9.9.8
)
FetchContent_MakeAvailable(glm)

SET(imgui_sources ../ext/imgui/imgui.cpp ../ext/imgui/imgui_draw.cpp ../ext/imgui/imgui_demo.cpp ../ext/imgui/imgui_widgets.cpp ../ext/imgui/imgui_tables.cpp ../ext/imgui/backends/imgui_impl_glfw.cpp)

add_executable(vuk_all_examples)
target_include_directories(vuk_all_examples SYSTEM PRIVATE ../ext/stb ../ext/imgui)
target_sources(vuk_all_examples PRIVATE example_browser.cpp imgui.cpp stbi.cpp ${imgui_sources})
target_compile_definitions(vuk_all_examples PRIVATE GLM_FORCE_SIZE_FUNC GLM_FORCE_EXPLICIT_CTOR GLM_ENABLE_EXPERIMENTAL GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_link_libraries(vuk_all_examples PRIVATE vuk)
target_link_libraries(vuk_all_examples PRIVATE vk-bootstrap glfw glm)
set_target_properties(vuk_all_examples
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(VUK_COMPILER_CLANGPP OR VUK_COMPILER_GPP)
    target_compile_options(vuk_all_examples PRIVATE -std=c++20 -fno-char8_t)
elseif(MSVC)
    target_compile_options(vuk_all_examples PRIVATE /std:c++latest /permissive- /Zc:char8_t-)
endif()

function(ADD_EXAMPLE name)
    set(FULL_NAME "vuk_example_${name}")
    add_executable(${FULL_NAME})
    target_sources(${FULL_NAME} PRIVATE "${name}.cpp" example_runner_single.cpp imgui.cpp stbi.cpp ${imgui_sources})
    target_sources(vuk_all_examples PRIVATE "${name}.cpp")
    target_include_directories(${FULL_NAME} SYSTEM PRIVATE ../ext/stb ../ext/imgui)
    target_compile_definitions(${FULL_NAME} PRIVATE GLM_FORCE_SIZE_FUNC GLM_FORCE_EXPLICIT_CTOR GLM_ENABLE_EXPERIMENTAL GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE)
    target_link_libraries(${FULL_NAME} PRIVATE vuk)
    target_link_libraries(${FULL_NAME} PRIVATE vk-bootstrap glfw glm)
    set_target_properties(${FULL_NAME}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    if(VUK_COMPILER_CLANGPP OR VUK_COMPILER_GPP)
	    target_compile_options(${FULL_NAME} PRIVATE -std=c++20 -fno-char8_t)
    elseif(MSVC)
	    target_compile_options(${FULL_NAME} PRIVATE /std:c++latest /permissive- /Zc:char8_t-)
    endif()
endfunction(ADD_EXAMPLE)

ADD_EXAMPLE(01_triangle)
ADD_EXAMPLE(02_cube)
ADD_EXAMPLE(03_multipass)
ADD_EXAMPLE(04_texture)
ADD_EXAMPLE(05_deferred)
ADD_EXAMPLE(06_msaa)
ADD_EXAMPLE(07_commands)
ADD_EXAMPLE(08_pipelined_compute)
ADD_EXAMPLE(09_persistent_descriptorset)
ADD_EXAMPLE(10_baby_renderer)
ADD_EXAMPLE(11_composition)
