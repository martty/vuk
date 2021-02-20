cmake_minimum_required(VERSION 3.7)
project(vuk-benchmarks)

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
  GIT_TAG        3.3.2
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm
  GIT_TAG        0.9.9.8
)
FetchContent_MakeAvailable(glm)

SET(imgui_sources ../ext/imgui/imgui.cpp ../ext/imgui/imgui_draw.cpp ../ext/imgui/imgui_demo.cpp ../ext/imgui/imgui_widgets.cpp ../ext/imgui/examples/imgui_impl_glfw.cpp)

function(ADD_BENCH name)
    set(FULL_NAME "vuk_bench_${name}")
    add_executable(${FULL_NAME})
    target_sources(${FULL_NAME} PRIVATE "${name}.cpp" bench_runner.cpp ../examples/imgui.cpp ../examples/stbi.cpp ${imgui_sources})
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
endfunction(ADD_BENCH)

ADD_BENCH(dependent_texture_fetches)
