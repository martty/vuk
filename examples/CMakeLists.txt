cmake_minimum_required(VERSION 3.15)
project(vuk-examples)

if(NOT VUK_USE_SHADERC)
    message(FATAL_ERROR "Building vuk examples require shaderc for building shaders, enable VUK_USE_SHADERC")
endif()
if(NOT VUK_EXTRA_IMGUI)
    message(FATAL_ERROR "Building vuk examples requires the imgui integration, enable VUK_EXTRA and VUK_EXTRA_IMGUI")
endif()
if(NOT VUK_EXTRA_INIT)
    message(FATAL_ERROR "Building vuk examples requires the simple init, enable VUK_EXTRA and VUK_EXTRA_INIT")
endif()

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw
  GIT_TAG        3.3.6
)

if (NOT APPLE)
    option(TRACY_ENABLE "Enable Tracy Profiler" ON)
else()
    option(TRACY_ENABLE "Enable Tracy Profiler" OFF)
endif()
set(TRACY_ON_DEMAND OFF)
FetchContent_Declare(tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG 2183962ef7fc8f721019c01d9170f5ea0d509555
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm
  GIT_TAG        cc98465e3508535ba8c7f6208df934c156a018dc
)
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb
  GIT_TAG 5c205738c191bcb0abc65c4febfa9bd25ff35234
)

FetchContent_MakeAvailable(
  glfw
  glm
  stb
  tracy
)

if(TRACY_ENABLE AND APPLE)
    message(FATAL_ERROR "MoltenVk's restrictions on query pools are too high to be usable")
endif()

# stb does not provide a CMake build script, hence we create a interface
# library here.
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE "${stb_SOURCE_DIR}")

add_library(vuk-example-framework STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/stbi.cpp"
  example_main.cpp
)
target_compile_definitions(vuk-example-framework PUBLIC GLM_FORCE_SIZE_FUNC GLM_FORCE_EXPLICIT_CTOR GLM_ENABLE_EXPERIMENTAL GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE TRACY_VK_USE_SYMBOL_TABLE)
if(VUK_COMPILER_CLANGPP OR VUK_COMPILER_GPP)
  target_compile_options(vuk-example-framework PUBLIC -std=c++20 -fno-char8_t)
elseif(MSVC)
  target_compile_options(vuk-example-framework PUBLIC /std:c++20 /permissive- /Zc:char8_t- /wd4068)
endif()
target_link_libraries(vuk-example-framework PUBLIC
  vuk
  glfw
  glm
  stb
  TracyClient
)
target_link_libraries(vuk-example-framework PRIVATE vuk-extra-imgui-platform)

add_executable(vuk_all_examples)
target_sources(vuk_all_examples PRIVATE example_browser.cpp)
target_link_libraries(vuk_all_examples PRIVATE vuk-example-framework)
target_compile_definitions(vuk_all_examples PUBLIC VUK_EX_PATH_TGT="$<TARGET_FILE_DIR:vuk_all_examples>" VUK_EX_PATH_ROOT="${CMAKE_SOURCE_DIR}")

add_library(vuk_example_runner_single OBJECT example_runner_single.cpp)
target_link_libraries(vuk_example_runner_single PRIVATE vuk-example-framework)
target_compile_definitions(vuk-example-framework PUBLIC VUK_EX_PATH_TGT="$<TARGET_FILE_DIR:vuk_all_examples>" VUK_EX_PATH_ROOT="${CMAKE_SOURCE_DIR}" TRACY_NO_CALLSTACK)

function(ADD_EXAMPLE name)
    set(FULL_NAME "vuk_example_${name}")
    add_executable(${FULL_NAME})

    add_library(${FULL_NAME}_obj OBJECT "${name}.cpp")
    target_link_libraries(${FULL_NAME}_obj PUBLIC vuk-example-framework)
    target_link_libraries(${FULL_NAME} PRIVATE ${FULL_NAME}_obj vuk_example_runner_single)
    target_link_libraries(vuk_all_examples PRIVATE ${FULL_NAME}_obj)

    copy_runtime_dlls(${FULL_NAME})
endfunction(ADD_EXAMPLE)

ADD_EXAMPLE(01_triangle)
if(VUK_USE_DXC)
    ADD_EXAMPLE(01_triangle_dxc)
endif()
if(VUK_USE_VCC)
    ADD_EXAMPLE(01_triangle_vcc)
endif()
if(VUK_USE_SLANG)
    ADD_EXAMPLE(01_triangle_slang)
endif()
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
ADD_EXAMPLE(12_rt_pipeline)

# standalone examples
add_executable(standalone_triangle standalone_triangle.cpp)
target_link_libraries(standalone_triangle PUBLIC vuk glfw glm)
target_compile_definitions(standalone_triangle PUBLIC VUK_EX_PATH_TGT="$<TARGET_FILE_DIR:standalone_triangle>" VUK_EX_PATH_ROOT="${CMAKE_SOURCE_DIR}")

#add_executable(standalone_multiwindow standalone_multiwindow.cpp)
#target_link_libraries(standalone_multiwindow PUBLIC vuk-example-framework)
