cmake_minimum_required(VERSION 3.7)
project(vuk)

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

cmake_policy(SET CMP0077 NEW)

IF (NOT CMAKE_BUILD_TYPE)
 SET(CMAKE_BUILD_TYPE "Debug")
ENDIF (NOT CMAKE_BUILD_TYPE)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
	message(STATUS "Using module to find Vulkan")
	find_package(Vulkan REQUIRED)
endif()

add_subdirectory(ext/vk-bootstrap)
add_subdirectory(ext/glfw)
add_subdirectory(ext/GSL)
add_subdirectory(ext/shaderc)
add_subdirectory(ext/SPIRV-Cross)
add_subdirectory(ext/smolog)
#add_subdirectory(ext/FiberTaskingLib)
add_subdirectory(ext/glm)
add_library(vuk)
target_compile_features(vuk PUBLIC cxx_std_20)

target_sources(vuk PRIVATE 
	src/Pipeline.hpp
	src/Pipeline.cpp
	src/Program.hpp
	src/Program.cpp
	src/Pool.cpp
	src/Pool.hpp
	src/Context.hpp
	src/Cache.hpp
	src/Cache.cpp
	src/Hash.hpp
	src/RenderGraph.hpp
	src/RenderGraph.cpp 
	src/VMA.cpp
	src/Allocator.hpp
	src/Allocator.cpp
	src/CommandBuffer.hpp
	src/Handle.hpp
	src/Context.cpp src/CommandBuffer.cpp)

target_include_directories(vuk PUBLIC include ext/mapbox-variant/include)
target_include_directories(vuk PUBLIC ext/shaderc/libshaderc/include)
target_include_directories(vuk PUBLIC ${Vulkan_INCLUDE_DIRS})
target_include_directories(vuk PUBLIC ext/plf_colony)
target_include_directories(vuk PUBLIC ext/VulkanMemoryAllocator/src)
target_include_directories(vuk PUBLIC ext/imgui src/)
target_sources(vuk PRIVATE ext/imgui/imgui.cpp ext/imgui/imgui_draw.cpp ext/imgui/imgui_demo.cpp ext/imgui/imgui_widgets.cpp ext/imgui/examples/imgui_impl_glfw.cpp)

target_compile_options(vuk PRIVATE /W3 /MP /bigobj /permissive- /GT)
target_compile_options(vuk PUBLIC /Zc:char8_t-)

target_link_libraries(vuk PUBLIC ${Vulkan_LIBRARY} vk-bootstrap glfw GSL spirv-cross-core shaderc smolog glm)

if (WIN32)
	target_compile_definitions(vuk PUBLIC NOMINMAX VC_EXTRALEAN WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
endif()
target_compile_definitions(vuk PUBLIC GLM_FORCE_SIZE_FUNC GLM_FORCE_EXPLICIT_CTOR GLM_ENABLE_EXPERIMENTAL GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE)
add_subdirectory(examples)