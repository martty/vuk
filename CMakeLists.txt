cmake_minimum_required(VERSION 3.7)
project(vuk)

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Vulkan REQUIRED)
#use shaderc shipped with SDK, has to be a source dependency because there are no debug binaries shipped
if(WIN32)
	add_subdirectory($ENV{VULKAN_SDK}/shaderc shaderc)
else()
	add_subdirectory($ENV{VULKAN_SDK}/source/shaderc shaderc)
endif()
add_subdirectory(ext/GSL)
add_subdirectory(ext/SPIRV-Cross)

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
	src/Context.cpp
	src/CommandBuffer.cpp)

target_include_directories(vuk PUBLIC ext/plf_colony)
target_include_directories(vuk PUBLIC ext/VulkanMemoryAllocator/src)
target_include_directories(vuk PUBLIC src/)

target_compile_options(vuk PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/permissive->")
target_link_libraries(vuk PUBLIC Vulkan::Vulkan GSL spirv-cross-core shaderc)

if (WIN32)
	target_compile_definitions(vuk PUBLIC NOMINMAX VC_EXTRALEAN WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
endif()
add_subdirectory(examples)