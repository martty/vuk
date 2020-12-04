cmake_minimum_required(VERSION 3.7)
project(vuk)

include(ExternalProject)
include(FetchContent)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_library(vuk)

find_package(Vulkan REQUIRED)

add_library(shaderc UNKNOWN IMPORTED)
if(WIN32)
	# use the version in the SDK	
	set_target_properties(shaderc PROPERTIES IMPORTED_LOCATION $ENV{VULKAN_SDK}/Lib/shaderc_shared.lib)
	set_property(TARGET shaderc PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{VULKAN_SDK}/Include)
else()
	# TODO
endif()

set(GSL_CXX_STANDARD 20)
set(SPIRV_CROSS_CLI OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_MSL OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "")

add_subdirectory(ext/SPIRV-Cross)

#target_compile_features(vuk PUBLIC cxx_std_20)

target_sources(vuk PRIVATE 
	src/Pipeline.cpp
	src/Program.cpp
	src/Pool.cpp
	src/Cache.cpp
	src/RenderGraph.cpp 
	src/RenderGraphUtil.cpp
	src/ExecutableRenderGraph.cpp
	src/VMA.cpp
	src/Allocator.cpp
	src/Context.cpp
	src/CommandBuffer.cpp
	src/InflightContext.cpp
	src/PerThreadContext.cpp
	src/Util.cpp
	src/Format.cpp)

target_include_directories(vuk PUBLIC ext/plf_colony)
target_include_directories(vuk PUBLIC ext/VulkanMemoryAllocator/src)
add_subdirectory(ext/robin-hood-hashing)
target_include_directories(vuk PUBLIC ext/concurrentqueue)
target_include_directories(vuk PUBLIC include)
target_include_directories(vuk PUBLIC src/)

string(FIND "${CMAKE_CXX_COMPILER}" "clang++" VUK_COMPILER_CLANGPP)
if(VUK_COMPILER_CLANGPP GREATER -1)
  set(VUK_COMPILER_CLANGPP ON)
else()
  set(VUK_COMPILER_CLANGPP OFF)
endif()

if(VUK_COMPILER_CLANGPP)
	target_compile_options(vuk PRIVATE -std=c++20 -fno-char8_t)
elseif(MSVC)
	target_compile_options(vuk PRIVATE /std:c++latest /permissive- /Zc:char8_t-)
endif()

target_link_libraries(vuk PUBLIC spirv-cross-core robin_hood)
target_link_libraries(vuk PUBLIC shaderc ${Vulkan_LIBRARIES})


if (WIN32)
	target_compile_definitions(vuk PUBLIC NOMINMAX VC_EXTRALEAN WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS _SILENCE_CLANG_CONCEPTS_MESSAGE)
endif()
target_compile_definitions(vuk PUBLIC VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=0)

option(VUK_BUILD_EXAMPLES "Build examples" OFF)

if(VUK_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

option(VUK_BUILD_DOCS "Build docs" OFF)
if(VUK_BUILD_DOCS)
	add_subdirectory(docs)
endif()
