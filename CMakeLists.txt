cmake_minimum_required(VERSION 3.7)
project(vuk)

include(ExternalProject)
include(FetchContent)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_library(vuk)

option(VUK_BUILD_EXAMPLES "Build examples" OFF)
option(VUK_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(VUK_BUILD_DOCS "Build docs" OFF)
option(VUK_LINK_TO_LOADER "Link \"statically\" to the loader" ON)
option(VUK_USE_VULKAN_SDK "Use the Vulkan SDK to locate headers and libraries" ON)
option(VUK_USE_SHADERC "Link in shaderc for runtime compilation of shaders" ON)

##### Using vuk with volk (or a similar library)
# step 1: turn off VUK_LINK_TO_LOADER and add_subdirectory vuk
# set(VUK_LINK_TO_LOADER OFF)
# add_subdirectory(vuk)
# step 2: replace the default <vulkan/vulkan.h> with your include
# target_compile_definitions(vuk PUBLIC VUK_CUSTOM_VULKAN_HEADER=<volk.h>)
# step 3: link vuk to the loader lib (this policy might be needed for link_libraries)
# cmake_policy(SET CMP0079 NEW)
# target_link_libraries(vuk PUBLIC volk)
#####
if(VUK_USE_VULKAN_SDK)
	find_package(Vulkan REQUIRED)
	if(VUK_USE_SHADERC)
		add_library(shaderc UNKNOWN IMPORTED)
		if(WIN32)
			# use the version in the SDK
			set_target_properties(shaderc PROPERTIES IMPORTED_LOCATION $ENV{VULKAN_SDK}/Lib/shaderc_shared.lib)
			set_property(TARGET shaderc PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{VULKAN_SDK}/Include)
			target_link_libraries(vuk PRIVATE shaderc)
		else()
			target_link_libraries(vuk PRIVATE shaderc_combined glslang MachineIndependent OSDependent OGLCompiler GenericCodeGen SPIRV SPIRV-Tools-opt SPIRV-Tools)
		endif()
	endif()
elseif(VUK_USE_SHADERC)
	target_link_libraries(vuk PRIVATE shaderc_combined glslang MachineIndependent OSDependent OGLCompiler GenericCodeGen SPIRV SPIRV-Tools-opt SPIRV-Tools)
endif()

target_compile_definitions(vuk PUBLIC VUK_USE_SHADERC=$<BOOL:${VUK_USE_SHADERC}>)

set(SPIRV_CROSS_CLI OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_MSL OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "")

add_subdirectory(ext/SPIRV-Cross)

# we don't enable this directly, because the order of flags disables /Zc:char8_t- on MSVC
#target_compile_features(vuk PUBLIC cxx_std_20)

target_sources(vuk PRIVATE 
	src/Pipeline.cpp
	src/Program.cpp
	src/Cache.cpp
	src/RenderGraph.cpp 
	src/RenderGraphUtil.cpp
	src/ExecutableRenderGraph.cpp
	src/VMA.cpp
	src/Allocator.cpp
	src/Context.cpp
	src/CommandBuffer.cpp
	src/PerThreadContext.cpp
	src/Util.cpp
	src/Format.cpp
	src/Name.cpp
	)

target_include_directories(vuk PUBLIC ext/plf_colony)
add_subdirectory(ext/VulkanMemoryAllocator)
add_subdirectory(ext/robin-hood-hashing)
target_include_directories(vuk PUBLIC ext/concurrentqueue)
target_include_directories(vuk PUBLIC include)

string(FIND "${CMAKE_CXX_COMPILER}" "clang++" VUK_COMPILER_CLANGPP)
if(VUK_COMPILER_CLANGPP GREATER -1)
  set(VUK_COMPILER_CLANGPP ON)
else()
  set(VUK_COMPILER_CLANGPP OFF)
endif()

string(FIND "${CMAKE_CXX_COMPILER}" "g++" VUK_COMPILER_GPP)
if(VUK_COMPILER_GPP GREATER -1)
  set(VUK_COMPILER_GPP ON)
else()
  set(VUK_COMPILER_GPP OFF)
endif()

if(VUK_COMPILER_CLANGPP OR VUK_COMPILER_GPP)
	target_compile_options(vuk PRIVATE -std=c++20 -fno-char8_t)
elseif(MSVC)
	target_compile_options(vuk PRIVATE /std:c++latest /permissive- /Zc:char8_t-)
endif()

target_link_libraries(vuk PUBLIC spirv-cross-core robin_hood VulkanMemoryAllocator)

if(VUK_LINK_TO_LOADER)
	if (VUK_USE_VULKAN_SDK)
		target_include_directories(vuk PUBLIC ${Vulkan_INCLUDE_DIRS})
		target_link_libraries(vuk PRIVATE ${Vulkan_LIBRARIES})
	else()
		target_link_libraries(vuk PRIVATE vulkan)
	endif()
endif()

if (WIN32)
	target_compile_definitions(vuk PUBLIC NOMINMAX VC_EXTRALEAN WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS _SILENCE_CLANG_CONCEPTS_MESSAGE)
endif()

if(VUK_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

if(VUK_BUILD_BENCHMARKS)
	add_subdirectory(benchmarks)
endif()

if(VUK_BUILD_DOCS)
	add_subdirectory(docs)
endif()
