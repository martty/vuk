cmake_minimum_required(VERSION 3.7)
project(vuk)

include(ExternalProject)
include(FetchContent)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# https://stackoverflow.com/a/52136398
function(target_link_libraries_system target access)
  set(libs ${ARGN})
  foreach(lib ${libs})
    get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${target} SYSTEM ${access} ${lib_include_dirs})
    target_link_libraries(${target} ${access} ${lib})
  endforeach(lib)
endfunction(target_link_libraries_system)

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
	src/Types.hpp
	src/Context.cpp
	src/CommandBuffer.cpp)

target_include_directories(vuk PUBLIC ext/plf_colony)
target_include_directories(vuk PUBLIC ext/VulkanMemoryAllocator/src)
target_include_directories(vuk PUBLIC ext/flat_hash_map)
target_include_directories(vuk PUBLIC ext/concurrentqueue)
target_include_directories(vuk PUBLIC src/)

target_precompile_headers(vuk PRIVATE <vulkan/vulkan.hpp>)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND NOT MSVC)
	target_compile_options(vuk PRIVATE -fno-char8_t)
elseif(MSVC)
	target_compile_options(vuk PRIVATE /permissive- /Zc:char8_t-)
endif()

target_link_libraries_system(vuk PUBLIC Vulkan::Vulkan spirv-cross-core)
target_link_libraries(vuk PUBLIC shaderc)


if (WIN32)
	target_compile_definitions(vuk PUBLIC NOMINMAX VC_EXTRALEAN WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS _SILENCE_CLANG_CONCEPTS_MESSAGE)
endif()

option(VUK_BUILD_VUSH "Build vush" ON)
option(VUK_BUILD_EXAMPLES "Build examples" OFF)
if(VUK_BUILD_VUSH)
	add_subdirectory(vush)
endif()

if(VUK_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()