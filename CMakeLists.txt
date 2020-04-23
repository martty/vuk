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

if(WIN32)
	set(VUK_SHADERC_PATH_D $ENV{VULKAN_SDK}/shaderc)
else()
	set(VUK_SHADERC_PATH_D $ENV{VULKAN_SDK}/source/shaderc)
endif()
set(VUK_SHADERC_PATH ${VUK_SHADERC_PATH_D} CACHE PATH "")
file(TO_CMAKE_PATH "${VUK_SHADERC_PATH}" VUK_SHADERC_PATH_N)

FetchContent_Declare(
  sshaderc
  SOURCE_DIR "${VUK_SHADERC_PATH_N}"
)

FetchContent_GetProperties(sshaderc)
if(NOT sshaderc_POPULATED)
  message(STATUS "Building shaderc in RelWithDebInfo")
  FetchContent_Populate(sshaderc)
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" -S ${sshaderc_SOURCE_DIR} -B ${sshaderc_BINARY_DIR} -DSHADERC_SKIP_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSHADERC_ENABLE_SHARED_CRT=ON -DSPIRV_SKIP_EXECUTABLES=ON -DENABLE_GLSLANG_BINARIES=OFF -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
  execute_process(COMMAND "${CMAKE_COMMAND}" --build ${sshaderc_BINARY_DIR} --config RelWithDebInfo)
  if(EXISTS ${sshaderc_BINARY_DIR}/libshaderc/shaderc_shared.dll)
	file(COPY ${sshaderc_BINARY_DIR}/libshaderc/shaderc_shared.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  else()
	add_custom_command(TARGET vuk POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${sshaderc_BINARY_DIR}/libshaderc/RelWithDebInfo/shaderc_shared.dll ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR})
  endif()
endif()

add_library(shaderc UNKNOWN IMPORTED)
if(EXISTS ${sshaderc_BINARY_DIR}/libshaderc/shaderc_shared.dll)
set_target_properties(shaderc PROPERTIES IMPORTED_LOCATION ${sshaderc_BINARY_DIR}/libshaderc/shaderc_shared.lib)
else()
set_target_properties(shaderc PROPERTIES IMPORTED_LOCATION ${sshaderc_BINARY_DIR}/libshaderc/RelWithDebInfo/shaderc_shared.lib)
endif()
set_property(TARGET shaderc PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${sshaderc_SOURCE_DIR}/libshaderc/include)

set(GSL_CXX_STANDARD 20)
add_subdirectory(ext/GSL)
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
target_include_directories(vuk PUBLIC src/)

target_compile_options(vuk PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/permissive->")
target_link_libraries_system(vuk PUBLIC Vulkan::Vulkan GSL spirv-cross-core)
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