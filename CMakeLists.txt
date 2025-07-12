cmake_minimum_required(VERSION 3.15)
project(vuk LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(ExternalProject)
include(FetchContent)
include(CMakeDependentOption)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(SpirvEmbed REQUIRED) # for embedding binary data

cmake_policy(SET CMP0174 NEW)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_library(vuk)

option(VUK_BUILD_EXAMPLES "Build examples" OFF)
option(VUK_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(VUK_BUILD_DOCS "Build docs" OFF)
option(VUK_LINK_TO_LOADER "Link \"statically\" to the loader" ON)
option(VUK_USE_VULKAN_SDK "Use the Vulkan SDK to locate headers and libraries" ON)
option(VUK_USE_SHADERC "Link in shaderc for runtime compilation of GLSL shaders" ON)
option(VUK_USE_DXC "Link in DirectXShaderCompiler for runtime compilation of HLSL shaders" OFF)
option(VUK_USE_VCC "Use Vcc for runtime compilation of C shaders (experimental)" OFF)
option(VUK_USE_SLANG "Use Slang for runtime compilation of Slang shaders" OFF)
option(VUK_BUILD_TESTS "Build tests" OFF)
option(VUK_FAIL_FAST "Trigger an assert upon encountering an error instead of propagating" OFF)
option(VUK_DEBUG_ALLOCATIONS "Dump VMA allocations and give them debug names" OFF)
# Non-core features
option(VUK_EXTRA "Enable extra features" ON)
cmake_dependent_option(VUK_EXTRA_IMGUI "Build imgui integration" ON "VUK_EXTRA" OFF)
set(VUK_EXTRA_IMGUI_PLATFORM_BACKEND "glfw" CACHE STRING "Platform backend for the imgui integration")
set_property(CACHE VUK_EXTRA_IMGUI_PLATFORM_BACKEND PROPERTY STRINGS glfw sdl2 sdl3 custom)
cmake_dependent_option(VUK_EXTRA_INIT "Build init helper" ON "VUK_EXTRA" OFF)

if(CMAKE_SIZEOF_VOID_P EQUAL "4")
	message(FATAL_ERROR "x86 is not supported.")
endif()

# Detect OS
set(VUK_OS_LINUX OFF)
set(VUK_OS_APPLE OFF)
set(VUK_OS_WINDOWS OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(VUK_OS_LINUX ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(VUK_OS_APPLE ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(VUK_OS_WINDOWS ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(VUK_OS_ANDROID ON)
else()
  message(FATAL_ERROR "unrecognised operating system '${CMAKE_SYSTEM_NAME}'")
endif()

message(STATUS "LINUX=${VUK_OS_LINUX}; APPLE=${VUK_OS_APPLE}; WINDOWS=${VUK_OS_WINDOWS}; ANDROID=${VUK_OS_ANDROID}")

# Detect compilers
set(VUK_COMPILER_CLANGPP OFF)
set(VUK_COMPILER_CLANGCL OFF)
set(VUK_COMPILER_GPP OFF)
set(VUK_COMPILER_MSVC OFF)
set(VUK_COMPILER_ANDROID OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    set(VUK_COMPILER_CLANGPP ON)
  elseif(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(VUK_COMPILER_CLANGCL ON)
  else()
    message(FATAL_ERROR "unrecognised clang frontend '${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}'")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(VUK_COMPILER_CLANGPP ON)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(VUK_COMPILER_GPP ON)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(VUK_COMPILER_MSVC ON)
else()
  message(FATAL_ERROR "unrecognised compiler '${CMAKE_CXX_COMPILER}' with ID '${CMAKE_CXX_COMPILER_ID}'")
endif()

message(STATUS "CLANGPP=${VUK_COMPILER_CLANGPP}; CLANGCL=${VUK_COMPILER_CLANGCL}; GPP=${VUK_COMPILER_GPP}; MSVC=${VUK_COMPILER_MSVC}")

set(BUILD_TESTS OFF)

FetchContent_Declare(
  robin-hood-hashing
  GIT_REPOSITORY https://github.com/martinus/robin-hood-hashing.git
  GIT_TAG bd632059ea0bb5cf444b33bdfb4eba2118300c62
)
FetchContent_Declare(
  plf_colony
  GIT_REPOSITORY https://github.com/mattreecebentley/plf_colony.git
  GIT_TAG c4c1a2e7f4346faa5d16e89ba1b4fa8c3d976bd2
)
FetchContent_Declare(
  small_vector
  GIT_REPOSITORY https://github.com/gharveymn/small_vector.git
  GIT_TAG 5b4ad3bd3dc3e1593a7e95cb3843a87b5ae21000
)
FetchContent_Declare(
  function2
  GIT_REPOSITORY https://github.com/Naios/function2.git
  GIT_TAG 43fc0ca473ecb081918709bd7d524d84c2ff8dce
)
FetchContent_Declare(
  concurrentqueue
  GIT_REPOSITORY https://github.com/cameron314/concurrentqueue
  GIT_TAG 0d54c6794510018f42b1b987f55c313dd1358320
)
FetchContent_Declare(
  vma
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
  GIT_TAG v3.2.1
)
set(SPIRV_CROSS_CLI OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_MSL OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "")
set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "")
FetchContent_Declare(
  spirv-cross
  GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross.git
  GIT_TAG 2c32b6bf86f3c4a5539aa1f0bacbd59fe61759cf
)

FetchContent_Declare(
  vuk-binary-dist
  GIT_REPOSITORY https://github.com/martty/vuk-binary-dist.git
  GIT_TAG 517d0bf43963725208794a43530341e65524edf1
  CONFIGURE_COMMAND " "
)

# for vuk-binary-dist, we want to populate but not add to the build
FetchContent_GetProperties(vuk-binary-dist)
if(NOT vuk-binary-dist_POPULATED)
  FetchContent_Populate(vuk-binary-dist)
endif()

FetchContent_MakeAvailable(
  robin-hood-hashing
  plf_colony
  small_vector
  concurrentqueue
  function2
  vma
  spirv-cross
)

if (NOT TARGET fmt::fmt)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 12.1.0
  )
  FetchContent_MakeAvailable(fmt)
endif()

# plf_colony does not provide a CMake build script, hence we create an
# interface library here.
add_library(plf_colony INTERFACE)
target_include_directories(plf_colony INTERFACE "${plf_colony_SOURCE_DIR}")

# VulkanMemoryAllocator (vma) does not provide a CMake build script, hence we
# create an interface library here.
add_library(vma INTERFACE)
target_include_directories(vma INTERFACE "${vma_SOURCE_DIR}/include")

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
			if($ENV{VULKAN_SDK} STREQUAL "")
				message(FATAL_ERROR "Env variable VULKAN_SDK not set, but VUK_USE_VULKAN_SDK is set")
			endif()
			# use the version in the SDK
			set_target_properties(shaderc PROPERTIES IMPORTED_LOCATION $ENV{VULKAN_SDK}/Lib/shaderc_shared.lib)
			set_property(TARGET shaderc PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{VULKAN_SDK}/Include)
			target_link_libraries(vuk PRIVATE shaderc)
		else()
			target_link_directories(vuk PUBLIC $ENV{VULKAN_SDK}/lib)
			find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)
			target_link_libraries(vuk PRIVATE Vulkan::shaderc_combined)
		endif()
	endif()
	if (VUK_USE_DXC)
		add_library(dxc UNKNOWN IMPORTED)
		if (WIN32)
			if($ENV{VULKAN_SDK} STREQUAL "")
				message(FATAL_ERROR "Env variable VULKAN_SDK not set, but VUK_USE_VULKAN_SDK is set")
			endif()
			set_target_properties(dxc PROPERTIES IMPORTED_LOCATION $ENV{VULKAN_SDK}/Lib/dxcompiler.lib)
			set_property(TARGET dxc PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{VULKAN_SDK}/Include)
			target_link_libraries(vuk PRIVATE dxc)
		else()
			target_link_libraries(vuk PRIVATE dxcompiler)
		endif()
	endif()
else()
	if (VUK_USE_SHADERC)
		find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)
		target_link_libraries(vuk PRIVATE Vulkan::shaderc_combined)
	endif()
	if (VUK_USE_DXC)
		target_link_libraries(vuk PRIVATE dxcompiler)
	endif()
endif()

if (VUK_USE_SHADERC)
	target_sources(vuk PRIVATE src/shader_compilers/shaderc.cpp)
endif()
if (VUK_USE_DXC)
	target_sources(vuk PRIVATE src/shader_compilers/dxc.cpp)
endif()
if (VUK_USE_VCC)
	FetchContent_Declare(
	  shady-fc
	  GIT_REPOSITORY https://github.com/martty/shady
	  GIT_TAG ci-win
	)
	FetchContent_GetProperties(shady-fc)

	if(NOT shady-fc_POPULATED)
		FetchContent_Populate(shady-fc)
	endif()

	ExternalProject_Add(shady2
		SOURCE_DIR
			${shady-fc_SOURCE_DIR}
		PREFIX ""
		BINARY_DIR shady-build
		CMAKE_ARGS -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=1 -DCMAKE_SHARED_LIBRARY_PREFIX="" -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
		INSTALL_COMMAND ""
		BUILD_ALWAYS OFF
		BUILD_COMMAND
			${CMAKE_COMMAND} --build .
		LOG_DOWNLOAD 1
		LOG_BUILD 1
		LOG_OUTPUT_ON_FAILURE 1
		BUILD_BYPRODUCTS <BINARY_DIR>/lib/driver.lib
    )

	ExternalProject_Get_property(shady2 BINARY_DIR)
	ExternalProject_Get_property(shady2 SOURCE_DIR)
	add_library(shady-rel SHARED IMPORTED GLOBAL)
	add_dependencies(shady-rel shady2)
	file(MAKE_DIRECTORY ${BINARY_DIR}/src/shady/api)
	target_include_directories(shady-rel INTERFACE ${SOURCE_DIR}/include ${BINARY_DIR}/src/shady/api)
	set_target_properties(shady-rel PROPERTIES IMPORTED_IMPLIB ${BINARY_DIR}/lib/driver.lib IMPORTED_LOCATION ${BINARY_DIR}/bin/driver.dll)

	target_sources(vuk PRIVATE src/shader_compilers/vcc.cpp)
	target_link_libraries(vuk PUBLIC shady-rel)
	target_compile_definitions(vuk PRIVATE "-DVUK_VCC_INCLUDE_DIR=\"${BINARY_DIR}/share/vcc/include/\"")
endif()

if (VUK_USE_SLANG)
	FetchContent_Declare(
	  slang-fc
	  GIT_REPOSITORY https://github.com/shader-slang/slang
	  GIT_TAG 708e1bbe1197d198831ffbe8601ad0d9bdb821f9
	  GIT_SUBMODULES external/miniz external/lz4 external/unordered_dense external/spirv-headers external/spirv-tools external/vulkan external/glslang external/lua
	  GIT_SUBMODULES_RECURSE ON
	)
	FetchContent_GetProperties(slang-fc)

	if(NOT slang-fc_POPULATED)
		FetchContent_Populate(slang-fc)
	endif()

	ExternalProject_Add(slang2
		SOURCE_DIR
			${slang-fc_SOURCE_DIR}
		PREFIX ""
		BINARY_DIR slang-build
		CMAKE_ARGS
			-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_SHARED_LIBRARY_PREFIX="" -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
			-DSLANG_ENABLE_GFX=0 -DSLANG_ENABLE_SLANGC=0	-DSLANG_ENABLE_TESTS=0	-DSLANG_ENABLE_EXAMPLES=0 -DSLANG_ENABLE_SLANGRT=0 -DSLANG_ENABLE_SLANGD=0
			-DSLANG_ENABLE_SLANGRT=0 -DSLANG_ENABLE_SLANG_RHI=0 -DSLANG_ENABLE_DXIL=0 -DSLANG_ENABLE_PREBUILT_BINARIES=0 -DSLANG_EMBED_CORE_MODULE_SOURCE=1
			-DSLANG_EMBED_CORE_MODULE=0
		INSTALL_COMMAND ""
		BUILD_ALWAYS OFF
		BUILD_COMMAND
			${CMAKE_COMMAND} --build .
		LOG_DOWNLOAD 1
		LOG_BUILD 1
		LOG_OUTPUT_ON_FAILURE 1
		BUILD_BYPRODUCTS <BINARY_DIR>/RelWithDebInfo/lib/slang.lib <BINARY_DIR>/RelWithDebInfo/lib/slang-glslang.lib
    )
	ExternalProject_Get_property(slang2 BINARY_DIR)
	ExternalProject_Get_property(slang2 SOURCE_DIR)
	add_library(slang-rel SHARED IMPORTED GLOBAL)
	add_dependencies(slang-rel slang2)
	target_include_directories(slang-rel INTERFACE ${SOURCE_DIR} ${SOURCE_DIR}/include)
	set_target_properties(slang-rel PROPERTIES IMPORTED_IMPLIB ${BINARY_DIR}/RelWithDebInfo/lib/slang.lib IMPORTED_LOCATION ${BINARY_DIR}/RelWithDebInfo/bin/slang.dll)
	add_library(slang-glslang SHARED IMPORTED GLOBAL)
	add_dependencies(slang-glslang slang2)
	set_target_properties(slang-glslang PROPERTIES IMPORTED_IMPLIB ${BINARY_DIR}/RelWithDebInfo/lib/slang-glslang.lib IMPORTED_LOCATION ${BINARY_DIR}/RelWithDebInfo/bin/slang-glslang.dll)
	target_link_libraries(vuk PRIVATE slang-rel slang-glslang)
	target_sources(vuk PRIVATE src/shader_compilers/slang.cpp)
endif()

target_compile_definitions(vuk
  PUBLIC
  VUK_USE_SHADERC=$<BOOL:${VUK_USE_SHADERC}>
  VUK_USE_DXC=$<BOOL:${VUK_USE_DXC}>
  VUK_USE_VCC=$<BOOL:${VUK_USE_VCC}>
  VUK_USE_SLANG=$<BOOL:${VUK_USE_SLANG}>
  VUK_BUILD_TESTS=$<BOOL:${VUK_BUILD_TESTS}>
  VUK_FAIL_FAST=$<BOOL:${VUK_FAIL_FAST}>
  VUK_DEBUG_ALLOCATIONS=$<BOOL:${VUK_DEBUG_ALLOCATIONS}>
  VUK_COMPILER_CLANGPP=$<BOOL:${VUK_COMPILER_CLANGPP}>
  VUK_COMPILER_CLANGCL=$<BOOL:${VUK_COMPILER_CLANGCL}>
  VUK_COMPILER_GPP=$<BOOL:${VUK_COMPILER_GPP}>
  VUK_COMPILER_MSVC=$<BOOL:${VUK_COMPILER_MSVC}>
  VUK_OS_WINDOWS=$<BOOL:${VUK_OS_WINDOWS}>
  VUK_OS_LINUX=$<BOOL:${$VUK_OS_LINUX}>
  VUK_OS_APPLE=$<BOOL:${VUK_OS_APPLE}>
  VUK_OS_ANDROID=$<BOOL:${VUK_OS_ANDROID}>
)

target_compile_features(vuk PUBLIC cxx_std_20)

target_sources(vuk PRIVATE
	src/Format.cpp
	src/Name.cpp
  src/IR.cpp
	src/IRPasses.cpp
	src/GraphDumper.cpp

	src/runtime/Cache.cpp
	src/runtime/RadixTree.cpp

	src/runtime/vk/Allocator.cpp
	src/runtime/vk/BufferAllocator.cpp	
	src/runtime/vk/Backend.cpp
	src/runtime/vk/CommandBuffer.cpp
	src/runtime/vk/Descriptor.cpp
	src/runtime/vk/DeviceFrameResource.cpp
	src/runtime/vk/DeviceLinearResource.cpp
	src/runtime/vk/DeviceVkResource.cpp
	src/runtime/vk/Pipeline.cpp
	src/runtime/vk/Program.cpp
	src/runtime/vk/Util.cpp
	src/runtime/vk/VkQueueExecutor.cpp
	src/runtime/vk/VkRuntime.cpp
)

# if extra features are enabled, add them
if(VUK_EXTRA_IMGUI)
	add_subdirectory(src/extra/imgui)
endif()

if(VUK_EXTRA_INIT)
	add_subdirectory(src/extra/init)
endif()

if(VUK_EXTRA)
	target_dist_spv(vuk SHADERS spd.cs.hlsl.spv)
endif()

target_include_directories(vuk PUBLIC include)

if(VUK_COMPILER_CLANGPP OR VUK_COMPILER_GPP)
  target_compile_options(vuk PRIVATE -fno-char8_t -Wreturn-type)
elseif(VUK_COMPILER_MSVC OR VUK_COMPILER_CLANGCL) # cl.exe or clang-cl
  # w4068 : unknown pragma
  target_compile_options(vuk PRIVATE /permissive- /Zc:char8_t- /wd4068)
endif()

if(VUK_COMPILER_CLANGPP OR VUK_COMPILER_CLANGCL)
  target_compile_options(vuk PRIVATE -Wno-nullability-completeness)
endif()

target_link_libraries(vuk
  PUBLIC
  function2
  plf_colony
  PRIVATE
  spirv-cross-core
  vma
  robin_hood
  fmt::fmt
  gch::small_vector
  concurrentqueue
)

if(VUK_LINK_TO_LOADER)
	if (VUK_USE_VULKAN_SDK)
		target_include_directories(vuk PUBLIC ${Vulkan_INCLUDE_DIRS})
		target_link_libraries(vuk PRIVATE ${Vulkan_LIBRARIES})
	else()
		target_link_libraries(vuk PRIVATE vulkan)
	endif()
else()
	if (VUK_USE_VULKAN_SDK)
		target_include_directories(vuk PUBLIC ${Vulkan_INCLUDE_DIRS})
	endif()
endif()

function(copy_runtime_dlls TARGET)
  get_property(already_applied TARGET "${TARGET}" PROPERTY _copy_runtime_dlls_applied)

  if (CMAKE_IMPORT_LIBRARY_SUFFIX AND NOT already_applied)
    add_custom_command(
      TARGET "${TARGET}" POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy
        -t  "$<TARGET_FILE_DIR:${TARGET}>" "$<TARGET_RUNTIME_DLLS:${TARGET}>"
      COMMAND_EXPAND_LISTS
    )

    set_property(TARGET "${TARGET}" PROPERTY _copy_runtime_dlls_applied 1)
  endif ()
endfunction()

if (WIN32)
	target_compile_definitions(vuk PUBLIC NOMINMAX VC_EXTRALEAN WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS _SILENCE_CLANG_CONCEPTS_MESSAGE _SILENCE_CXX23_ALIGNED_STORAGE_DEPRECATION_WARNING)
endif()

if(VUK_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

#if(VUK_BUILD_BENCHMARKS)
#	add_subdirectory(benchmarks)
#endif()

if(VUK_BUILD_DOCS)
	add_subdirectory(docs)
endif()


if(VUK_BUILD_TESTS)
	enable_testing()

	FetchContent_Declare(
	  doctest
	  GIT_REPOSITORY https://github.com/kociap/doctest
	  GIT_TAG        88b9982182fd670966b37d9271323b0169db6143
	)
	FetchContent_Declare(
	  vk-bootstrap
	  GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
	  GIT_TAG        cf8df11a0a071463009031cb474099dacffe90ed
	)

	FetchContent_MakeAvailable(vk-bootstrap doctest)

	target_link_libraries(vuk PUBLIC doctest::doctest)
	target_sources(vuk PRIVATE tests/Test.cpp)
	include(doctest_force_link_static_lib_in_target) # until we can use cmake 3.24
	add_executable(vuk-tests tests/Test.cpp
					#tests/00_misc.cpp
					#tests/01_semantics.cpp
					#tests/02_rg_errors.cpp
					#tests/03_commands.cpp
					#tests/04_arrays.cpp
					#tests/05_renderpass.cpp
					#tests/06_mt.cpp
					#tests/X1_SPD.cpp
					#tests/radix_tree.cpp
					#tests/frame_allocator.cpp
					tests/buffer_ptr_view.cpp
	)
	#target_compile_features(vuk-tests PRIVATE cxx_std_17)
	target_include_directories(vuk-tests PRIVATE ext/renderdoc)
	target_link_libraries(vuk-tests PRIVATE vuk doctest::doctest vk-bootstrap fmt::fmt)
	target_compile_definitions(vuk-tests PRIVATE VUK_TEST_RUNNER)

	copy_runtime_dlls(vuk-tests)

  include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake) # until we can use cmake 3.24
	doctest_discover_tests(vuk-tests)

	if(VUK_COMPILER_CLANGPP OR VUK_COMPILER_GPP)
			target_compile_options(vuk-tests PRIVATE -fno-char8_t)
	elseif(MSVC)
			# w4068 : unknown pragma
			target_compile_options(vuk-tests PRIVATE /std:c++latest /permissive- /Zc:char8_t- /wd4068)
	endif()

	if(VUK_COMPILER_CLANGPP OR VUK_COMPILER_CLANGCL)
		 target_compile_options(vuk-tests PRIVATE -Wno-nullability-completeness)
	endif()
else()
	target_compile_definitions(vuk PRIVATE DOCTEST_CONFIG_DISABLE)
endif()
