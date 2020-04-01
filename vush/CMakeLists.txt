cmake_minimum_required(VERSION 3.7)
project(vush)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# https://stackoverflow.com/a/52136398
function(target_link_libraries_system target access)
  set(libs ${ARGN})
  foreach(lib ${libs})
    get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${target} SYSTEM ${access} ${lib_include_dirs})
    target_link_libraries(${target} ${access} ${lib})
  endforeach(lib)
endfunction(target_link_libraries_system)

set(GSL_CXX_STANDARD 20)

add_executable(vush)
target_compile_features(vush PUBLIC cxx_std_20)

target_sources(vush PRIVATE 
    vush.cpp
)

target_include_directories(vush PUBLIC ../ext/plf_colony)
target_include_directories(vush PUBLIC .)

target_compile_options(vush PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/permissive->")
target_link_libraries_system(vush PUBLIC GSL)

if (WIN32)
	target_compile_definitions(vuk PUBLIC NOMINMAX VC_EXTRALEAN WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS _SILENCE_CLANG_CONCEPTS_MESSAGE)
endif()