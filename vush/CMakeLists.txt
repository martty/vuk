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

add_library(vush)
target_compile_features(vush PUBLIC cxx_std_20)

if(VUSH_USE_EXTERNAL_JSON)
  find_package(nlohmann_json 3.2.0 REQUIRED)
else()
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  add_subdirectory(../ext/json nlohmann)
endif()

target_sources(vush PRIVATE 
    vush.cpp
)

target_include_directories(vush SYSTEM PUBLIC ../ext/plf_colony)
target_include_directories(vush SYSTEM PUBLIC ../ext/Mustache)
target_include_directories(vush PUBLIC .)

target_compile_options(vush PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/permissive->")
target_link_libraries_system(vush PUBLIC GSL nlohmann_json::nlohmann_json)

if (WIN32)
	target_compile_definitions(vuk PUBLIC NOMINMAX VC_EXTRALEAN WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS _SILENCE_CLANG_CONCEPTS_MESSAGE)
endif()

option(TESTING "ACTIVATE UNITTESTING" ON)

if (TESTING)
    enable_testing()

    if(TARGET Catch2::Catch2)
    else()
        add_subdirectory(../ext/Catch2 Catch2)
        set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/ext/Catch2/contrib" ${CMAKE_MODULE_PATH})
    endif()

    add_executable(vush-tests test_runner.cpp)
    target_link_libraries(vush-tests vush Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(vush-tests)
endif()
