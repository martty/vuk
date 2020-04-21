cmake_minimum_required(VERSION 3.7)

##
## PROJECT
## name and version
##
project(nlohmann_json VERSION 3.7.3 LANGUAGES CXX)

##
## TARGET
## create target and add include path
##
add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
    target_compile_features(nlohmann_json INTERFACE cxx_range_for)
else()
    target_compile_features(nlohmann_json INTERFACE cxx_std_11)
endif()

target_include_directories(nlohmann_json INTERFACE "${PROJECT_SOURCE_DIR}/single_include")

## add debug view definition file for msvc (natvis)
if (MSVC)
    set(NLOHMANN_ADD_NATVIS TRUE)
    set(NLOHMANN_NATVIS_FILE "nlohmann_json.natvis")
    target_sources(nlohmann_json INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/${NLOHMANN_NATVIS_FILE}")
endif()
