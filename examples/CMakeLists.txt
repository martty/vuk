cmake_minimum_required(VERSION 3.7)
project(vuk-examples)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(vuk_all_examples)
target_link_libraries(vuk_all_examples PUBLIC vuk)
target_include_directories(vuk_all_examples PRIVATE ../ext/stb)
target_sources(vuk_all_examples PRIVATE example_browser.cpp imgui.cpp stbi.cpp)

function(ADD_EXAMPLE name)
    set(FULL_NAME "vuk_example_${name}")
    add_executable(${FULL_NAME})
    target_sources(${FULL_NAME} PRIVATE "${name}.cpp" example_runner_single.cpp imgui.cpp stbi.cpp)
    target_sources(vuk_all_examples PRIVATE "${name}.cpp")
    target_include_directories(${FULL_NAME} PRIVATE ../ext/stb)
    target_link_libraries(${FULL_NAME} PUBLIC vuk)
endfunction(ADD_EXAMPLE)

ADD_EXAMPLE(01_triangle)
ADD_EXAMPLE(02_cube)
ADD_EXAMPLE(03_multipass)
ADD_EXAMPLE(04_texture)
ADD_EXAMPLE(05_deferred)
