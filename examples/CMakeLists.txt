cmake_minimum_required(VERSION 3.7)
project(vuk-examples)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

function(ADD_EXAMPLE name)
set(FULL_NAME "vuk_example_${name}")
add_executable(${FULL_NAME})
target_sources(${FULL_NAME} PUBLIC 
 "${name}.cpp"
 example_runner_single.cpp)
target_link_libraries(${FULL_NAME} PUBLIC vuk)
endfunction(ADD_EXAMPLE)

ADD_EXAMPLE(01_triangle)
ADD_EXAMPLE(02_cube)
ADD_EXAMPLE(03_multipass)
