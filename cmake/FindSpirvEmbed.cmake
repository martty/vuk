cmake_minimum_required(VERSION 3.5)

include(vtkEncodeString)

# based on: https://github.com/tomilov/sah_kd_tree/blob/develop/cmake/Shaders.cmake
# IMPORTANT: you must use this function in the same directory as the target, otherwise CMake doesn't pick up the command
# usage: target_shaders(<target name> COMPILER <compiler name> ARGS <compiler args> SHADERS <input shader files>)
# last arg must be the cli flag indicating output, such as -o or -Fo
function(target_shaders target)
    get_property(c_ext_def TARGET ${target} PROPERTY C_EXTENSIONS DEFINED)
    get_property(c_ext TARGET ${target} PROPERTY C_EXTENSIONS)
    if(c_ext_def AND NOT c_ext)
        message(FATAL_ERROR "Shader binary include requires C extensions (for asm()) in target ${target}")
    endif()
    cmake_parse_arguments(PARSE_ARGV 1 target_shaders "" "COMPILER" "SHADERS;ARGS")
    foreach(shader_file IN LISTS target_shaders_SHADERS)
        target_sources("${target}" PRIVATE "${shader_file}")
        # construct the output file name
        cmake_path(RELATIVE_PATH shader_file OUTPUT_VARIABLE rel_path)
        cmake_path(GET shader_file FILENAME fname)
        cmake_path(APPEND CMAKE_CURRENT_BINARY_DIR ${fname} OUTPUT_VARIABLE bin_path)
        cmake_path(APPEND_STRING bin_path ".spv" OUTPUT_VARIABLE output_file)
        string(REPLACE "." "_" fname_us ${fname})
        # invoke shader compiler
        add_custom_command(OUTPUT ${output_file}
                           MAIN_DEPENDENCY "${shader_file}"
                           VERBATIM
                           WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                           COMMAND "${target_shaders_COMPILER}"
                           ARGS ${target_shaders_ARGS} "${output_file}" "${shader_file}"
        )
        vtk_encode_string(INPUT ${output_file} NAME "${fname_us}_shader" SOURCE_OUTPUT ofile HEADER_OUTPUT hfile BINARY)
        target_sources("${target}" PRIVATE ${ofile})
        cmake_path(GET hfile PARENT_PATH ofname)
        target_include_directories("${target}" PUBLIC "${ofname}")
    endforeach()
endfunction()

# IMPORTANT: you must use this function in the same directory as the target, otherwise CMake doesn't pick up the command
function(target_dist_spv target)
    get_property(c_ext_def TARGET ${target} PROPERTY C_EXTENSIONS DEFINED)
    get_property(c_ext TARGET ${target} PROPERTY C_EXTENSIONS)
    if(c_ext_def AND NOT c_ext)
        message(FATAL_ERROR "Shader binary include requires C extensions (for asm()) in target ${target}")
    endif()
    cmake_parse_arguments(PARSE_ARGV 1 target_shaders "" "" "SHADERS")
    foreach(shader_file IN LISTS target_shaders_SHADERS)
        # construct the output file name
        cmake_path(APPEND vuk-binary-dist_SOURCE_DIR "bin" ${shader_file} OUTPUT_VARIABLE in_file)
        cmake_path(APPEND CMAKE_CURRENT_BINARY_DIR ${shader_file} OUTPUT_VARIABLE output_file)
        cmake_path(GET shader_file FILENAME fname)
        string(REPLACE "." "_" fname_us ${fname})

        vtk_encode_string(INPUT ${in_file} NAME "${fname_us}_shader" SOURCE_OUTPUT ofile HEADER_OUTPUT hfile BINARY)
        target_sources("${target}" PRIVATE ${ofile})
        cmake_path(GET hfile PARENT_PATH ofname)
        target_include_directories("${target}" PUBLIC "${ofname}")
    endforeach()
endfunction()