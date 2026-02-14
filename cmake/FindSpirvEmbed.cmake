cmake_minimum_required(VERSION 3.5)

include(vtkEncodeString)

# based on: https://github.com/tomilov/sah_kd_tree/blob/develop/cmake/Shaders.cmake
# IMPORTANT: you must use this function in the same directory as the target, otherwise CMake doesn't pick up the command
# usage: target_shaders(<target name> COMPILER <compiler name> ARGS <compiler args> SHADERS <input shader files>)
# last arg must be the cli flag indicating output, such as -o or -Fo
function(target_shaders target)
    cmake_parse_arguments(PARSE_ARGV 1 target_shaders "" "COMPILER" "SHADERS;ARGS")
    foreach(shader_file IN LISTS target_shaders_SHADERS)
         # construct the output file name
        cmake_path(RELATIVE_PATH shader_file OUTPUT_VARIABLE rel_path)
        cmake_path(GET shader_file FILENAME fname)
        cmake_path(APPEND CMAKE_SOURCE_DIR "bin" ${fname} OUTPUT_VARIABLE bin_path)
        cmake_path(APPEND_STRING bin_path ".spv" OUTPUT_VARIABLE output_file)

        add_custom_target("vuk_shader_binaries_${shader_file}")
        # invoke shader compiler
        add_custom_command(TARGET "vuk_shader_binaries_${shader_file}"
                           MAIN_DEPENDENCY "${shader_file}"
                           VERBATIM
                           WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                           COMMAND "${target_shaders_COMPILER}"
                           ARGS ${target_shaders_ARGS} "${output_file}" "${shader_file}"
        )
        add_dependencies(vuk_shader_binaries "vuk_shader_binaries_${shader_file}")
        cmake_path(GET output_file FILENAME fname)
        string(REPLACE "." "_" fname_us ${fname})
        vtk_encode_string(INPUT ${output_file} NAME "${fname_us}_shader" SOURCE_OUTPUT ofile HEADER_OUTPUT hfile BINARY)
        target_sources("${target}" PRIVATE ${ofile})
        cmake_path(GET hfile PARENT_PATH ofname)
        target_include_directories("${target}" PUBLIC "${ofname}")
    endforeach()
endfunction()