function(compile_shaders target)
    cmake_parse_arguments(PARSE_ARGV 1 target_shaders "" "COMPILER" "SHADERS;ARGS")
    foreach(shader_file IN LISTS target_shaders_SHADERS)
        target_sources("${target}" PRIVATE "${shader_file}")
        # construct the output file name
        cmake_path(RELATIVE_PATH shader_file OUTPUT_VARIABLE rel_path)
        cmake_path(GET shader_file FILENAME fname)
        cmake_path(APPEND CMAKE_SOURCE_DIR "bin" ${fname} OUTPUT_VARIABLE bin_path)
        cmake_path(APPEND_STRING bin_path ".spv" OUTPUT_VARIABLE output_file)

        # invoke shader compiler
        add_custom_command(OUTPUT ${output_file}
                           MAIN_DEPENDENCY "${shader_file}"
                           VERBATIM
                           WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                           COMMAND "${target_shaders_COMPILER}"
                           ARGS ${target_shaders_ARGS} "${output_file}" "${shader_file}"
        )
    endforeach()
endfunction()