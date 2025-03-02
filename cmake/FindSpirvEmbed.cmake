cmake_minimum_required(VERSION 3.5)

# based on: https://github.com/tomilov/sah_kd_tree/blob/develop/cmake/Shaders.cmake
# IMPORTANT: you must use this function in the same directory as the target, otherwise CMake doesn't pick up the command
# usage: target_shaders(<target name> COMPILER <compiler name> ARGS <compiler args> SHADERS <input shader files>)
# last arg must be the cli flag indicating output, such as -o or -Fo
function(target_shaders target)
    cmake_parse_arguments(PARSE_ARGV 1 target_shaders "" "COMPILER" "SHADERS;ARGS")
    foreach(shader_file IN LISTS target_shaders_SHADERS)
        target_sources("${target}" PRIVATE "${shader_file}")
        # construct the output file name
        cmake_path(RELATIVE_PATH shader_file OUTPUT_VARIABLE rel_path)
        cmake_path(GET shader_file FILENAME fname)
        cmake_path(APPEND CMAKE_CURRENT_BINARY_DIR ${fname} OUTPUT_VARIABLE bin_path)
        cmake_path(APPEND_STRING bin_path ".spv" OUTPUT_VARIABLE output_file)

        # invoke shader compiler
        add_custom_command(OUTPUT ${output_file}
                           MAIN_DEPENDENCY "${shader_file}"
                           VERBATIM
                           WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                           COMMAND "${target_shaders_COMPILER}"
                           ARGS ${target_shaders_ARGS} "${output_file}" "${shader_file}"
        )
        # generate data embedding
        # this will use incbin on non-MSVC and RC files on MSVC
        string(REPLACE "." "_" embed_name ${shader_file})
        EMBED_TARGET(${embed_name} ${output_file})
        target_sources("${target}" PRIVATE ${EMBED_${embed_name}_OUTPUTC})
        # for MSVC, we sidestep CMake and compile the res files manually, so we can refer to their path
        # and put the resulting .res files in the INTERFACE link options for libraries that don't link
        # and link directly for targets that do
        if(VUK_COMPILER_MSVC)
            add_custom_command(OUTPUT ${output_file}.res
                               MAIN_DEPENDENCY ${EMBED_${embed_name}_OUTPUTRC}
                               DEPENDS ${output_file}
                               VERBATIM
                               WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                               COMMAND ${CMAKE_RC_COMPILER}
                               ARGS /nologo /fo ${output_file}.res ${EMBED_${embed_name}_OUTPUTRC}
            )
            target_sources("${target}" PRIVATE ${output_file}.res)
            get_target_property(target_type "${target}" TYPE)
            if (target_type STREQUAL "STATIC_LIBRARY")
                target_link_options("${target}" INTERFACE ${output_file}.res)
            elseif(target_type STREQUAL "OBJECT_LIBRARY")
                target_link_options("${target}" INTERFACE ${output_file}.res)
            else()
                target_link_options("${target}" PRIVATE ${output_file}.res)
            endif()
        endif()
    endforeach()
endfunction()