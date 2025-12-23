find_package(Doxygen REQUIRED)

# Find all the public headers
get_target_property(VUK_PUBLIC_HEADER_DIR vuk INTERFACE_INCLUDE_DIRECTORIES)
file(GLOB_RECURSE VUK_PUBLIC_HEADERS ${VUK_PUBLIC_HEADER_DIR}/*.hpp)

file(STRINGS doxy_dirs.txt DOXYGEN_INPUT_DIR_LIST)

# Convert relative paths to absolute paths
set(DOXYGEN_INPUT_DIR_LIST_ABS)
foreach(dir ${DOXYGEN_INPUT_DIR_LIST})
    get_filename_component(abs_dir "${dir}" ABSOLUTE)
    list(APPEND DOXYGEN_INPUT_DIR_LIST_ABS "${abs_dir}")
endforeach()

list(JOIN DOXYGEN_INPUT_DIR_LIST_ABS " " DOXYGEN_INPUT_DIR)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

#Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                   DEPENDS ${VUK_PUBLIC_HEADERS}
                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                   MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                   COMMENT "Generating docs")

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

find_package(Sphinx REQUIRED)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

# Only regenerate Sphinx when:
# - Doxygen has rerun
# - Our doc files have been updated
# - The Sphinx config has been updated
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                   COMMAND 
                     ${SPHINX_EXECUTABLE} -b html
                     # Tell Breathe where to find the Doxygen output
                     -Dbreathe_projects.vuk=${DOXYGEN_OUTPUT_DIR}/xml
                   ${SPHINX_SOURCE} ${SPHINX_BUILD}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   DEPENDS
                   # Other docs files you want to track should go here (or in some variable)
                   index.rst
				   topics/runtime.rst
				   topics/allocators.rst
				   topics/rendergraph.rst
				   topics/commandbuffer.rst
                   topics/vsl.rst
                   topics/extra.rst
                   ${DOXYGEN_INDEX_FILE}
                   MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                   COMMENT "Generating documentation with Sphinx")

# Nice named target so we can run the job easily
add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})