# - Provide a macro to embed binary files into the executable.
#
# This file is part of the Embed project: https://github.com/magcks/embed
#
# The module defines the macros:
#
#  EMBED_TARGET(<Name> <BinFile>)
#
# which will create  a custom rule to a assembly file. <BinFile> is
# the path to the binary file.
#
# The macro defines a set of variables:
#  EMBED_${Name}_DEFINED       - true is the macro ran successfully
#  EMBED_${Name}_INPUT         - The input source file, an alias for <BinFile>
#  EMBED_${Name}_OUTPUTS       - The source file generated
#
#  ====================================================================
#  Example:
#
#   find_package(Embed REQUIRED)
#   EMBED_TARGET(SHADER source.glsl)
#   add_executable(example main.cc ${EMBED_SHADER_OUTPUTS})
#  ====================================================================


cmake_minimum_required(VERSION 3.5)

set_property(GLOBAL PROPERTY RES_ID 16384)
set(STRUCT
"#include \"stddef.h\"
struct Res {
	const char *data\;
	const size_t size\;
}\;"
)

macro(EMBED_TARGET Name Input)
	get_filename_component(InputAbs "${Input}" REALPATH)
	if(VUK_COMPILER_MSVC)
		set(OutputRC "${CMAKE_CURRENT_BINARY_DIR}/${Name}.rc")
		set(OutputC "${CMAKE_CURRENT_BINARY_DIR}/${Name}.c")
		set(Outputs ${OutputRC} ${OutputC})
		get_property(RES_ID_L GLOBAL PROPERTY RES_ID)
		set(RCCODE "${RES_ID_L} RCDATA \"${InputAbs}\"\n")
		set(CODE
"#include \"windows.h\"
${STRUCT}
struct Res ${Name}(void) {
	HMODULE handle = GetModuleHandle(NULL)\;
	HRSRC res = FindResource(handle, MAKEINTRESOURCE(${RES_ID_L}), RT_RCDATA)\;
	struct Res r = {
		(const char*) LockResource(LoadResource(handle, res)),
		SizeofResource(handle, res)
	}\;
	return r\;
}"
		)
		file(WRITE ${OutputRC} ${RCCODE})
		file(WRITE ${OutputC} ${CODE})
		math(EXPR RES_ID_L "${RES_ID_L}+1")
		set_property(GLOBAL PROPERTY RES_ID ${RES_ID_L})
	else()
		if(APPLE)
			set(Section ".const_data")
			set(mangle "_")
		else()
			set(Section ".section .rodata")
			set(mangle "")
		endif()
		set(CODE
"${STRUCT}
asm(
	\"${Section}\\n\"
	\".align ${CMAKE_SIZEOF_VOID_P}\\n\"
	\"${mangle}data: .incbin \\\"${InputAbs}\\\"\\n\"
	\"${mangle}end_data:\\n\"
)\;
extern const char data[]\;
extern const char end_data[]\;
struct Res ${Name}(void) {
	struct Res r = { data, end_data - data }\;
	return r\;
}"
		)
		set(OutputC "${CMAKE_CURRENT_BINARY_DIR}/${Name}.c")
		set(Outputs ${OutputC})
		file(WRITE ${OutputC} ${CODE})

		add_custom_command(
			OUTPUT ${OutputC}
			COMMAND ${CMAKE_COMMAND} -E touch ${OutputC}.timestamp
			DEPENDS ${Input}
		)
	endif()
	set(EMBED_${Name}_DEFINED TRUE)
	set(EMBED_${Name}_INPUT ${Input})
	set(EMBED_${Name}_OUTPUTS ${Outputs})
	set(EMBED_${Name}_OUTPUTC ${OutputC})
	set(EMBED_${Name}_OUTPUTRC ${OutputRC})
endmacro()