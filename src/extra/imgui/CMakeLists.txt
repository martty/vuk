FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG 4c0604ec2ee9a9121ea4ecb55543ff507ca81bcc
)

FetchContent_MakeAvailable(
  imgui
)

# imgui does not provide a CMake build script, hence we create a library here.
add_library(imgui OBJECT
  "${imgui_SOURCE_DIR}/imgui.cpp"
  "${imgui_SOURCE_DIR}/imgui_draw.cpp"
  "${imgui_SOURCE_DIR}/imgui_demo.cpp"
  "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
  "${imgui_SOURCE_DIR}/imgui_tables.cpp"
)
target_include_directories(imgui PUBLIC "${imgui_SOURCE_DIR}")

target_sources(vuk PRIVATE ImGuiIntegration.cpp)
target_shaders(imgui COMPILER glslangValidator ARGS --target-env vulkan1.2 -o SHADERS imgui.vert imgui.frag)

if (VUK_IMGUI_PLATFORM_BACKEND STREQUAL "")
	message(FATAL_ERROR "VUK_IMGUI_PLATFORM_BACKEND must be set to a valid imgui platform backend or to the string ''custom'")
endif()

if (NOT VUK_IMGUI_PLATFORM_BACKEND STREQUAL "custom")
	add_library(vuk-extra-imgui-platform INTERFACE)
	target_sources(vuk-extra-imgui-platform INTERFACE "${imgui_SOURCE_DIR}/backends/imgui_impl_${VUK_EXTRA_IMGUI_PLATFORM_BACKEND}.cpp")
endif()
target_link_libraries(vuk PUBLIC imgui)