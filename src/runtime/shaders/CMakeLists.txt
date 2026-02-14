find_package(Vulkan COMPONENTS glslangValidator dxc)

set(DXC_FLAGS -spirv -fspv-target-env=vulkan1.2 -fvk-use-gl-layout -Zpc -O3 -Fo)
set(GLSLANG_FLAGS --target-env vulkan1.2)
add_subdirectory(SPD)

add_library(vuk_shaders)

target_shaders(vuk_shaders COMPILER glslangValidator ARGS ${GLSLANG_FLAGS} -o SHADERS compute_clear.comp)

target_link_libraries(vuk PUBLIC vuk_shaders)