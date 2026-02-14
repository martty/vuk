find_package(Vulkan COMPONENTS glslangValidator dxc)

set(DXC_FLAGS -spirv -fspv-target-env=vulkan1.2 -fvk-use-gl-layout -Zpc -O3 -Fo)
add_subdirectory(SPD)