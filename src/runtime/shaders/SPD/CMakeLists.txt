add_library(SPD)

target_shaders(SPD COMPILER Vulkan::dxc_exe ARGS -T cs_6_7 ${DXC_FLAGS} SHADERS spd.cs.hlsl)

target_link_libraries(vuk PUBLIC SPD)