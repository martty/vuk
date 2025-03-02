FetchContent_Declare(
  vk-bootstrap
  GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
  GIT_TAG        1f07cf143429c671a3dd778a8644e3f26668475d
)

FetchContent_MakeAvailable(
  vk-bootstrap
)

target_sources(vuk PRIVATE SimpleInit.cpp)
target_link_libraries(vuk PUBLIC vk-bootstrap)