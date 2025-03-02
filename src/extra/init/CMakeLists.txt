FetchContent_Declare(
  vk-bootstrap
  GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
  GIT_TAG        e7ea02d1917016f2dc6ba543b1a61366d7827788
)

FetchContent_MakeAvailable(
  vk-bootstrap
)

target_sources(vuk PRIVATE SimpleInit.cpp)
target_link_libraries(vuk PUBLIC vk-bootstrap)