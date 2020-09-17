cmake_minimum_required(VERSION 3.10)

find_package(Catch2 REQUIRED)
include_directories(${PROJECT_SOURCE_DIR})

set(SOURCES hflt_tests_catch2.cpp utils.cpp)
add_executable(hflt-tests ${SOURCES})
target_link_libraries(hflt-tests hflt Catch2::Catch2 stdc++fs)

include(Catch)
catch_discover_tests(hflt-tests)

# move test assets to build folder
add_custom_command(
  TARGET hflt-tests
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/data"
          $<TARGET_FILE_DIR:hflt-tests>/data
)
add_custom_command(
  TARGET hflt-tests
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_LIST_DIR}/assets"
          $<TARGET_FILE_DIR:hflt-tests>/assets
)
add_custom_command(
  TARGET hflt-tests
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/models/sst2_trained"
    $<TARGET_FILE_DIR:hflt-tests>/models/sst2_trained
)


