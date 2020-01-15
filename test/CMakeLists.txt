cmake_minimum_required(VERSION 3.10)

mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

include_directories(${PROJECT_SOURCE_DIR})

set(SOURCES test_squad_utils.cpp)
add_executable(squad_utils-gtest ${SOURCES})
target_link_libraries(squad_utils-gtest gtest gtest_main hflt)
gtest_discover_tests(squad_utils-gtest WORKING_DIRECTORY $<TARGET_FILE_DIR:squad_utils-gtest>)

add_custom_command(TARGET squad_utils-gtest POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${PROJECT_SOURCE_DIR}/data" $<TARGET_FILE_DIR:squad_utils-gtest>/data)

