cmake_minimum_required(VERSION 3.10)

mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

include_directories(${PROJECT_SOURCE_DIR})

set(SOURCES hflt_tests.cpp)
add_executable(hflt-gtest ${SOURCES})
target_link_libraries(hflt-gtest hflt gtest_main gtest)
gtest_discover_tests(hflt-gtest WORKING_DIRECTORY $<TARGET_FILE_DIR:hflt-gtest>)

# move test assets to build folder
add_custom_command(TARGET hflt-gtest POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${PROJECT_SOURCE_DIR}/data" $<TARGET_FILE_DIR:hflt-gtest>/data)
add_custom_command(TARGET hflt-gtest POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${CMAKE_CURRENT_LIST_DIR}/assets" $<TARGET_FILE_DIR:hflt-gtest>/assets)
add_custom_command(TARGET hflt-gtest POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${PROJECT_SOURCE_DIR}/models/sst2_trained" $<TARGET_FILE_DIR:hflt-gtest>/models/sst2_trained)
