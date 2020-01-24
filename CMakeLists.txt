cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set the project name: Huggingface Libtorch i.e. HfLt
project(hflt VERSION 0.0.1 DESCRIPTION "huggingface transformers inference in c++")

# add sentencepiece, consider making this a Find*.cmake file
if (DEFINED ENV{SENTENCEPIECE_ROOT})
    set(SENTENCEPIECE_ROOT $ENV{SENTENCEPIECE_ROOT})
else()
    set(SENTENCEPIECE_ROOT ${PROJECT_SOURCE_DIR}/third_party/local)
endif()
find_library(SENTENCEPIECE_LIBRARIES IMPORTED
    NAMES sentencepiece libsentencepiece
    PATHS "${PROJECT_SOURCE_DIR}/third_party/local"
    HINTS "${SENTENCEPIECE_ROOT}/lib")
set(SENTENCEPIECE_INCLUDE_DIRS ${SENTENCEPIECE_ROOT}/include)
include_directories(${SENTENCEPIECE_INCLUDE_DIRS})

# alternative method but apparently required pkg-config which isn't always available
#set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${SENTENCEPIECE_ROOT}/lib/pkgconfig")
#find_package(PkgConfig REQUIRED)
#pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)
#target_link_libraries(hflt "${SENTENCEPIECE_LINK_LIBRARIES})

# add boost config_utils. which is header only so no components needed to be specified 
# https://stackoverflow.com/questions/6646405/how-do-you-add-boost-libraries-in-cmakelists-txt
find_package(Boost 1.45.0)
include_directories(${Boost_INCLUDE_DIRS})

# add torch
find_package(Torch REQUIRED IMPORTED)

# add nlohmann json
find_package(nlohmann_json PATHS "${PROJECT_SOURCE_DIR}/third_party/local"  REQUIRED IMPORTED)

add_subdirectory(src)

# tests
option(BUILD_TEST "Build c++ tests" OFF)
if (BUILD_TEST)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()

#include(CMakePackageConfigHelpers)
#configure_package_config_file(
#        "hflt.cmake.in"
#        "hfltConfig.cmake"
#        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hflt
#        PATH_VARS
#        CMAKE_INSTALL_LIBDIR)
#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hfltConfig.cmake"
#        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hflt")
#install(EXPORT hfltTargets
#        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hflt"
#        FILE hfltTargets.cmake)
