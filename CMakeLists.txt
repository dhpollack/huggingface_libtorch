cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set the project name: Huggingface Libtorch i.e. HfLt
project(
  hflt
  VERSION 0.0.1
  DESCRIPTION "huggingface transformers inference in c++"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(DO_CLANG_TIDY "clang tidy output" ON)
if(DO_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
  set(CMAKE_CXX_CLANG_TIDY
      "${CLANG_TIDY_EXE};-checks=-*,readability-*,modernize-*,-modernize-use-trailing-return-type;-header-filter='${CMAKE_SOURCE_DIR}/*'"
      CACHE STRING "" FORCE
  )
  # set(CMAKE_CXX_CLANG_TIDY clang-tidy
  # -checks=-*,readability-*,modernize-*,llvm-*)
endif()

# add sentencepiece, consider making this a Find*.cmake file
if(DEFINED ENV{SENTENCEPIECE_ROOT})
  set(SENTENCEPIECE_ROOT $ENV{SENTENCEPIECE_ROOT})
else()
  set(SENTENCEPIECE_ROOT ${PROJECT_SOURCE_DIR}/third_party/local)
endif()
find_library(
  SENTENCEPIECE_LIBRARIES IMPORTED
  NAMES sentencepiece libsentencepiece
  PATHS "${PROJECT_SOURCE_DIR}/third_party/local"
  HINTS "${SENTENCEPIECE_ROOT}/lib"
)
set(SENTENCEPIECE_INCLUDE_DIRS ${SENTENCEPIECE_ROOT}/include)
include_directories(${SENTENCEPIECE_INCLUDE_DIRS})
if(FALSE)
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/sentencepiece")
set(SENTENCEPIECE_LIBRARIES sentencepiece sentencepiece_train)
endif()

# add boost config_utils. which is header only so no components needed to be
# specified
# https://stackoverflow.com/questions/6646405/how-do-you-add-boost-libraries-in-cmakelists-txt
find_package(Boost 1.45.0)
include_directories(${Boost_INCLUDE_DIRS})

# add torch
find_package(Torch REQUIRED IMPORTED)

# add nlohmann json
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/json" EXCLUDE_FROM_ALL)

# add structopt
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/structopt" EXCLUDE_FROM_ALL)

add_subdirectory(src)

# tests
option(HFLT_BUILD_TEST "Build c++ tests" OFF)
if(HFLT_BUILD_TEST)
  include(CTest)
  add_subdirectory(test)
endif()

# coverage
find_program(LCOV_PATH lcov)
# Where TARGET_NAME is the executable target's name
add_custom_target(
  ccov-${CMAKE_PROJECT_NAME}
  COMMAND LLVM_PROFILE_FILE=${CMAKE_PROJECT_NAME}.profraw
          $<TARGET_FILE:${CMAKE_PROJECT_NAME}>
  COMMAND llvm-profdata merge -sparse ${CMAKE_PROJECT_NAME}.profraw -o
          ${CMAKE_PROJECT_NAME}.profdata
  COMMAND llvm-cov report $<TARGET_FILE:${CMAKE_PROJECT_NAME}>
          -instr-profile=${CMAKE_PROJECT_NAME}.profdata
  DEPENDS ${CMAKE_PROJECT_NAME}
)

include(CMakePackageConfigHelpers)
configure_package_config_file("hflt.cmake.in" "hfltConfig.cmake" INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hflt PATH_VARS CMAKE_INSTALL_LIBDIR)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hfltConfig.cmake" DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hflt")
install(EXPORT hfltTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hflt" FILE hfltTargets.cmake)
