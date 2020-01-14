cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0074 NEW)

# set the project name
project(huggingface-albert)

# tests
option(BUILD_TEST "Build c++ tests" OFF)
if (BUILD_TEST)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()

set(HEADER_FILES
    src/data_utils.h
    src/squad_utils.h
    )

set(SOURCE_FILES
    src/data_utils.cpp
    src/run_model.cpp
    #src/squad_utils.cpp
    )

# add the executable
add_executable(huggingface-albert ${SOURCE_FILES})

# add sentencepiece, consider making this a Find*.cmake file
if (DEFINED ENV{SENTENCEPIECE_ROOT})
    set(SENTENCEPIECE_ROOT $ENV{SENTENCEPIECE_ROOT})
else()
    set(SENTENCEPIECE_ROOT ${PROJECT_SOURCE_DIR}/third_party/sp)
endif()
find_library(SENTENCEPIECE_LIBRARIES
    NAMES sentencepiece libsentencepiece
    HINTS "${SENTENCEPIECE_ROOT}/lib")
set(SENTENCEPIECE_INCLUDE_DIRS ${SENTENCEPIECE_ROOT}/include)
include_directories(${SENTENCEPIECE_INCLUDE_DIRS})
target_link_libraries(huggingface-albert "${SENTENCEPIECE_LIBRARIES}")
# alternative method but apparently required pkg-config which isn't always available
#set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${SENTENCEPIECE_ROOT}/lib/pkgconfig")
#find_package(PkgConfig REQUIRED)
#pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)
#target_link_libraries(huggingface-albert "${SENTENCEPIECE_LINK_LIBRARIES})

# add boost tokenizer, which is header only so no components needed to be specified 
# https://stackoverflow.com/questions/6646405/how-do-you-add-boost-libraries-in-cmakelists-txt
find_package(Boost 1.45.0)
include_directories(${Boost_INCLUDE_DIRS})

# add torch
find_package(Torch REQUIRED)
target_link_libraries(huggingface-albert "${TORCH_LIBRARIES}")
set_property(TARGET huggingface-albert PROPERTY CXX_STANDARD 14)

# add nlohmann json
find_package(nlohmann_json PATHS "${PROJECT_SOURCE_DIR}/third_party/local"  REQUIRED)
target_link_libraries(huggingface-albert nlohmann_json::nlohmann_json)

