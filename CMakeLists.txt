cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0074 NEW)

# set the project name: Huggingface Libtorch i.e. HfLt
project(hflt)

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
    )

set(UTIL_FILES
    src/squad_utils.cpp
   )

# add the executable
add_library(hflt SHARED ${UTIL_FILES})
add_executable(hflt-bin ${SOURCE_FILES})

# add sentencepiece, consider making this a Find*.cmake file
if (DEFINED ENV{SENTENCEPIECE_ROOT})
    set(SENTENCEPIECE_ROOT $ENV{SENTENCEPIECE_ROOT})
else()
    set(SENTENCEPIECE_ROOT ${PROJECT_SOURCE_DIR}/third_party/local)
endif()
find_library(SENTENCEPIECE_LIBRARIES
    NAMES sentencepiece libsentencepiece
    PATHS "${PROJECT_SOURCE_DIR}/third_party/local"
    HINTS "${SENTENCEPIECE_ROOT}/lib")
set(SENTENCEPIECE_INCLUDE_DIRS ${SENTENCEPIECE_ROOT}/include)
include_directories(${SENTENCEPIECE_INCLUDE_DIRS})
#target_link_libraries(huggingface-albert "${SENTENCEPIECE_LIBRARIES}")
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
target_link_libraries(hflt-bin "${TORCH_LIBRARIES}")
set_property(TARGET hflt-bin PROPERTY CXX_STANDARD 14)


# add nlohmann json
find_package(nlohmann_json PATHS "${PROJECT_SOURCE_DIR}/third_party/local"  REQUIRED)
target_link_libraries(hflt 
                      ${SENTENCEPIECE_LIBRARIES} 
                      #${TORCH_LIBRARIES}
                      nlohmann_json::nlohmann_json)
set_property(TARGET hflt PROPERTY CXX_STANDARD 14)
target_link_libraries(hflt-bin hflt)
#target_link_libraries(huggingface-albert nlohmann_json::nlohmann_json)

