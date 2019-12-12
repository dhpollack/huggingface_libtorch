cmake_minimum_required(VERSION 3.10)

# set the project name
project(huggingface-albert)

# add the executable
add_executable(huggingface-albert data_utils.cpp)

# add sentencepiece
find_library(SENTENCEPIECE_LIBRARY
    NAMES sentencepiece libsentencepiece libsentencepiece_train
    HINTS "${CMAKE_PREFIX_PATH}/lib"
)
target_link_libraries(huggingface-albert "${SENTENCEPIECE_LIBRARY}")

# add boost tokenizer  
# https://stackoverflow.com/questions/6646405/how-do-you-add-boost-libraries-in-cmakelists-txt
find_package(Boost 1.71.0 COMPONENTS tokenizer)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(huggingface-albert ${Boost_LIBRARIES})

# add torch
find_package(Torch REQUIRED)
target_link_libraries(huggingface-albert "${TORCH_LIBRARIES}")
set_property(TARGET huggingface-albert PROPERTY CXX_STANDARD 11)






