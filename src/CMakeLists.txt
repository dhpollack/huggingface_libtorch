cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0074 NEW)

set(HEADER_FILES
    data_utils.h
    squad_utils.h
    )

set(SOURCE_FILES
    #data_utils.cpp
    run_model.cpp
    )

set(UTIL_FILES
    squad_utils.cpp
    data_utils.cpp
    tokenizer_base.cpp
    config_utils.cpp
   )

# add the shared library and executable
add_library(hflt SHARED ${UTIL_FILES})
add_executable(hflt-bin ${SOURCE_FILES})

# link the executable to the shared library
set(CMAKE_INSTALL_RPATH "${ORIGIN}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

set_target_properties(hflt-bin PROPERTIES OUTPUT_NAME "hflt")
set_target_properties(hflt PROPERTIES 
	                   VERSION ${PROJECT_VERSION} 
                           SOVERSION 0
                           PUBLIC_HEADER squad_utils.h)

# makes working with subdirectories easier, but right now not used
target_include_directories(hflt PRIVATE .)

# install
include(GNUInstallDirs)
install(TARGETS hflt
        EXPORT hfltTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS hflt-bin
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# link target libraries
target_link_libraries(hflt PUBLIC
                      ${TORCH_LIBRARIES}
                      )
target_link_libraries(hflt PRIVATE 
                      ${SENTENCEPIECE_LIBRARIES} 
                      nlohmann_json::nlohmann_json
                      )
target_link_libraries(hflt-bin hflt)

