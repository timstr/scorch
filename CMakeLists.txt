cmake_minimum_required(VERSION 3.10.0)

project(scorch C CXX)

set(scorch_includes
    include/scorch.hpp
)

add_library(scorch STATIC ${scorch_includes})
target_include_directories(scorch
	PUBLIC include
)
set_property(TARGET scorch PROPERTY CXX_STANDARD 17)
set_target_properties(scorch PROPERTIES LINKER_LANGUAGE CXX)

set(SCORCH_GENERATE_EXAMPLE OFF CACHE BOOL "When set to ON, the scorch example target will be generated")

if (SCORCH_GENERATE_EXAMPLE)
    add_executable(scorch_demo demo.cpp)
    set_property(TARGET scorch_demo PROPERTY CXX_STANDARD 17)
    target_link_libraries(scorch_demo PUBLIC scorch)

    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT scorch_demo)
endif()
