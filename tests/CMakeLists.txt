cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_testing()

project(tests)

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)

include_directories(
    ${PARENT_DIR}/include
    )
