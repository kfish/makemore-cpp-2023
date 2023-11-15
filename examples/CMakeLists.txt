cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)

add_compile_options(-O3 -Wall -Wextra -Wpedantic)

project(examples)

find_package(PythonLibs 3 REQUIRED)

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${PARENT_DIR}/submodule/eigen/
    ${PARENT_DIR}/submodule/matplotlib-cpp
    ${PARENT_DIR}/include
    )

add_executable(matplotlib-example matplotlib-example.cpp)
add_test(NAME matplotlib-example
         COMMAND matplotlib-example)
target_link_libraries(matplotlib-example ${PYTHON_LIBRARIES})

add_executable(matplotlib-eigen-example matplotlib-eigen-example.cpp)
add_test(NAME matplotlib-eigen-example
         COMMAND matplotlib-eigen-example)
target_link_libraries(matplotlib-eigen-example ${PYTHON_LIBRARIES})

add_executable(distribution distribution.cpp)
add_test(NAME distribution
         COMMAND distribution)

add_executable(bigram-array bigram-array.cpp)
add_test(NAME bigram-array
         COMMAND bigram-array)
target_link_libraries(bigram-array ${PYTHON_LIBRARIES})

add_executable(bigram bigram.cpp)
add_test(NAME bigram
         COMMAND bigram)
target_link_libraries(bigram ${PYTHON_LIBRARIES})

add_executable(onehot-emma onehot-emma.cpp)
add_test(NAME onehot-emma
         COMMAND onehot-emma)
target_link_libraries(onehot-emma ${PYTHON_LIBRARIES})

add_executable(regression0 regression0.cpp)
add_test(NAME regression0
         COMMAND regression0)

add_executable(logit-backprop logit-backprop.cpp)
add_test(NAME logit-backprop
         COMMAND logit-backprop)
target_link_libraries(logit-backprop ${PYTHON_LIBRARIES})

add_executable(logit-node-backprop logit-node-backprop.cpp)
add_test(NAME logit-node-backprop
         COMMAND logit-node-backprop)
target_link_libraries(logit-node-backprop ${PYTHON_LIBRARIES})
