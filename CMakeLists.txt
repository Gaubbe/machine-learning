cmake_minimum_required(VERSION 3.25.3)
project(machine-learning)

file(GLOB_RECURSE ML_SOURCES ./machine-learning/*.cpp)

add_executable(ml ${ML_SOURCES})
target_include_directories(ml PRIVATE ./lib)
