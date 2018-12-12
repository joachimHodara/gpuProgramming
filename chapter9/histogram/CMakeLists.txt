cmake_minimum_required(VERSION 2.8)
set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(histogram LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME} "main.cu")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
