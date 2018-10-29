cmake_minimum_required(VERSION 2.8)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(gpu_programming)
add_executable(${PROJECT_NAME} "imflip.cpp")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME} pthread opencv_core opencv_imgcodecs opencv_imgproc opencv_highgui)
