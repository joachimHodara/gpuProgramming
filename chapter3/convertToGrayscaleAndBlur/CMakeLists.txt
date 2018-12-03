cmake_minimum_required(VERSION 2.8)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(convertToGrayScale LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME} "main.cu")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

#target_link_libraries(${PROJECT_NAME} opencv_core opencv_imgcodecs opencv_imgproc opencv_highgui)
#target_include_directories(${PROJECT_NAME} PRIVATE /usr/include/opencv4/)
