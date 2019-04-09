cmake_minimum_required(VERSION 2.8)

project(FaceDetection C CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/ncnn
        ${CMAKE_CURRENT_LIST_DIR}/ncnn/layer
        ${CMAKE_CURRENT_LIST_DIR}/include)

file(GLOB MTCNN_SRC ${CMAKE_CURRENT_LIST_DIR}/include/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
        )
set(MTCNN_COMPILE_CODE ${MTCNN_SRC})
file(GLOB NCNN_SRC
        ${CMAKE_CURRENT_LIST_DIR}/ncnn/*.h
        ${CMAKE_CURRENT_LIST_DIR}/ncnn/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ncnn/layer/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ncnn/layer/*.h
        ${CMAKE_CURRENT_LIST_DIR}/ncnn/layer/x86/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ncnn/layer/x86/*.h
        ${CMAKE_CURRENT_LIST_DIR}/ncnn/layer/arm/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ncnn/layer/arm/*.h
        )
set(NCNN_COMPILE_CODE ${NCNN_SRC})
add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
add_definitions(-Ofast)
add_definitions(-ffast-math)
add_definitions(-ftree-vectorize)
add_definitions(-fvisibility=hidden -fvisibility-inlines-hidden)
add_library(FaceDetection SHARED ${MTCNN_COMPILE_CODE} ${NCNN_COMPILE_CODE})
