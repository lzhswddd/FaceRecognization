cmake_minimum_required(VERSION 2.8)

project(FaceIdentification C CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/tools
        ${CMAKE_CURRENT_LIST_DIR}/src)

file(GLOB IDENTIFICATION_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/include/*.h
        ${CMAKE_CURRENT_LIST_DIR}/tools/*.cpp
        )
set(IDENTIFICATION_COMPILE_CODE ${IDENTIFICATION_SRC})
add_library(FaceIdentification SHARED ${IDENTIFICATION_COMPILE_CODE})
