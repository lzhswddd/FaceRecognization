cmake_minimum_required(VERSION 2.8)

project(FaceRecognization C CXX)
set(BUILD_USE_64BITS on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Bin)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/FaceDetection)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/FaceAlignment)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/FaceIdentification)

FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include 
    ${CMAKE_CURRENT_LIST_DIR}/FaceDetection/include 
    ${CMAKE_CURRENT_LIST_DIR}/FaceIdentification/include
    ${CMAKE_CURRENT_LIST_DIR}/FaceAlignment/include
    ${CMAKE_CURRENT_LIST_DIR}/MatTools/include
    ${OpenCV_INCLUDE_DIRS})

file(GLOB FR_SRC 
    ${CMAKE_CURRENT_LIST_DIR}/include/*.h 
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/MatTools/include/*.h
    ${CMAKE_CURRENT_LIST_DIR}/MatTools/src/*.cpp
    )
set(FR_COMPILE_CODE ${FR_SRC})

LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/FaceDetection)
LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/FaceAlignment)
LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/FaceIdentification)

# Target
add_library(FaceRecognization SHARED ${FR_COMPILE_CODE})
TARGET_LINK_LIBRARIES(FaceRecognization FaceDetection FaceAlignment FaceIdentification ${OpenCV_LIBS})

