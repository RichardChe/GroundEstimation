cmake_minimum_required(VERSION 2.8)

project( GroundEstimation )

set(OpenCV_DIR "/home/lenovo/opencv-3.0.0/release")

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

list(APPEND SOURCEFILES ${PROJECT_SOURCE_DIR}/src/MatrixOp.cpp
						${PROJECT_SOURCE_DIR}/src/GroundEstimator.cpp
						${PROJECT_SOURCE_DIR}/src/RansacPlane.cpp
						${PROJECT_SOURCE_DIR}/src/SGMStereo.cpp
						${PROJECT_SOURCE_DIR}/src/Visualizer.cpp
						${PROJECT_SOURCE_DIR}/src/main.cpp)

add_executable(${PROJECT_NAME} ${SOURCEFILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
