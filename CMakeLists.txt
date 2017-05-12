cmake_minimum_required(VERSION 2.8)

set(OpenCV_DIR "/home/lenovo/opencv-3.0.0/release")
project( GroundEstimation )

find_package( OpenCV REQUIRED )
<<<<<<< HEAD
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

list(APPEND SOURCEFILES ${PROJECT_SOURCE_DIR}/src/MatrixOp.cpp
						${PROJECT_SOURCE_DIR}/src/GroundEstimator.cpp
						${PROJECT_SOURCE_DIR}/src/RansacPlane.cpp
						${PROJECT_SOURCE_DIR}/src/SGMStereo.cpp
						${PROJECT_SOURCE_DIR}/src/Visualizer.cpp
						${PROJECT_SOURCE_DIR}/src/main.cpp)

add_executable(test ${SOURCEFILES})
target_link_libraries(test ${OpenCV_LIBS})
=======
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(SYSUStereo SHARED SYSUStereo.cpp)
target_link_libraries(SYSUStereo ${OpenCV_LIBS})

add_library(GroundEstimator SHARED GroundEstimator.cpp)
target_link_libraries(GroundEstimator SYSUStereo ${OpenCV_LIBS})

#add_executable( test main.cpp )
#target_link_libraries( test SYSUStereo ${OpenCV_LIBS} )

add_executable( test main.cpp)
target_link_libraries(test SYSUStereo GroundEstimator ${OpenCV_LIBS})
>>>>>>> ee751d06e7b243f505ed47bd3eebb29b03b0ff9c
