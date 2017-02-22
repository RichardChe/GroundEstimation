cmake_minimum_required(VERSION 2.8)

set(OpenCV_DIR "/home/lenovo/opencv-3.0.0/release")
project( GroundEstimation )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(SYSUStereo SHARED SYSUStereo.cpp)
target_link_libraries(SYSUStereo ${OpenCV_LIBS})

add_library(GroundEstimator SHARED GroundEstimator.cpp)
target_link_libraries(GroundEstimator SYSUStereo ${OpenCV_LIBS})

#add_executable( test main.cpp )
#target_link_libraries( test SYSUStereo ${OpenCV_LIBS} )

add_executable( test main.cpp)
target_link_libraries(test SYSUStereo GroundEstimator ${OpenCV_LIBS})
