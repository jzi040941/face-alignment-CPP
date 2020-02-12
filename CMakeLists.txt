cmake_minimum_required(VERSION 2.8)
project(face_alignment_cpp)
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -O3")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(face_alignment_cpp_src
  ${CMAKE_CURRENT_LIST_DIR}/src/FaceDetectorOpencv.cpp
)

set(face_alignment_cpp_include
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/include
)

set(face_alignment_cpp_libs
  ${OpenCV_LIBS}
)

set(face_alignment_cpp_test
  ${CMAKE_CURRENT_LIST_DIR}/test/face_alignment_cpp_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test/face_alignment_cpp_Realtimetest.cpp
) 
#-- Setup CMake to run tests
enable_testing()

include_directories(${face_alignment_cpp_include})

foreach(test ${face_alignment_cpp_test})
  get_filename_component(test_name ${test} NAME_WE)
  add_executable(${test_name}
    ${face_alignment_cpp_src}
    ${test}
  )
  target_link_libraries(${test_name} ${face_alignment_cpp_libs})
  add_test(NAME ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} COMMAND ${test_name})
endforeach()

#include_directories( ${face_alignment_cpp_include} )
#add_executable(classifier prediction.cpp)
#target_link_libraries(classifier ${OpenCV_LIBS} ${TORCH_LIBRARIES})
