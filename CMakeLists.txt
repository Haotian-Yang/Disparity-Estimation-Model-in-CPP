#cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
cmake_minimum_required(VERSION 3.16 )
project(custom_ops)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#set(CMAKE_PREFIX_PATH "/usr/local/include/opencv/opencv4")
find_package(Torch REQUIRED)
include_directories(${TORCH_LIBRARIES_DIRS})

find_package(OpenCV 3 REQUIRED)

add_executable(cppLoadTorch cppLoadTorch.cpp)
target_link_libraries(cppLoadTorch ${TORCH_LIBRARIES} ${OpenCV_LIBS})
#target_link_libraries(cppLoadTorch ${OpenCV_LIBS})
set_property(TARGET cppLoadTorch PROPERTY CXX_STANDARD 14)
