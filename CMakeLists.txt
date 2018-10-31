cmake_minimum_required(VERSION 3.5)
project(algo_practice)

# CMake Package Collection
FIND_PACKAGE(OpenCV CONFIG PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
FIND_PACKAGE(Boost COMPONENTS system filesystem REQUIRED)

set(LIB_NAME "stixel_practice")

set(LIB_SRC
  src/main.cpp
  src/stixel_generator.cpp
  src/stereo_colorize.cpp
  src/ground_estimator.cpp
  src/hough_trans.cpp
  )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})

# Generate Exe Files
add_executable("${LIB_NAME}_test" ${LIB_SRC})
target_link_libraries("${LIB_NAME}_test" 
  ${OpenCV_LIBS} 
  ${Boost_LIBRARIES})

