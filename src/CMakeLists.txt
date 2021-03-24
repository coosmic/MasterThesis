cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(PlantGrowthMonitoring)

find_package(PCL 1.3 REQUIRED COMPONENTS)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pgm main.cpp)
target_link_libraries(pgm ${PCL_LIBRARIES})
