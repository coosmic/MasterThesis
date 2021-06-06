cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(PlantGrowthMonitoring)

find_package(PCL 1.3 REQUIRED COMPONENTS)

find_package(CGAL REQUIRED PATHS /home/solomon/Thesis/CGAL-5.2)

find_package(OpenGR QUIET PATHS /home/solomon/Thesis/OpenGR/OpenGR/build)
include(CGAL_OpenGR_support)

if(TARGET CGAL::OpenGR_support)
    message(OpenGR support!)
endif()

find_package(libpointmatcher QUIET)
include(CGAL_pointmatcher_support)

if(TARGET CGAL::pointmatcher_support)
    message(pointmatcher support!)
endif()

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pgm main.cpp)
target_link_libraries(pgm ${PCL_LIBRARIES})
target_link_libraries(pgm CGAL::CGAL CGAL::pointmatcher_support CGAL::OpenGR_support)
