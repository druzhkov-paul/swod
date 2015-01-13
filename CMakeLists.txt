CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(PROJECT_NAME swod)
project(${PROJECT_NAME})

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
elseif (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W3")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE})

set(SWOD_LIBRARY ${PROJECT_NAME})
set(SWOD_TESTS "${PROJECT_NAME}_test")

find_package(OpenCV REQUIRED)
find_package(MPI QUIET)


set(SWOD_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${SWOD_INCLUDE}")
#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")

set(LIBRARY_DEPS ${OpenCV_LIBS})

# BUILD
# TODO. add options to build tests and 3rdparty or not
add_subdirectory(3rdparty)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(apps)

# REPORT
message( STATUS "")
message( STATUS "=========== General configuration for ${PROJECT_NAME} ===========")
message( STATUS "")
message( STATUS "   Configuration:       ${CMAKE_BUILD_TYPE}")
message( STATUS "   OpenCV build path:   ${OpenCV_DIR}")
message( STATUS "   MPI :                ${MPI_FOUND}")
message( STATUS "======================================================")
message( STATUS "")
