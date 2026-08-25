cmake_minimum_required(VERSION 3.23)

project(
  startorch
  VERSION 0.1.0
  LANGUAGES CXX)

option(BUILD_SHARED_LIBS "Build startorch as a shared library" ON)
option(BUILD_TESTS "Build startorch unit tests" OFF)
option(BUILD_EXAMPLES "Build startorch example binaries" OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE STARTORCH_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(startorch ${STARTORCH_SOURCES})
add_library(startorch::startorch ALIAS startorch)

target_include_directories(
  startorch
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MSVC)
  target_compile_options(startorch PRIVATE /W4 /permissive-)
else()
  target_compile_options(startorch PRIVATE -Wall -Wextra -Wpedantic
                                           -Wconversion)
endif()

if(BUILD_TESTS)
  enable_testing()
  include(FetchContent)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

  add_executable(startorch_test ${TEST_SOURCES})
  target_link_libraries(startorch_test PRIVATE startorch::startorch
                                               GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(startorch_test)
endif()

if(BUILD_EXAMPLES)
  file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp")

  foreach(EXAMPLE_FILE ${EXAMPLE_SOURCES})
    cmake_path(GET EXAMPLE_FILE STEM EXAMPLE_TARGET_NAME)

    add_executable(${EXAMPLE_TARGET_NAME} ${EXAMPLE_FILE})
    target_link_libraries(${EXAMPLE_TARGET_NAME} PRIVATE startorch::startorch)
  endforeach()
endif()
