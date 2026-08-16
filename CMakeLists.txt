cmake_minimum_required(VERSION 3.23)
project(startorch VERSION 0.1.0 LANGUAGES CXX)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

option(BUILD_SHARED_LIBS "Build startorch as a shared library" ON)
option(BUILD_TESTS "Build startorch tests using GoogleTest" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE STARTORCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(startorch ${STARTORCH_SOURCES})

target_include_directories(startorch
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_library(startorch::startorch ALIAS startorch)

if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")

    add_executable(startorch_test ${TEST_SOURCES})
    target_link_libraries(startorch_test PRIVATE startorch GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(startorch_test)
endif()
