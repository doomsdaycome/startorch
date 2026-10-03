cmake_minimum_required(VERSION 3.23)

project(startorch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build startorch as a shared library" OFF)
option(BUILD_TESTS "Build startorch unit tests" OFF)
option(BUILD_EXAMPLES "Build startorch example binaries" OFF)

add_library(startorch)
add_library(startorch::startorch ALIAS startorch)

target_sources(
    startorch
    PRIVATE
        src/darkside/memory/allocator.cpp
        src/darkside/memory/arena.cpp
        src/startorch/scalar/value.cpp
        src/startorch/scalar/pointer.cpp
        src/startorch/system/machine.cpp
        src/startorch/system/memory.cpp
)

target_include_directories(
    startorch
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(BUILD_SHARED_LIBS)
    include(GenerateExportHeader)
    generate_export_header(startorch)
endif()

if(BUILD_EXAMPLES)
    add_executable(startorch_example examples/main.cpp)
    target_link_libraries(startorch_example PRIVATE startorch::startorch)
endif()

if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL
            https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(startorch_test tests/test_tensor.cpp)

    target_link_libraries(
        startorch_test
        PRIVATE startorch::startorch GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(startorch_test)
endif()
