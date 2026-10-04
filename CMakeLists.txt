cmake_minimum_required(VERSION 3.23)

project(startorch VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

option(BUILD_SHARED_LIBS "Build startorch as a shared library" OFF)
option(BUILD_TESTS "Build startorch unit tests" OFF)
option(BUILD_EXAMPLES "Build startorch example binaries" OFF)

add_library(startorch)
add_library(startorch::startorch ALIAS startorch)

target_sources(
    startorch
    PRIVATE
        src/darkside/memory/buffer.cpp
        src/darkside/scalar/utilities.cpp
        src/startorch/engine/memory.cpp
        src/startorch/engine/system.cpp
        src/startorch/scalar/pointer.cpp
        src/startorch/scalar/reference.cpp
        src/startorch/scalar/value.cpp
)

target_include_directories(
    startorch
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(startorch PRIVATE CUDA::cudart)

set_target_properties(
    startorch
    PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS OFF
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
