cmake_minimum_required(VERSION 3.18)
project(busy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA GPU support" ON)

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        add_executable(busy busy.cpp gpu_worker.cu gpu_worker.h)
        target_compile_definitions(busy PRIVATE USE_CUDA)
        set_target_properties(busy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # Find CUDA runtime
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(busy PRIVATE CUDA::cudart pthread)

        message(STATUS "CUDA support enabled")
    else()
        message(STATUS "CUDA compiler not found, building without GPU support")
        add_executable(busy busy.cpp)
        target_link_libraries(busy PRIVATE pthread)
    endif()
else()
    message(STATUS "CUDA support disabled")
    add_executable(busy busy.cpp)
    target_link_libraries(busy PRIVATE pthread)
endif()

# Optimize for performance
target_compile_options(busy PRIVATE -O2 -march=native)
