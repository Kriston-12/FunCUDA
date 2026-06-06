cmake_minimum_required(VERSION 3.18)

project(Sgemm LANGUAGES CXX CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

function(add_reduce_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)

    target_compile_options(${target_name} PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
    )
endfunction()

add_reduce_executable(v0_global_memory v0_global_memory.cu)
