cmake_minimum_required(VERSION 3.18)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()

project(Reduce LANGUAGES CXX CUDA)

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

# add_reduce_executable(my_reduce_v0_global_memory my_reduce_v0_global_memory.cu)

# Enable this after my_reduce_v1_shared_memory.cu has a main().
# add_reduce_executable(my_reduce_v1_shared_memory my_reduce_v1_shared_memory.cu)

# add_reduce_executable(my_reduce_v2_warp_shuffle my_reduce_v2_warp_shuffle.cu)
add_reduce_executable(my_reduce_v2_no_divergence_bank_conflict my_reduce_v2_no_divergence_bank_conflict.cu)