cmake_minimum_required(VERSION 3.24)

project(nano_infer
  VERSION 0.1.0
  DESCRIPTION "GPT-2 inference engine written from scratch in C++ and CUDA"
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Builds for the GPU in this machine. Override for a release build, for example
# -DCMAKE_CUDA_ARCHITECTURES="86;89;90".
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Off by default. Fast math changes numerics, and the first milestone is
# matching HuggingFace logits to a tight tolerance.
option(NANOINFER_FAST_MATH "Enable -use_fast_math in device code" OFF)
option(NANOINFER_LINEINFO "Emit line info for Nsight Compute source view" ON)

add_library(nanoinfer_core STATIC
  src/weights.cpp
  src/kernels/layernorm.cu
)

target_include_directories(nanoinfer_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(nanoinfer_core PUBLIC CUDA::cudart)

target_compile_options(nanoinfer_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

# Register pressure and occupancy show up here long before a profiler run does.
target_compile_options(nanoinfer_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

if(NANOINFER_LINEINFO)
  target_compile_options(nanoinfer_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

if(NANOINFER_FAST_MATH)
  target_compile_options(nanoinfer_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
endif()

add_executable(nano-infer src/main.cpp)
target_link_libraries(nano-infer PRIVATE nanoinfer_core)

enable_testing()
add_subdirectory(tests)
