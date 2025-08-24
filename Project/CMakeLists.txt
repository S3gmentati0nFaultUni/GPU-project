cmake_minimum_required(VERSION 3.24)

project(GPUProject LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA standard
set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_HOST_COMPILER g++)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Debugging statement
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

# Add test executable
add_executable(
	mst
	DataStructure/Graph.cu
	DataStructure/ReadGraph.cu
	mst.cu
	MST_hk.cu
)