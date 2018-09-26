# ########################################################################
# Practical Course: GPU Programming in Computer Vision (SS 2018)
# Technical University of Munich, Computer Vision Group
# Bjoern Haefner, Robert Maier, David Schubert
# ########################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

PROJECT(KillingFusionCuda LANGUAGES C CXX)

# ------------------------------------------------------------------------
# CMake configuration
IF(NOT CMAKE_BUILD_TYPE)
    # CMAKE_BUILD_TYPE: RelWithDebInfo, Release or Debug
    SET(CMAKE_BUILD_TYPE RelWithDebInfo)
ENDIF()

# CMake paths
# Externals directories (from environment variable "EXT")
LIST(APPEND CMAKE_PREFIX_PATH "$ENV{EXT}")
# include files in cmake subfolder
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# CMake executable output path
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# ------------------------------------------------------------------------
# dependencies

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch sm_35)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v --use_fast_math -O3 --compiler-options -Wall)

# Eigen
#SET(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/third_party/include/eigen3)
FIND_PACKAGE(Eigen REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# OpenCV
FIND_PACKAGE(OpenCV 2.4.8 COMPONENTS core highgui imgproc video REQUIRED
)
IF(OPENCV_CORE_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
ENDIF()

# ------------------------------------------------------------------------
# C++ flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -Wno-deprecated -pthread")
IF(CMAKE_COMPILER_IS_GNUCC)
    # Add -O0 to remove optimizations when using gcc
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
ENDIF()

#==============================================================================
# C++11 support
#==============================================================================
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
IF(COMPILER_SUPPORTS_CXX11)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    IF(BUILD_CUDA)
        SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
    ENDIF()
ELSEIF(COMPILER_SUPPORTS_CXX0X)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    IF(BUILD_CUDA)
        SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++0x)
    ENDIF()
ELSE()
    MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
ENDIF()


# ------------------------------------------------------------------------
# preprocessor defines

# source directory
ADD_DEFINITIONS(-DKILLINGFUSION_SOURCE_DIR=${PROJECT_SOURCE_DIR})

# ------------------------------------------------------------------------
# include directory
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)

# ------------------------------------------------------------------------
# executable
SET(APP_NAME "killingfusion")
CUDA_ADD_EXECUTABLE(${APP_NAME}
        src/marching_cubes.h src/marching_cubes.cpp
        src/tsdf_volume.h src/tsdf_volume.cpp
        src/dataset.h src/dataset.cpp
        src/helper.cuh src/helper.cu
		src/convolution.cuh src/convolution.cu
		src/divergence.cuh src/divergence.cu
		src/optimizer.cuh src/optimizer.cu
        src/mat.h
        src/main.cu
)
TARGET_LINK_LIBRARIES(${APP_NAME} ${OpenCV_LIBS})
TARGET_COMPILE_OPTIONS(${APP_NAME} PRIVATE -std=c++11)
