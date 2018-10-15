// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Perform reduction  by finding the maximum absolute value in a grid
// ########################################################################
#include "reduction.cuh"

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include "helper.cuh"

#include "cublas_v2.h"

void findAbsMax(cublasHandle_t handle, float* maxVal, const float * d_array, size_t width, size_t height, size_t depth)
{
    int maxIdx = 0;
    // If cublas was able to find the max value, copy the value from the index in the gpu memory to host memory
    if(cublasIsamax(handle, width*height*depth, d_array, 1, &maxIdx) == CUBLAS_STATUS_SUCCESS)
    {
        // cublas has fortran like 1 based indexing
        cudaMemcpy(maxVal, d_array+maxIdx-1, sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        *maxVal = abs(*maxVal);
    }
    else
    {
        std::cout << std::endl <<"[ERROR] CUBLAS encountered an error while calculating the max value";
    }
}