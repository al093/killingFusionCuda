// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "reduction.cuh"

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include "helper.cuh"

#include "cublas_v2.h"

void findAbsMax(float* maxVal, const float * d_array, size_t width, size_t height, size_t depth)
{
    // create cublas handle
    cublasHandle_t handle;
    // TODO maybe it can be initialized in optimize class just once to improve timing
    cublasCreate(&handle);
    int maxIdx = 0;
    //if cublas was able to find the max value, copy the value from the index in the gpu memory to host memory
    if(cublasIsamax(handle, width*height*depth, d_array, 1, &maxIdx) == CUBLAS_STATUS_SUCCESS)
    {
        //cublas has fortran like 1 based indexing
        cudaMemcpy(maxVal, d_array+maxIdx-1, sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        //std::cout<<"\nMax value found was: "<< *maxVal;
        *maxVal = abs(*maxVal);
    }
    else
        std::cout<<"\n[ERROR] CUBLAS encountered an error while calculating the max value";
    
    cublasDestroy(handle);
}
