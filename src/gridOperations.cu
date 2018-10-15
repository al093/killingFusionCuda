// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Auxiliar operations over a 3D grid (sum, multiplication, mask creation
// and thresholding)
// ########################################################################
#include "gridOperations.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"

#include "cublas_v2.h"

__global__
void computeMaskKernel(bool *d_mask, const float *d_phiN,
                 const size_t width, const size_t height, const size_t depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x < width && y < height && z < depth)
    {
        size_t idx = x + y*width + z*width*height;
        if(d_phiN[idx] < 1.0 && d_phiN[idx] > -1.0)
        {
            d_mask[idx] = true;
        }
        else
        {
            d_mask[idx] = false;
        }
    }
}

__global__
void addArrayKernel(float* d_arrayA, const float* d_arrayB, const float scalar, const size_t width, const size_t height, const size_t depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x < width && y < height && z < depth)
    {
        size_t idx = x + y*width + z*width*height;
        d_arrayA[idx] += scalar*d_arrayB[idx];
    }
}

__global__
void addWeightedArrayKernel(float* arrayOut, float* weightOut, const float* arrayIn1, const float* arrayIn2, const float* weight1, const float* weight2, const size_t width, const size_t height, const size_t depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x < width && y < height && z < depth)
    {
        size_t idx = x + y*width + z*width*height;
        float sumWeights = weight1[idx] + weight2[idx];
        if (arrayIn1[idx] == -1 && arrayIn2[idx] == -1)
        {
            arrayOut[idx] = -1;
        }
        else
        {
            arrayOut[idx] = (weight1[idx]*arrayIn1[idx] + weight2[idx]*arrayIn2[idx]) / sumWeights;
        }
        weightOut[idx] = sumWeights;
    }

}

__global__
void multiplyArraysKernel(float* arrayOut, const float* arrayIn1, const float* arrayIn2, const size_t width, const size_t height, const size_t depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x < width && y < height && z < depth)
    {
        size_t idx = x + y*width + z*width*height;
        arrayOut[idx] = arrayIn1[idx] * arrayIn2[idx];
    }

}

__global__
void thresholdArrayKernel(float* arrayOut, const float* arrayIn, float threshold, const size_t width, const size_t height, const size_t depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x < width && y < height && z < depth)
    {
        size_t idx = x + y*width + z*width*height;
        if (arrayIn[idx] < threshold)
        {
            arrayOut[idx] = 0.0f;
        }
    }

}

void computeMask(bool *d_mask, const float *d_phiN,
                 const size_t width, const size_t height, const size_t depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    computeMaskKernel <<<grid, blockSize>>> (d_mask, d_phiN, width, height, depth);

    CUDA_CHECK;
}

void addArray(float* d_arrayA, const float* d_arrayB, const float scalar,
              const size_t width, const size_t height, const size_t depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    addArrayKernel <<<grid, blockSize>>> (d_arrayA, d_arrayB, scalar,
                                          width, height, depth);
}

void addWeightedArray(float* arrayOut, float* weightOut, const float* arrayIn1, const float* arrayIn2, const float* weight1, const float* weight2, const size_t width, const size_t height, const size_t depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    addWeightedArrayKernel <<<grid, blockSize>>> (arrayOut, weightOut, arrayIn1, arrayIn2, weight1, weight2, width, height, depth);
}

void multiplyArrays(float* arrayOut, const float* arrayIn1, const float* arrayIn2, const size_t width, const size_t height, const size_t depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    multiplyArraysKernel <<<grid, blockSize>>> (arrayOut, arrayIn1, arrayIn2, width, height, depth);
}

void thresholdArray(float* arrayOut, const float* arrayIn, const float threshold, const size_t width, const size_t height, const size_t depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    thresholdArrayKernel <<<grid, blockSize>>> (arrayOut, arrayIn, threshold, width, height, depth);
}
