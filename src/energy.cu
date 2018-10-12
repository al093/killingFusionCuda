// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "energy.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"

#include "cublas_v2.h"

__global__
void computeDataEnergyKernel(float *d_dataEnergyArray, 
                             const float *d_phiNDeformed, const float *d_phiGlobal,
                             const size_t width, const size_t height, const size_t depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x < width && y < height && z < depth)
    {
        size_t idx = x + y*width + z*width*height;
        float diff = d_phiNDeformed[idx] - d_phiGlobal[idx];
        d_dataEnergyArray[idx] = diff * diff;
    }
}

__global__
void computeLevelSetEnergyKernel(float *d_levelSetEnergyArray,
                                 const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ,
                                 const size_t width, const size_t height, const size_t depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x < width && y < height && z < depth)
    {
        size_t idx = x + y*width + z*width*height;
        float norm = sqrt(d_gradPhiNDeformedX[idx]*d_gradPhiNDeformedX[idx] + d_gradPhiNDeformedY[idx]*d_gradPhiNDeformedY[idx] + d_gradPhiNDeformedZ[idx]*d_gradPhiNDeformedZ[idx]);
        d_levelSetEnergyArray[idx] = 0.5 * (norm - 1) * (norm - 1);
    }
}

__global__
void computeKillingEnergyKernel(float *d_killingEnergyArray, const float gamma,
                                const float* d_dux, const float* d_duy, const float* d_duz,
                                const float* d_dvx, const float* d_dvy, const float* d_dvz,
                                const float* d_dwx, const float* d_dwy, const float* d_dwz,
                                const size_t width, const size_t height, const size_t depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x < width && y < height && z < depth)
    {
        size_t idx = x + y*width + z*width*height;
        d_killingEnergyArray[idx] = (1.0+gamma)*(d_dux[idx]*d_dux[idx] + d_dvy[idx]*d_dvy[idx] + d_dwz[idx]*d_dwz[idx])+
                                      d_duy[idx]*d_duy[idx] + d_duz[idx]*d_duz[idx] +
                                      d_dvx[idx]*d_dvx[idx] + d_dvz[idx]*d_dvz[idx] +
                                      d_dwx[idx]*d_dwx[idx] + d_dwy[idx]*d_dwy[idx] +
                                      2.0*gamma*(d_duy[idx]*d_dvx[idx] + d_duz[idx]*d_dwx[idx] + d_dwy[idx]*d_dvz[idx]);
    }
}

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
    }
}

void computeDataEnergy(float *dataEnergy, const float *d_phiNDeformed, const float *d_phiGlobal,
                       const size_t width, const size_t height, const size_t depth)
{
    float* d_dataEnergyArray;
    cudaMalloc(&d_dataEnergyArray, (width * height * depth) * sizeof(float)); CUDA_CHECK;

    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    computeDataEnergyKernel <<<grid, blockSize>>> (d_dataEnergyArray, 
                                                   d_phiNDeformed, d_phiGlobal,
                                                   width, height, depth);
    CUDA_CHECK;

    // Create cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Calculate the sum of the energy
    cublasSasum(handle, width*height*depth, d_dataEnergyArray, sizeof(float), dataEnergy);
	  *dataEnergy = 0.5 * *dataEnergy;
    cublasDestroy(handle);
    cudaFree(d_dataEnergyArray);
}

void computeLevelSetEnergy(float *levelSetEnergy,
                           const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ,
                           const size_t width, const size_t height, const size_t depth)
{
    float* d_levelSetEnergyArray;
    cudaMalloc(&d_levelSetEnergyArray, (width * height * depth) * sizeof(float)); CUDA_CHECK;

    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    computeLevelSetEnergyKernel <<<grid, blockSize>>> (d_levelSetEnergyArray, 
                                                       d_gradPhiNDeformedX, d_gradPhiNDeformedY, d_gradPhiNDeformedZ,
                                                       width, height, depth);
    CUDA_CHECK;

    // Create cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Calculate the sum of the energy
    cublasSasum(handle, width*height*depth, d_levelSetEnergyArray, sizeof(float), levelSetEnergy);

    // Free cuda memory and cublas handle
    cublasDestroy(handle);
    cudaFree(d_levelSetEnergyArray);
}



void computeKillingEnergy(float *killingEnergy, const float gamma,
                          const float* d_dux, const float* d_duy, const float* d_duz,
                          const float* d_dvx, const float* d_dvy, const float* d_dvz,
                          const float* d_dwx, const float* d_dwy, const float* d_dwz,
                          const size_t width, const size_t height, const size_t depth)
{
    float* d_killingEnergyArray;
    cudaMalloc(&d_killingEnergyArray, (width * height * depth) * sizeof(float)); CUDA_CHECK;

    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    computeKillingEnergyKernel <<<grid, blockSize>>> (d_killingEnergyArray, gamma,
                                                      d_dux, d_duy, d_duz,
                                                      d_dvx, d_dvy, d_dvz,
                                                      d_dwx, d_dwy, d_dwz,
                                                      width, height, depth);
    CUDA_CHECK;

    // Create cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Calculate the sum of the energy
    cublasSasum(handle, width*height*depth, d_killingEnergyArray, sizeof(float), killingEnergy);

    // Free cuda memory and cublas handle
    cublasDestroy(handle);
    cudaFree(d_killingEnergyArray);
}

void computeMask(bool *d_mask, const float *d_phiN,
                 const size_t width, const size_t height, const size_t depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    computeMaskKernel <<<grid, blockSize>>> (d_mask, d_phiN, width, height, depth);

    CUDA_CHECK;
}
