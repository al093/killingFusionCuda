// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "energyDerivatives.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"

__global__
void computeDataTermDerivativeKernel(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW, 
                                    const float *d_phiNDeformed, const float *d_phiGlobal,
                                    const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ,
                                    const size_t width, const size_t height, const size_t depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x<width && y<height && z<depth)
    {
        size_t idx = x + y*width + z*width*height;

        float scalar = (d_phiNDeformed[idx] - d_phiGlobal[idx]);

        d_dEdataU[idx] += scalar*d_gradPhiNDeformedX[idx];
        d_dEdataV[idx] += scalar*d_gradPhiNDeformedY[idx];
        d_dEdataW[idx] += scalar*d_gradPhiNDeformedZ[idx];
    }
}

__global__
void computeDataTermDerivativeKernel(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW, 
                               const float *d_hessPhiXX, const float *d_hessPhiXY, const float *d_hessPhiXZ,
                               const float *d_hessPhiYY, const float *d_hessPhiYZ, const float *d_hessPhiZZ,
                               const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ,
                               const size_t width, const size_t height, const size_t depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x<width && y<height && z<depth)
    {
        size_t idx = x + y*width + z*width*height;

        float gradNorm = pow(d_gradPhiNDeformedX[idx], 2) + pow(d_gradPhiNDeformedY[idx], 2) + pow(d_gradPhiNDeformedZ[idx], 2);
        gradNorm = sqrt(gradNorm);
        
        float scalar = (gradNorm - 1.0)/(gradNorm+0.00001);
        
        d_dEdataU[idx] += scalar*(d_hessPhiXX[idx]*d_gradPhiNDeformedX[idx] + d_hessPhiXY[idx]*d_gradPhiNDeformedY[idx] + d_hessPhiXZ[idx]*d_gradPhiNDeformedZ[idx]);
        d_dEdataV[idx] += scalar*(d_hessPhiXY[idx]*d_gradPhiNDeformedX[idx] + d_hessPhiYY[idx]*d_gradPhiNDeformedY[idx] + d_hessPhiYZ[idx]*d_gradPhiNDeformedZ[idx]);
        d_dEdataW[idx] += scalar*(d_hessPhiXZ[idx]*d_gradPhiNDeformedX[idx] + d_hessPhiYZ[idx]*d_gradPhiNDeformedY[idx] + d_hessPhiZZ[idx]*d_gradPhiNDeformedZ[idx]);
    }
}


void computeDataTermDerivative(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW, 
                               const float *d_phiNDeformed, const float *d_phiGlobal,
                               const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ,
                               const size_t width, const size_t height, const size_t depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    computeDataTermDerivativeKernel <<<grid, blockSize>>> (d_dEdataU, d_dEdataV, d_dEdataW, 
                                                     d_phiNDeformed, d_phiGlobal,
                                                     d_gradPhiNDeformedX, d_gradPhiNDeformedY, d_gradPhiNDeformedZ,
                                                     width, height, depth);
}


void computeLevelSetDerivative(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW, 
                               const float *d_hessPhiXX, const float *d_hessPhiXY, const float *d_hessPhiXZ,
                               const float *d_hessPhiYY, const float *d_hessPhiYZ, const float *d_hessPhiZZ,
                               const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ,
                               const size_t width, const size_t height, const size_t depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);
    
    computeDataTermDerivativeKernel <<<grid, blockSize>>> (d_dEdataU, d_dEdataV, d_dEdataW, 
                                                           d_hessPhiXX, d_hessPhiXY, d_hessPhiXZ,
                                                           d_hessPhiYY, d_hessPhiYZ, d_hessPhiZZ,
                                                           d_gradPhiNDeformedX, d_gradPhiNDeformedY, d_gradPhiNDeformedZ,
                                                           width, height, depth);
}