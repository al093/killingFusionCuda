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
void computeLevelSetDerivativeKernel(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW, 
                               const float *d_hessPhiXX, const float *d_hessPhiXY, const float *d_hessPhiXZ,
                               const float *d_hessPhiYY, const float *d_hessPhiYZ, const float *d_hessPhiZZ,
                               const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ, const float ws,
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
        
        float scalar = ws*(gradNorm - 1.0)/(gradNorm+0.00001);
        
        d_dEdataU[idx] += scalar*(d_hessPhiXX[idx]*d_gradPhiNDeformedX[idx] + d_hessPhiXY[idx]*d_gradPhiNDeformedY[idx] + d_hessPhiXZ[idx]*d_gradPhiNDeformedZ[idx]);
        d_dEdataV[idx] += scalar*(d_hessPhiXY[idx]*d_gradPhiNDeformedX[idx] + d_hessPhiYY[idx]*d_gradPhiNDeformedY[idx] + d_hessPhiYZ[idx]*d_gradPhiNDeformedZ[idx]);
        d_dEdataW[idx] += scalar*(d_hessPhiXZ[idx]*d_gradPhiNDeformedX[idx] + d_hessPhiYZ[idx]*d_gradPhiNDeformedY[idx] + d_hessPhiZZ[idx]*d_gradPhiNDeformedZ[idx]);
    }
}

__global__
void computeMotionRegularizerDerivativeKernel(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW,
                                              const float *d_lapU, const float *d_lapV, const float *d_lapW,
                                              const float *d_divX, const float *d_divY, const float *d_divZ,
                                              const float wk, const float gamma,
                                              const size_t width, const size_t height, const size_t depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x<width && y<height && z<depth)
    {
        size_t idx = x + y*width + z*width*height;
        d_dEdataU[idx] += -2.0*wk*d_lapU[idx] -2.0*wk*gamma*d_divX[idx];
        d_dEdataV[idx] += -2.0*wk*d_lapV[idx] -2.0*wk*gamma*d_divY[idx];
        d_dEdataW[idx] += -2.0*wk*d_lapW[idx] -2.0*wk*gamma*d_divZ[idx];
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
		float sumWeights = max(weight1[idx], 0.0f) + max(weight2[idx], 0.0f);
        arrayOut[idx] = (max(weight1[idx], 0.0f)*arrayIn1[idx] + max(weight2[idx], 0.0f)*arrayIn2[idx]) / (sumWeights + 0.00001f);
		weightOut[idx] = sumWeights;
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
                               const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ, const float wk,
                               const size_t width, const size_t height, const size_t depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);
    
    computeLevelSetDerivativeKernel <<<grid, blockSize>>> (d_dEdataU, d_dEdataV, d_dEdataW, 
                                                           d_hessPhiXX, d_hessPhiXY, d_hessPhiXZ,
                                                           d_hessPhiYY, d_hessPhiYZ, d_hessPhiZZ,
                                                           d_gradPhiNDeformedX, d_gradPhiNDeformedY, d_gradPhiNDeformedZ, wk,
                                                           width, height, depth);
}

void computeMotionRegularizerDerivative(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW,
                                        const float *d_lapU, const float *d_lapV, const float *d_lapW,
                                        const float *d_divX, const float *d_divY, const float *d_divZ,
                                        const float ws, const float gamma,
                                        const size_t width, const size_t height, const size_t depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);
    
    computeMotionRegularizerDerivativeKernel <<<grid, blockSize>>> (d_dEdataU, d_dEdataV, d_dEdataW,
                                                                    d_lapU, d_lapV, d_lapW,
                                                                    d_divX, d_divY, d_divZ,
                                                                    ws, gamma,
                                                                    width, height, depth);
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
