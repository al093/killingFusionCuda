// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "gradient.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"

__global__
void computeGradient3DXKernel(float* gradX, const float* gridIn, const size_t w, const size_t h, const size_t d)
{
    size_t x = threadIdx.x +  (size_t) blockDim.x * blockIdx.x;
	size_t y = threadIdx.y +  (size_t) blockDim.y * blockIdx.y;
	size_t z = threadIdx.z +  (size_t) blockDim.z * blockIdx.z;
	if (x < w && y < h && z < d)
	{
		size_t YZShift = (size_t)w*y + (size_t)w*h*z;
		size_t indVoxel = x + YZShift;
		size_t indNextX = min(x+1, (size_t) w-1) + YZShift;
		size_t indPreviousX = max(x-1, (size_t) 0) + YZShift;
		gradX[indVoxel] = -0.5 * gridIn[indPreviousX] + 0.5 * gridIn[indNextX];
	}
}

__global__
void computeGradient3DYKernel(float* gradY, const float* gridIn, const size_t w, const size_t h, const size_t d)
{
	size_t x = threadIdx.x +  (size_t) blockDim.x * blockIdx.x;
	size_t y = threadIdx.y +  (size_t) blockDim.y * blockIdx.y;
	size_t z = threadIdx.z +  (size_t) blockDim.z * blockIdx.z;
	if (x < w && y < h && z < d)
	{
		size_t XZShift = x + (size_t)w*h*z;
		size_t indVoxel = (size_t)w*y + XZShift;
		size_t indNextY = (size_t)w*min(y+1, (size_t) h-1) + XZShift;
		size_t indPreviousY = (size_t)w*max(y-1, (size_t) 0) + XZShift;
		gradY[indVoxel] = -0.5 * gridIn[indPreviousY] + 0.5 * gridIn[indNextY];
	}
}

__global__
void computeGradient3DZKernel(float* gradZ, const float* gridIn, const size_t w, const size_t h, const size_t d)
{
	size_t x = threadIdx.x +  (size_t) blockDim.x * blockIdx.x;
	size_t y = threadIdx.y +  (size_t) blockDim.y * blockIdx.y;
	size_t z = threadIdx.z +  (size_t) blockDim.z * blockIdx.z;
	if (x < w && y < h && z < d)
	{
		size_t sliceSize = (size_t)w*h;
		size_t XYShift = x + (size_t)w*y;
		size_t indVoxel = sliceSize*z + XYShift;
		size_t indNextZ = sliceSize*min(z+1, (size_t) d-1) + XYShift;
		size_t indPreviousZ = sliceSize*max(z-1, (size_t) 0) + XYShift;
		gradZ[indVoxel] = -0.5 * gridIn[indPreviousZ] + 0.5 * gridIn[indNextZ];
	}
}

void computeGradient3DX(float* gradX, const float* gridIn, const size_t w, const size_t h, const size_t d)
{
    if (!gradX || !gridIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 8, 3);
    dim3 grid = computeGrid3D(block, w, h, d);

    // run cuda kernel
	computeGradient3DXKernel <<<grid, block>>> (gradX, gridIn, w, h, d); 
    // check for errors
	CUDA_CHECK;
}

void computeGradient3DY(float* gradY, const float* gridIn, const size_t w, const size_t h, const size_t d)
{
    if (!gradY || !gridIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 8, 3);
    dim3 grid = computeGrid3D(block, w, h, d);

    // run cuda kernel
	computeGradient3DYKernel <<<grid, block>>> (gradY, gridIn, w, h, d); 
    // check for errors
	CUDA_CHECK;
}

void computeGradient3DZ(float* gradZ, const float* gridIn, const size_t w, const size_t h, const size_t d)
{
    if (!gradZ || !gridIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 8, 3);
    dim3 grid = computeGrid3D(block, w, h, d);

    // run cuda kernel
	computeGradient3DZKernel <<<grid, block>>> (gradZ, gridIn, w, h, d); 
    // check for errors
	CUDA_CHECK;
}