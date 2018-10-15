// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Computes magnitude of the a 3D vector field
// ########################################################################
#include "magnitude.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"

__global__
void computeMagnitudeKernel(float* magOut, const float* gridInX, const float* gridInY, const float* gridInZ, int w, int h, int d)
{
    int x = threadIdx.x +  (size_t) blockDim.x * blockIdx.x;
	int y = threadIdx.y +  (size_t) blockDim.y * blockIdx.y;
	int z = threadIdx.z +  (size_t) blockDim.z * blockIdx.z;
	if (x < w && y < h && z < d)
	{
		size_t sliceSize = (size_t)w*h;
		size_t indVoxel = x + (size_t)w*y + sliceSize*z;
		
		magOut[indVoxel] = sqrt(gridInX[indVoxel] * gridInX[indVoxel] + gridInY[indVoxel] * gridInY[indVoxel] + gridInZ[indVoxel] * gridInZ[indVoxel]);
	}
}

void computeMagnitude(float* magOut, const float* gridInX, const float* gridInY, const float* gridInZ, int w, int h, int d)
{
    if (!magOut || !gridInX || !gridInY || !gridInZ)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 8, 3);
    dim3 grid = computeGrid3D(block, w, h, d);

    // run cuda kernel
    // execute kernel for convolution using global memory
	computeMagnitudeKernel <<<grid, block>>> (magOut, gridInX, gridInY, gridInZ, w, h, d); 
    // check for errors
	CUDA_CHECK;
}
