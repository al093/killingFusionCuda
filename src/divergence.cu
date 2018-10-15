// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Computes divergence operator from the partial derivatives
// ########################################################################
#include "divergence.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeDivergenceKernel(float *divOut, const float *dx, const float *dy, const float *dz, const size_t w, const size_t h, const size_t d)
{
    // compute divergence
	int x = threadIdx.x +  (size_t) blockDim.x * blockIdx.x;
	int y = threadIdx.y +  (size_t) blockDim.y * blockIdx.y;
	int z = threadIdx.z +  (size_t) blockDim.z * blockIdx.z;
	if (x < w && y < h && z < d)
	{
		size_t sliceSize = (size_t)w*h;
		size_t ind = x + (size_t)w*y + sliceSize*z;
		divOut[ind] = dx[ind] + dy[ind] + dz[ind];
	}
}

void computeDivergence3D(float *divOut, const float *dx, const float *dy, const float *dz, const size_t w, const size_t h, const size_t d)
{
    // calculate block and grid size
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid3D(block, w, h, d);

    // run cuda kernel
    computeDivergenceKernel <<<grid, block>>> (divOut, dx, dy, dz, w, h, d); 
    // check for errors
	CUDA_CHECK;
}
