// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Cuda Functions to calculate the energy derivatives and also helper function for using 3D texture memory.
// ########################################################################

#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include <cuda_runtime.h>
#include <iostream>

/*
struct deviceStruct
{
	__device__ void interpolate3DKernel(float *d_outputValues, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth)
	{
		int x = threadIdx.x + blockIdx.x*blockDim.x;
	    int y = threadIdx.y + blockIdx.y*blockDim.y;
	    int z = threadIdx.z + blockIdx.z*blockDim.z;

	    float fx = x;
	    float fy = y;
	    float fz = z;

	    if(x<width && y<height && z<depth)
	    {
	        size_t idx = x + y*width + z*width*height;
	        //Remember!! to always add 0.5, the voxels have actual values at their centers, and the size of a voxel is 1x1x1, so need to add .5, .5, .5 for center 
	        d_outputValues[idx] = tex3D(phi, fx + d_u[idx] + 0.5, fy + d_v[idx] + 0.5, fz + d_w[idx] + 0.5);
	    }
	}
};

*/
class Interpolator
{
	public:
		Interpolator(float* h_phi, int width, int height, int depth);
		~Interpolator();
		
		//cudaTextureObject_t tex=0;
		cudaTextureObject_t texPhil;
		cudaArray *cuArray_phi1;
		void interpolate3D(float *d_phiInterpolated, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth);

	protected:

		void uploadToTextureMemory(float* h_phi, int w, int h, int d);
		void freeTextureMemory();
	//__global__
	//void Interpolator::interpolate3DKernel(float *d_outputValues, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth);
		//_device__ void interpolate3DKernel(float *d_outputValues, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth);
		//deviceStruct interpolateStruct;
		//cudaTextureObject_t phi;
};
#endif

//https://devtalk.nvidia.com/default/topic/802257/working-with-cuda-and-class-methods/