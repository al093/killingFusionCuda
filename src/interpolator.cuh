// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Cuda Functions to calculate the energy derivatives and also helper function for using 3D texture memory.
// ########################################################################

#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include <cuda_runtime.h>
#include <iostream>

class Interpolator
{
	public:
		Interpolator(float* h_phi, int width, int height, int depth);
		~Interpolator();

		cudaTextureObject_t texPhil;
		cudaArray *cuArray_phi1;
		void interpolate3D(float *d_phiInterpolated, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth);

	protected:

		void uploadToTextureMemory(float* h_phi, int w, int h, int d);
		void freeTextureMemory();
};
#endif

//https://devtalk.nvidia.com/default/topic/802257/working-with-cuda-and-class-methods/
