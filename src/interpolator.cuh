// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Interpolator object to interpolate a base grid using a given vector field
// ########################################################################

#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include <cuda_runtime.h>
#include <iostream>

class Interpolator
{
	public:
		Interpolator(float* h_grid, int width, int height, int depth);
		~Interpolator();

		cudaTextureObject_t texGrid;
		cudaArray *cuArray_grid;
		void interpolate3D(float *d_gridInterpolated, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth);

	protected:

		void uploadToTextureMemory(float* d_grid, int w, int h, int d);
		void freeTextureMemory();
};
#endif