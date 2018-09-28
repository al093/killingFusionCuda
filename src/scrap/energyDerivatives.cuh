// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Cuda Functions to calculate the energy derivatives and also helper function for using 3D texture memory.
// ########################################################################

#ifndef ENERGY_DERIVATIVES_H
#define ENERGY_DERIVATIVES_H

#include <iostream>

void uploadToTextureMemory(float* h_phi, int w, int h, int d);

void freeTextureMemory();

void test3dInterpolation(float *d_phiInterpolated, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth);

#endif