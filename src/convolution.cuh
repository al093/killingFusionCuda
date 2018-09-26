// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_CONVOLUTION_H
#define TUM_CONVOLUTION_H

#include <iostream>

void createConvolutionKernel(float *kernel, int kradius, float sigma);

void computeConvolution(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc);

void computeConvolutionTextureMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc);
void computeConvolutionSharedMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc);
void computeConvolutionSharedConstantMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc);
void computeConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int c);
void computeConvolution3D(float *gridOut, const float *gridIn, const float *kernel, int kradius, int w, int h, int d);

#endif
