// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// defines the host functions which call the device functions kernels for calculating the energy
// ########################################################################

#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <iostream>

void getSlice(float* sliceOut, const float* gridIn, const size_t sliceInd, const size_t w, const size_t h);

void plotSlice(const float* d_array, const size_t z, const std::string imageTitle, const size_t posX, const size_t posY, const size_t w, const size_t h, const size_t d);

void plotDeformation(const float* d_u, const float* d_v, const float* d_w, const size_t sliceZval, const size_t width, const size_t height, const size_t depth);

#endif
