// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_REDUCTION_FINDMAX_H
#define TUM_REDUCTION_FINDMAX_H

#include "cublas_v2.h"

void findAbsMax(cublasHandle_t handle, float* maxVal, const float * d_array, size_t width, size_t height, size_t depth);

#endif
