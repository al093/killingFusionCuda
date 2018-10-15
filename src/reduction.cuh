// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Perform reduction  by finding the maximum absolute value in a grid
// ########################################################################
#ifndef TUM_REDUCTION_FINDMAX_H
#define TUM_REDUCTION_FINDMAX_H

#include "cublas_v2.h"

void findAbsMax(cublasHandle_t handle, float* maxVal, const float * d_array, size_t width, size_t height, size_t depth);

#endif
