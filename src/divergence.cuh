// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_DIVERGENCE_H
#define TUM_DIVERGENCE_H

#include <iostream>

void computeDivergence3DCuda(float *divOut, const float *dx, const float *dy, const float *dz, const size_t w, const size_t h, const size_t d);

#endif
