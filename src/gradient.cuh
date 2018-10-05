#ifndef TUM_GRADIENT_H
#define TUM_GRADIENT_H

#include <iostream>

void computeGradient3DX(float* gradX, const float* gridIn, const size_t w, const size_t h, const size_t d);
void computeGradient3DY(float* gradY, const float* gridIn, const size_t w, const size_t h, const size_t d);
void computeGradient3DZ(float* gradZ, const float* gridIn, const size_t w, const size_t h, const size_t d);

#endif
