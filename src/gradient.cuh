// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Computes the 3D gradient via central differences
// ########################################################################
#ifndef TUM_GRADIENT_H
#define TUM_GRADIENT_H

#include <iostream>

void computeGradient3DX(float* gradX, const float* gridIn, const size_t w, const size_t h, const size_t d);
void computeGradient3DY(float* gradY, const float* gridIn, const size_t w, const size_t h, const size_t d);
void computeGradient3DZ(float* gradZ, const float* gridIn, const size_t w, const size_t h, const size_t d);

#endif
