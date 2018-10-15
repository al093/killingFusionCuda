// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Computes the divergence from the first order partial derivatives
// ########################################################################
#ifndef TUM_DIVERGENCE_H
#define TUM_DIVERGENCE_H

#include <iostream>

void computeDivergence3D(float *divOut, const float *dx, const float *dy, const float *dz, const size_t w, const size_t h, const size_t d);

#endif
