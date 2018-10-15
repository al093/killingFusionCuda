// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Computes magnitude of the a 3D vector field
// ########################################################################
#ifndef TUM_MAGNITUDE_H
#define TUM_MAGNITUDE_H

#include <iostream>

void computeMagnitude(float* magOut, const float* gridInX, const float* gridInY, const float* gridInZ, int w, int h, int d);

#endif
