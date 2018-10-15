// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Auxiliar operations over a 3D grid (sum, multiplication, mask creation
// and thresholding)
// ########################################################################

#ifndef GRID_OPERATIONS_H
#define GRID_OPERATIONS_H

void computeMask(bool *d_mask, const float *d_phiN,
                 const size_t width, const size_t height, const size_t depth);
                 
void addArray(float* d_arrayA, const float* d_arrayB, const float scalar,
              const size_t width, const size_t height, const size_t depth);

void addWeightedArray(float* arrayOut, float* weightOut, const float* arrayIn1, const float* arrayIn2,
                      const float* weight1, const float* weight2, const size_t width, const size_t height, const size_t depth);

void multiplyArrays(float* arrayOut, const float* arrayIn1, const float* arrayIn2,
                    const size_t width, const size_t height, const size_t depth);

void thresholdArray(float* arrayOut, const float* arrayIn, const float threshold,
                    const size_t width, const size_t height, const size_t depth);

#endif