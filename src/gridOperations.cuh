// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// defines the host functions which call the device functions kernels 
// for Grid operations like add and reduce and mask creation
// ########################################################################

#ifndef GRID_OPERATIONS_H
#define GRID_OPERATIONS_H

void computeMask(bool *d_mask, const float *d_phiN,
                 const size_t width, const size_t height, const size_t depth);
                 

void addWeightedArray(float* arrayOut, float* weightOut, const float* arrayIn1, const float* arrayIn2,
                      const float* weight1, const float* weight2, const size_t width, const size_t height, const size_t depth);

void multiplyArrays(float* arrayOut, const float* arrayIn1, const float* arrayIn2,
                    const size_t width, const size_t height, const size_t depth);

void thresholdArray(float* arrayOut, const float* arrayIn, const float threshold,
                    const size_t width, const size_t height, const size_t depth);

#endif