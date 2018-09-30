// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// defines the host functions which call the device functions kernels for calculating the energy
// ########################################################################

#ifndef ENERGY_H
#define ENERGY_H

void computeDataEnergy(float *dataEnergy, const float *d_phiNDeformed, const float *d_phiGlobal,
                       const size_t width, const size_t height, const size_t depth);

void computeLevelSetEnergy(float *levelSetEnergy,
                           const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ,
                           const size_t width, const size_t height, const size_t depth);

//TODO to be implemented
void computeMotionRegularizerEnergy();

#endif