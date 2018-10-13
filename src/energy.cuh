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
                           const bool* d_mask, const size_t width, const size_t height, const size_t depth);

void computeKillingEnergy(float *killingEnergy, const float gamma,
                          const float* d_dux, const float* d_duy, const float* d_duz,
                          const float* d_dvx, const float* d_dvy, const float* d_dvz,
                          const float* d_dwx, const float* d_dwy, const float* d_dwz,
                          const size_t width, const size_t height, const size_t depth);

void computeMask(bool *d_mask, const float *d_phiN,
                 const size_t width, const size_t height, const size_t depth);

#endif
