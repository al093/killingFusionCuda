// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// defines the host functions which call the device functions kernels for calculating the energy derivatives
// ########################################################################

#ifndef ENERGY_DERIVATIVES_2_H
#define ENERGY_DERIVATIVES_2_H

void computeDataTermDerivative(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW, 
                               const float *d_phiNDeformed, const float *d_phiGlobal,
                               const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ,
                               const size_t width, const size_t height, const size_t depth);


void computeLevelSetDerivative(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW, 
                               const float *d_hessPhiXX, const float *d_hessPhiXY, const float *d_hessPhiXZ,
                               const float *d_hessPhiYY, const float *d_hessPhiYZ, const float *d_hessPhiZZ,
                               const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ,
                               const size_t width, const size_t height, const size_t depth);

#endif