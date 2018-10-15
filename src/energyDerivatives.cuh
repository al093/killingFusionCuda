// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Computes the energy derivatives for Killing fusion
// ########################################################################

#ifndef ENERGY_DERIVATIVES_2_H
#define ENERGY_DERIVATIVES_2_H

void computeDataTermDerivative(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW, 
                               const float *d_phiNDeformed, const float *d_phiGlobal,
                               const bool *d_mask,
                               const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ,
                               const size_t width, const size_t height, const size_t depth);


void computeLevelSetDerivative(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW, 
                               const float *d_hessPhiXX, const float *d_hessPhiXY, const float *d_hessPhiXZ,
                               const float *d_hessPhiYY, const float *d_hessPhiYZ, const float *d_hessPhiZZ,
                               const float *d_gradPhiNDeformedX, const float *d_gradPhiNDeformedY, const float *d_gradPhiNDeformedZ,
                               const bool *d_mask, const float ws, const float tsdfGradScale, const float voxelSize,
                               const size_t width, const size_t height, const size_t depth);

void computeMotionRegularizerDerivative(float *d_dEdataU, float *d_dEdataV, float *d_dEdataW,
                                        const float *d_lapU, const float *d_lapV, const float *d_lapW,
                                        const float *d_divX, const float *d_divY, const float *d_divZ,
                                        const bool *d_mask, const float wk, const float gamma,
                                        const size_t width, const size_t height, const size_t depth);
#endif
