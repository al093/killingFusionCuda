// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Visualization for plotting slices, vector fields and energy
// ########################################################################
#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <iostream>

void getSlice(float* sliceOut, const float* gridIn, const size_t sliceInd, const size_t dim, const size_t w, const size_t h, const size_t d);

void plotSlice(const float* d_array, const size_t sliceInd, const size_t dim, const std::string imageTitle, const size_t posX, const size_t posY, const size_t w, const size_t h, const size_t d);

void plotVectorField(const float* d_u, const float* d_v, const float* d_w,
                     const float* d_sdf, const size_t sliceZval,
                     const std::string sFileU, const std::string sFileV, const std::string sFileW,
                     const std::string sFileWeights, const std::string sPlotName, const int frameNumber,
                     const size_t width, const size_t height, const size_t depth);
                     
void plotEnergy(const float* data, const float* levelSet, const float* killing,
                     const float* total, const size_t arraySize,
                     const std::string sFileData, const std::string sFileLevelSet, const std::string sFileKilling,
                     const std::string sFileTotal, const std::string sPlotName, const int frameNumber,
                     const size_t width, const size_t height, const size_t depth);

#endif