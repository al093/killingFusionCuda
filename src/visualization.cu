// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "visualization.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


void getSlice(float* sliceOut, const float* gridIn, const size_t sliceInd, const size_t w, const size_t h)
{
  for(int i = 0; i < w*h; i++)
  {
    sliceOut[i] = gridIn[i + (w*h) * sliceInd];
  }
}

void plotSlice(const float* d_array, const size_t z, const std::string imageTitle, const size_t posX, const size_t posY, const size_t w, const size_t h, const size_t d)
{
    float* h_array = new float[h * w * d];
    float* slice = new float[h * w];
    cudaMemcpy(h_array, d_array, (h * w * d) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
   /* int sizes[] = {(int) w, (int) h, (int) d};
    cv::Mat mat3D(3, sizes, CV_32FC1, cv::Scalar(0));*/
    cv::Mat matSlice(h, w, CV_32F);
    getSlice(slice, h_array, z, w, h);
    convertLayeredToMat(matSlice, slice);
    //getSliceFromMat(mat3D, z, matSlice);
    // Normalize the slice
    double min, max;
    cv::minMaxLoc(matSlice, &min, &max);
    cv::resize(matSlice, matSlice, cv::Size(), 4, 4);
    showImage(imageTitle, (matSlice - min) / (max - min), posX, posY);
}