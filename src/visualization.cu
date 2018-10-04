// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "visualization.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"
#include <stdlib.h>
#include <fstream>
#include <string>

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

    delete[] slice;
    delete[] h_array;
}

void plotDeformation(const float* d_u, const float* d_v, const float* d_w, const size_t sliceZval, const size_t width, const size_t height, const size_t depth)
{
    
    float* u = new float[width*height*depth];
    float* v = new float[width*height*depth];
    float* w = new float[width*height*depth];

    cudaMemcpy(u, d_u, (width*height*depth) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(v, d_v, (width*height*depth) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(w, d_w, (width*height*depth) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    // save the u , v and w to disk (./bin/result)
    std::ofstream outU("./bin/result/u.txt");
    std::ofstream outV("./bin/result/v.txt");
    std::ofstream outW("./bin/result/w.txt");

    //the storing should be done in such a way that python reshape can reshape it correctly
        for (size_t idx = sliceZval*width*height; idx<(sliceZval+1)*width*height; ++idx) {
            outU << u[idx] << " ";
            outV << v[idx] << " ";
            outW << w[idx] << " ";
        }

    outU.close();
    outV.close();
    outW.close();

    delete[] u;
    delete[] v;
    delete[] w;

    //call python script to plot the quiver plot
    if(system(NULL))
    {
        std::string command = "python ../src/quiverPlot2D.py";
        system(command.c_str());
    }
    else
        std::cout<<"\nUnable to access the command prompt/ terminal. Will not be able to show deformation quiver plot";
}
