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
#include "visualization.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"
#include <stdlib.h>
#include <fstream>
#include <string>

void getSlice(float* sliceOut, const float* gridIn, const size_t sliceInd, const size_t dim, const size_t w, const size_t h, const size_t d)
{
    // X dimension
    if (dim == 0)
    {
        for(size_t j = 0; j < h; j++)
        {
            for (size_t k = 0; k < d; k++)
            {
               sliceOut[j + k * h] = gridIn[sliceInd + j * w + k * h*w]; 
            }
        }
    }
    // Y dimension
    else if (dim == 1)
    {
        
        for(size_t i = 0; i < w; i++)
        {
            for (size_t k = 0; k < d; k++)
            {
                sliceOut[i + k * w] = gridIn[i + sliceInd * w + k * h*w];
            }
        }
    }
    // Z dimension
    else
    {
        for(size_t i = 0; i < w*h; i++)
        {
            sliceOut[i] = gridIn[i + (w*h) * sliceInd];
        }
    }
}


void plotSlice(const float* d_array, const size_t sliceInd, const size_t dim, const std::string imageTitle, const size_t posX, const size_t posY, const size_t w, const size_t h, const size_t d)
{
    // Determine required size for the slice
    size_t matH, matW;
    if (dim == 0)
    {
        matH = h;
        matW = d;
    }
    else if (dim == 1)
    {
        matH = w;
        matW = d;
    }
    else
    {
        matH = h;
        matW = w;
    }
    float* h_array = new float[h * w * d];
    float* slice = new float[matH * matW];
    cudaMemcpy(h_array, d_array, (h * w * d) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cv::Mat matSlice(matH, matW, CV_32F);
    // Extact slice
    getSlice(slice, h_array, sliceInd, dim, w, h, d);
    convertLayeredToMat(matSlice, slice);
    // Normalize and resize the slice
    double min, max;
    cv::minMaxLoc(matSlice, &min, &max);
    cv::resize(matSlice, matSlice, cv::Size(), 4, 4);
    showImage(imageTitle, (matSlice - min) / (max - min), posX, posY);
    imwrite( "./bin/result/" + imageTitle + ".jpg", (matSlice - min) / (max - min) * 255);
    // Delete data structures
    delete[] slice;
    delete[] h_array;
}

void plotVectorField(const float* d_u, const float* d_v, const float* d_w, const float* d_sdf, const size_t sliceZval,
                     const std::string sFileU, const std::string sFileV, const std::string sFileW, const std::string sFileSdf,
                     const std::string sPlotName, const int frameNumber,
                     const size_t width, const size_t height, const size_t depth)
{
    
    float* u = new float[width*height*depth];
    float* v = new float[width*height*depth];
    float* w = new float[width*height*depth];
    float* sdf = new float[width*height*depth];

    cudaMemcpy(u, d_u, (width*height*depth) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(v, d_v, (width*height*depth) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(w, d_w, (width*height*depth) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(sdf, d_sdf, (width*height*depth) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    // Save the u , v and w to disk (./bin/result)
    std::ofstream outU(sFileU);
    std::ofstream outV(sFileV);
    std::ofstream outW(sFileW);
    std::ofstream outSdf(sFileSdf);

    // The storing should be done in such a way that python reshape can reshape it correctly
    for (size_t idx = sliceZval*width*height; idx<(sliceZval+1)*width*height; idx++) {
      outU << u[idx] << " ";
      outV << v[idx] << " ";
      outW << w[idx] << " ";
      outSdf << sdf[idx] << " ";
    }

    outU.close();
    outV.close();
    outW.close();
    outSdf.close();

    delete[] u;
    delete[] v;
    delete[] w;
    delete[] sdf;

    // Call python script to plot the quiver plot
    if(system(NULL))
    {
        std::string command = "python ../src/quiverPlot2D.py " +
                               sFileU + " " + sFileV + " " + sFileW + " " +
                               sFileSdf + " " + sPlotName + " " + std::to_string(frameNumber);

        auto retVal = system(command.c_str());
    }
    else
        std::cout << std::endl << "Unable to access the command prompt/ terminal. Will not be able to show deformation quiver plot";
}


void plotEnergy(const float* data, const float* levelSet, const float* killing,
                     const float* total, const size_t arraySize,
                     const std::string sFileData, const std::string sFileLevelSet, const std::string sFileKilling,
                     const std::string sFileTotal, const std::string sPlotName, const int frameNumber,
                     const size_t width, const size_t height, const size_t depth)
{

    // save the energies to disk (./bin/result)
    std::ofstream outData(sFileData);
    std::ofstream outLevelSet(sFileLevelSet);
    std::ofstream outKilling(sFileKilling);
    std::ofstream outTotal(sFileTotal);

    for (size_t idx = 0; idx<=arraySize; ++idx) {
      outData << data[idx] << " ";
      outLevelSet << levelSet[idx] << " ";
      outKilling << killing[idx] << " ";
      outTotal << total[idx] << " ";
    }

    outData.close();
    outLevelSet.close();
    outKilling.close();
    outTotal.close();

    //call python script to plot the quiver plot
    if(system(NULL))
    {
        std::string command = "python ../src/energyPlot.py " +
                               sFileData + " " + sFileLevelSet + " " + sFileKilling + " " +
                               sFileTotal + " " + sPlotName + " " + std::to_string(frameNumber);

        auto retVal = system(command.c_str());
    }
    else
        std::cout<<"\nUnable to access the command prompt/ terminal. Will not be able to show/write Energy Plots ";
}