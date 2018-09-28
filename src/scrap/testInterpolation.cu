// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

//READ ME!!
///This is a test code to confirm the behaviour of interpolation
//A sample test kernel : test3dInterpolation is implemented in energyDerivatives.cu file
//The main example here is to show the uploadAllToTextureMemory funtion usage
//and the how to use tex3D funtion which gives interpolated output in the test3dInterpolationKernel at energyDerivatives.cu file.

#include <iostream>
#include <string>
#include <stdlib.h>

//#include "gamma.cuh"
#include "helper.cuh"

#include "energyDerivatives.cuh"


int main(int argc,char **argv)
{
    
    cudaDeviceSynchronize();  CUDA_CHECK;
    
    // allocate raw input image array
    int w = 80;
    int h = 80;
    int d = 80;
    
    size_t memSize = h*w*d;
    //input Phi
    float *phi = (float*)calloc(memSize, sizeof(float));
    
    //interpolated Phi (the output for this test kernel, which will just be the interpolated values)
    float *phiInterpolated = (float*)calloc(memSize, sizeof(float));
    
    // the u, v and w voxel grid, they specify the interpolation (or the shifts in x, y, and z axis for each voxel)
    float *psiU = (float*)calloc(memSize, sizeof(float));
    float *psiV = (float*)calloc(memSize, sizeof(float));
    float *psiW = (float*)calloc(memSize, sizeof(float));

    //Only few center voxels have 1 in them
    for(int i = 39; i<=40; i++)
        for(int j = 39; j<=40; j++)
            for(int k = 39; k<=40; k++)
                phi[i + j*w + k*w*h] = 1.0;

    //only interpolate the points along the X-axis, so only psiU is non zero.
    //this basically means the grid is shifted by .5 towards right and interpolated
    for(int i = 0; i<80; i++){
        for(int j = 0; j<80; j++){    
            for(int k = 0; k<80; k++){ 
                psiU[i + j*w + k*w*h] = -0.5;
            }
        }
    }

    // only d_psi is require to be allocated here, the input (phi) will be allocate by the uploadAllTextures Function
    float *d_psiU = NULL;
    float *d_psiV = NULL;
    float *d_psiW = NULL;
    float *d_phiInterpolated = NULL;
    cudaMalloc(&d_psiU, memSize*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_psiV, memSize*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_psiW, memSize*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_phiInterpolated, memSize*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(d_psiU, psiU, memSize*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_psiV, psiV, memSize*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_psiW, psiW, memSize*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;


    //----------------------------------------------------------------------------------
    /////Important functions!!
    //This uploads Phi to the Texture Memory.
    uploadToTextureMemory(phi, w, h, d);

    //this is a test kernel, more such can be made in the energyDerivatives.cu/h files.
    test3dInterpolation(d_phiInterpolated, d_psiU, d_psiV, d_psiW, w, h, d);
    cudaDeviceSynchronize();

    //Remember to call this after the texture memory is not required.
    freeTextureMemory();
    //----------------------------------------------------------------------------------


    cudaMemcpy(phiInterpolated, d_phiInterpolated, memSize*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    std::cout << "\nInput: ";
    for(int k = 37; k<=42; k++)
    {
        std::cout << "\n\n\nZ = " << k;
        for(int j = 37; j<=42; j++)
        {
            std::cout << "\n";
            for(int i = 37; i<=42; i++)
            {
                std::cout << "  " << phi[i + j*w + k*w*h];
            }
        }
    }
    
    std::cout << "\n----------Output------------";
    for(int k = 37; k<=42; k++)
    {
        std::cout << "\n\n\nZ = " << k;
        for(int j = 37; j<=42; j++)
        {
            std::cout << "\n";
            for(int i = 37; i<=42; i++)
            {
                std::cout << "  " << phiInterpolated[i + j*w + k*w*h];
            }
        }
    }
    // ### Free allocated arrays
    // TODO free cuda memory of all device arrays
    cudaFree(d_phiInterpolated);
    cudaFree(d_psiU);
    cudaFree(d_psiV);
    cudaFree(d_psiW);
    
    // TODO free memory of all host arrays
    delete[] psiU;
    delete[] psiV;
    delete[] psiW;
    delete[] phi;
    delete[] phiInterpolated;

    return 0;
}



