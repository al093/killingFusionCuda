///Cuda Functions to calculate the energy derivatives and also helper function for using 3D texture memory.
#include "energyDerivatives.cuh"
#include <iostream>
#include <cuda_runtime.h>

#include <math.h>
#include "helper.cuh"


//create multiple global Texture objects here
texture<float, cudaTextureType3D, cudaReadModeElementType> phi;
cudaArray *cuArray_phi;

__global__
void test3dInterpolationKernel(float *d_outputValues, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    float fx = x;
    float fy = y;
    float fz = z;

    if(x<width && y<height && z<depth)
    {
        size_t idx = x + y*width + z*width*height;
        //Remember!! to always add 0.5, the voxels have actual values at their centers, and the size of a voxel is 1x1x1, so need to add .5, .5, .5 for center 
        d_outputValues[idx] = tex3D(phi, fx + d_u[idx] + 0.5, fy + d_v[idx] + 0.5, fz + d_w[idx] + 0.5);
    }
}


///use this method to bind Texture memory to Cuda array.
///TODO currently testing for only one 3D voxel grid
void uploadToTextureMemory(float* h_phi, int w, int h, int d)
{
    //define channel format descriptor
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    //set the grid size
    cudaExtent extent;
    extent.width = w;
    extent.height = h;
    extent.depth = d;

    //define and allocate 3D cuda array
    cudaMalloc3DArray(&cuArray_phi, &desc, extent);
    CUDA_CHECK;

    //copy from host memory to CudaArray in device memory
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)h_phi, extent.width*sizeof(float), extent.width, extent.height);
    copyParams.dstArray = cuArray_phi;
    copyParams.kind = cudaMemcpyHostToDevice;
    copyParams.extent = extent;
    cudaMemcpy3D(&copyParams);
    CUDA_CHECK;

    //set texture parameters
    phi.normalized = false;                      // access with normalized phiture coordinates
    phi.filterMode = cudaFilterModeLinear;      // linear interpolation
    phi.addressMode[0] = cudaAddressModeClamp;   // wrap phiture coordinates
    phi.addressMode[1] = cudaAddressModeClamp;
    phi.addressMode[2] = cudaAddressModeClamp;

    // bind array to 3D texture
    cudaBindTextureToArray(phi, cuArray_phi, desc);
    CUDA_CHECK;
}

void freeTextureMemory()
{
    cudaUnbindTexture(phi);
    cudaFreeArray(cuArray_phi);
}


void test3dInterpolation(float *d_phiInterpolated, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth)
{
    dim3 blockSize(width/10, height/10, depth/10);
    dim3 gridSize(10,10,10);

    test3dInterpolationKernel <<<gridSize, blockSize>>> (d_phiInterpolated, d_u, d_v, d_w, width, height, depth);
}
