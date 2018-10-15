// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Interpolator object to interpolate a base grid using a given vector field
// ########################################################################
#include "interpolator.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include "helper.cuh"

Interpolator::Interpolator(float* h_phi, int width, int height, int depth)
{
    uploadToTextureMemory(h_phi, width, height, depth);
}


Interpolator::~Interpolator(){
    freeTextureMemory();
}

__global__
void interpolate3DKernel(float *d_outputValues, cudaTextureObject_t tex, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth)
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
        // Add 0.5 to get the actual values at the voxels' centers (voxel size 1x1x1) 
        d_outputValues[idx] = tex3D<float>(tex, fx + d_u[idx] + 0.5, fy + d_v[idx] + 0.5, fz + d_w[idx] + 0.5);
    }
}

void Interpolator::uploadToTextureMemory(float* d_grid, int w, int h, int d)
{
    // Define channel format descriptor
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    // Set the grid size
    cudaExtent extent;
    extent.width = w;
    extent.height = h;
    extent.depth = d;

    // Allocate 3D cuda array
    cudaMalloc3DArray(&cuArray_grid, &desc, extent);
    CUDA_CHECK;

    // Copy from device memory to CudaArray in device memory
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)d_grid, extent.width*sizeof(float), extent.width, extent.height);
    copyParams.dstArray = cuArray_grid;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.extent = extent;
    cudaMemcpy3D(&copyParams);
    CUDA_CHECK;

    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array  = cuArray_grid;
    cudaTextureDesc     texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
    cudaCreateTextureObject(&texGrid, &texRes, &texDescr, NULL);
    CUDA_CHECK;
}

void Interpolator::freeTextureMemory()
{
    cudaFreeArray(cuArray_grid); CUDA_CHECK;
	cudaDestroyTextureObject(texGrid); CUDA_CHECK;
}


void Interpolator::interpolate3D(float *d_gridInterpolated, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    interpolate3DKernel <<<grid, blockSize>>> (d_gridInterpolated, texGrid, d_u, d_v, d_w, width, height, depth);
}
