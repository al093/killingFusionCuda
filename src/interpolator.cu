///Cuda Functions to calculate the energy derivatives and also helper function for using 3D texture memory.
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
        //Remember!! to always add 0.5, the voxels have actual values at their centers, and the size of a voxel is 1x1x1, so need to add .5, .5, .5 for center 
        d_outputValues[idx] = tex3D<float>(tex, fx + d_u[idx] + 0.5, fy + d_v[idx] + 0.5, fz + d_w[idx] + 0.5);
    }
}


///use this method to bind Texture memory to Cuda array.
///TODO currently testing for only one 3D voxel grid
void Interpolator::uploadToTextureMemory(float* h_phi, int w, int h, int d)
{
    cudaArray *cuArray_phi;
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

    
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(float));
    resDesc.resType = cudaResourceTypeLinear;
         // linear interpolation

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = false;                      // access with normalized phiture coordinates
    texDesc.addressMode[0] = cudaAddressModeClamp;   // wrap phiture coordinates
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;

/*
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
*/
//-----------------------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------------------
    // BEGIN WEB CODE
       //curand Random Generator (needs compiler link -lcurand)
        curandGenerator_t gen;
        curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen,1235ULL+i);
        curandGenerateUniform(gen, d_NoiseTest, cubeSizeNoiseTest);//writing data to d_NoiseTest
        curandDestroyGenerator(gen);

        //cudaArray Descriptor
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        //cuda Array
        cudaArray *d_cuArr;
        checkCudaErrors(cudaMalloc3DArray(&d_cuArr, &channelDesc, make_cudaExtent(SizeNoiseTest*sizeof(float),SizeNoiseTest,SizeNoiseTest), 0));
        cudaMemcpy3DParms copyParams = {0};


        //Array creation
        copyParams.srcPtr   = make_cudaPitchedPtr(d_NoiseTest, SizeNoiseTest*sizeof(float), SizeNoiseTest, SizeNoiseTest);
        copyParams.dstArray = d_cuArr;
        copyParams.extent   = make_cudaExtent(SizeNoiseTest,SizeNoiseTest,SizeNoiseTest);
        copyParams.kind     = cudaMemcpyDeviceToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
        //Array creation End

        cudaResourceDesc    texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array  = d_cuArr;
        cudaTextureDesc     texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));
        texDescr.normalizedCoords = false;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;
        texDescr.readMode = cudaReadModeElementType;
        checkCudaErrors(cudaCreateTextureObject(&texNoise[i], &texRes, &texDescr, NULL));}

void Interpolator::freeTextureMemory()
{
    /*cudaUnbindTexture(phi);
    cudaFreeArray(cuArray_phi);*/
}


void Interpolator::interpolate3D(float *d_phiInterpolated, const float *d_u, const float *d_v, const float *d_w, int width, int height, int depth)
{
    dim3 blockSize(32, 8, 1);
    dim3 grid = computeGrid3D(blockSize, width, height, depth);

    interpolate3DKernel <<<grid, blockSize>>> (d_phiInterpolated, tex, d_u, d_v, d_w, width, height, depth);
}
