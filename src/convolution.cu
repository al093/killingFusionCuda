// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "convolution.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


// TODO (6.3) define constant memory for convolution kernel
const size_t MAX_KERNEL_SIZE = (3*20+1)*(3*20+1);
__constant__ float constKernel[MAX_KERNEL_SIZE];
// TODO (6.2) define texture for image
texture<float, 2, cudaReadModeElementType> texRef;

__global__
void computeConvolutionTextureMemKernel(float *imgOut, const float *kernel, int kradius, int w, int h, int nc)
{
	// TODO (6.2) compute convolution using texture memory
    int x = threadIdx.x + (size_t) blockDim.x * blockIdx.x;
	int y = threadIdx.y + (size_t) blockDim.y * blockIdx.y;
	size_t z = threadIdx.z + (size_t) blockDim.z * blockIdx.z;

	if (x < w && y < h && z < nc)
	{
		size_t nOmega = (size_t)w*h;
		int l = 2*kradius + 1;
		size_t indPixelImg = x + (size_t)w*y + nOmega*z;
		float sumPixel = 0.0;
		for (int i = 0; i < l; i++)
		{
			for (int j = 0; j < l; j++)
			{
				// Check boundary conditions
				size_t xCoord = min(max(x + i - kradius, 0), w - 1);
				size_t yCoord = min(max(y + j - kradius, 0), h - 1);
				size_t indKernel = i + (size_t)(2*kradius + 1)*j;
				float pixVal = tex2D(texRef, xCoord+0.5f, yCoord+0.5f + h*z);
				sumPixel = sumPixel + pixVal * kernel[indKernel];
			}
		}
		imgOut[indPixelImg] = sumPixel;
	}
}

__global__
void computeConvolutionSharedConstantMemKernel(float *imgOut, const float *imgIn, int kradius, int w, int h, int nc)
{
    // TODO (6.1) compute convolution using shared memory (with a 3D grid ONLY)
	extern __shared__ float s_data[];
	size_t x = threadIdx.x + (size_t) blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + (size_t) blockDim.y * blockIdx.y;
	size_t z = threadIdx.z + (size_t) blockDim.z * blockIdx.z;

	// Fill out the shared memory window
	size_t sharedMemSize = (blockDim.x + 2*kradius) * (blockDim.y + 2*kradius) * blockDim.z;
	size_t nThreads = blockDim.x * blockDim.y * blockDim.z; 
	size_t threadId = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
	size_t nOmega = (size_t)w*h;
	size_t sharedMemW = blockDim.x + 2*kradius;
	for (int s = threadId; s < sharedMemSize; s = s + nThreads)
	{
		int y_s = s / sharedMemW;
		int x_s = s - (y_s * sharedMemW);
		// Compute position in image
		int xImg = x_s - kradius + (blockIdx.x * blockDim.x);
		int yImg = y_s - kradius + (blockIdx.y * blockDim.y) ;
		// Clamping conditions
		size_t xImgClamp = min(max(xImg, 0), w - 1);
		size_t yImgClamp = min(max(yImg, 0), h - 1);
			
		size_t indImgClamp = xImgClamp + (size_t)w*yImgClamp + nOmega*z;
		s_data[s] = imgIn[indImgClamp];
	}
	// Wait until every thread has copied its part of data
	__syncthreads();
	// Perform actual convolution
	if (x < w && y < h && z < nc)
	{
		size_t indImg = x + (size_t)w*y + nOmega*z;
		
		int l = 2*kradius + 1;
		float sumPixel = 0.0;
		for (int i = 0; i < l; i++)
		{
			for (int j = 0; j < l; j++)
			{
				size_t s_dataInd = (threadIdx.x + i) + (threadIdx.y + j) * (blockDim.x + 2*kradius);
				size_t indKernel = i + (size_t)(2*kradius + 1)*j;
				sumPixel = sumPixel + s_data[s_dataInd] * constKernel[indKernel];
			}
		}
		imgOut[indImg] = sumPixel;
	}
}

__global__
void computeConvolutionSharedMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    // TODO (6.1) compute convolution using shared memory (with a 3D grid ONLY)
	extern __shared__ float s_data[];
	size_t x = threadIdx.x + (size_t) blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + (size_t) blockDim.y * blockIdx.y;
	size_t z = threadIdx.z + (size_t) blockDim.z * blockIdx.z;

	// Fill out the shared memory window
	size_t sharedMemSize = (blockDim.x + 2*kradius) * (blockDim.y + 2*kradius) * blockDim.z;
	size_t nThreads = blockDim.x * blockDim.y * blockDim.z; 
	size_t threadId = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
	size_t nOmega = (size_t)w*h;
	size_t sharedMemW = blockDim.x + 2*kradius;
	for (int s = threadId; s < sharedMemSize; s = s + nThreads)
	{
		int y_s = s / sharedMemW;
		int x_s = s - (y_s * sharedMemW);
		// Compute position in image
		int xImg = x_s - kradius + (blockIdx.x * blockDim.x);
		int yImg = y_s - kradius + (blockIdx.y * blockDim.y) ;
		// Clamping conditions
		size_t xImgClamp = min(max(xImg, 0), w - 1);
		size_t yImgClamp = min(max(yImg, 0), h - 1);
			
		size_t indImgClamp = xImgClamp + (size_t)w*yImgClamp + nOmega*z;
		s_data[s] = imgIn[indImgClamp];
	}
	// Wait until every thread has copied its part of data
	__syncthreads();
	// Perform actual convolution
	if (x < w && y < h && z < nc)
	{
		size_t indImg = x + (size_t)w*y + nOmega*z;
		
		int l = 2*kradius + 1;
		float sumPixel = 0.0;
		for (int i = 0; i < l; i++)
		{
			for (int j = 0; j < l; j++)
			{
				size_t s_dataInd = (threadIdx.x + i) + (threadIdx.y + j) * (blockDim.x + 2*kradius);
				size_t indKernel = i + (size_t)(2*kradius + 1)*j;
				sumPixel = sumPixel + s_data[s_dataInd] * kernel[indKernel];
			}
		}
		imgOut[indImg] = sumPixel;
	}
}


__global__
void computeConvolutionGlobalMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int c)
{
    // TODO (5.4) compute convolution using global memory
    int x = threadIdx.x +  (size_t) blockDim.x * blockIdx.x;
	int y = threadIdx.y +  (size_t) blockDim.y * blockIdx.y;
	if (x < w && y < h)
	{
		size_t nOmega = (size_t)w*h;
		int l = 2*kradius + 1;
		size_t indPixelImg = x + (size_t)w*y + nOmega*c;
		float sumPixel = 0.0;
		for (int i = 0; i < l; i++)
		{
			for (int j = 0; j < l; j++)
			{
				// Check boundary conditions
				size_t xCoord = min(max(x + i - kradius, 0), w - 1);
				size_t yCoord = min(max(y + j - kradius, 0), h - 1);
				size_t indImgShift = xCoord + (size_t)w*yCoord + nOmega*c;
				size_t indKernel = i + (size_t)(2*kradius + 1)*j;
				sumPixel = sumPixel + imgIn[indImgShift] * kernel[indKernel];
			}
		}
		imgOut[indPixelImg] = sumPixel;
	}
}

__global__
void computeConvolution3DGlobalMemKernel(float* gridOut, const float* gridIn, const float* kernel, int kradius, int w, int h, int d)
{
    // compute convolution using global memory
    int x = threadIdx.x +  (size_t) blockDim.x * blockIdx.x;
	int y = threadIdx.y +  (size_t) blockDim.y * blockIdx.y;
	int z = threadIdx.z +  (size_t) blockDim.z * blockIdx.z;
	if (x < w && y < h && z < d)
	{
		int l = 2*kradius + 1;
		size_t sliceSize = (size_t)w*h;
		size_t indVoxel = x + (size_t)w*y + sliceSize*z;
		float sumVoxels = 0.0;
		for (int i = 0; i < l; i++)
		{
			for (int j = 0; j < l; j++)
			{
				for (int k = 0; k < l; k++)
				{
					// Check boundary conditions
					size_t xCoord = min(max(x + i - kradius, 0), w - 1);
					size_t yCoord = min(max(y + j - kradius, 0), h - 1);
					size_t zCoord = min(max(z + k - kradius, 0), d - 1);
					size_t indVoxelShift = xCoord + (size_t)w*yCoord + sliceSize*zCoord;
					size_t indKernel = i + (size_t)l*j + (size_t)l*l*k;
					sumVoxels = sumVoxels + gridIn[indVoxelShift] * kernel[indKernel];
				}
			}
		}
		gridOut[indVoxel] = sumVoxels;
	}
}

void createConvolutionKernel(float *kernel, int kradius, float sigma)
{
    // TODO (5.1) fill (gaussian) convolution kernel
	float mean = 0.0;
	float sum = 0.0;
	int l = 2*kradius + 1;
	for (int i = 0; i < l; i++)
	{
		for (int j = 0; j < l; j++)
		{
			float dist = sqrt(pow(i - kradius, 2) + pow(j - kradius, 2));
			kernel[i + l*j] = (1.0 / (sigma * sqrt(2.0*M_PI))) * exp(-(0.5) * pow((dist - mean) / sigma, 2));
			sum = sum + kernel[i + l*j];
		}
	}
	// Normalize the filter
	for (int i = 0; i < l; i++)
	{
		for (int j = 0; j < l; j++)
		{
			kernel[i + l*j] = kernel[i + l*j] / sum;
		}
	}
}


void computeConvolution(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // TODO (5.3) compute convolution on CPU
	size_t nOmega = (size_t)w*h;
	int l = 2*kradius + 1;
	for (size_t c = 0; c < nc; c++)
	{
		for (int x = 0; x < w; x++)
		{
			for (int y = 0; y < h; y++)
			{
				size_t indPixelImg = x + (size_t)w*y + nOmega*c;
				float sumPixel = 0.0;
				for (int i = 0; i < l; i++)
				{
					for (int j = 0; j < l; j++)
					{
						size_t xCoord = min(max(x + i - kradius, 0), w - 1);
						size_t yCoord = min(max(y + j - kradius, 0), h - 1);
						size_t indImgShift = xCoord + (size_t)w*yCoord + nOmega*c;
						size_t indKernel = i + (size_t)(2*kradius + 1)*j;
						sumPixel = sumPixel + imgIn[indImgShift] * kernel[indKernel];
					}
				}
				imgOut[indPixelImg] = sumPixel;
			}
		}
	}
}


void computeConvolutionTextureMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (6.2) specify suitable block size
    dim3 grid = computeGrid3D(block, w, h, nc);

    // TODO (6.2) bind texture
	texRef.addressMode[0] = cudaAddressModeClamp;		// clamp x to border
	texRef.addressMode[1] = cudaAddressModeClamp;		// clamp y to border
	texRef.filterMode = cudaFilterModeLinear;			// linear interpolation
	texRef.normalized = false;							// access as (x+0.5f, y+0.5f + h*z), not as ((x+0.5f)/w, (y+0.5f)/h)
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, &texRef, imgIn, &desc, w, h*nc, w*sizeof(imgIn[0]));
    // run cuda kernel
    // TODO (6.2) execute kernel for convolution using global memory
	computeConvolutionTextureMemKernel <<<grid, block>>> (imgOut, kernel, kradius, w, h, nc);
    // TODO (6.2) unbind texture
	cudaUnbindTexture(texRef);
    // check for errors
    // TODO (6.2)
	CUDA_CHECK;
}


void computeConvolutionSharedMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (6.1) specify suitable block size
    dim3 grid = computeGrid3D(block, w, h, nc);

    // TODO (6.1) calculate shared memory size
	size_t smBytes = ((block.x + 2*kradius) * (block.y + 2*kradius) * block.z) * sizeof(float);

    // run cuda kernel
    // TODO (6.1) execute kernel for convolution using global memory
	computeConvolutionSharedMemKernel <<<grid, block, smBytes>>> (imgOut, imgIn, kernel, kradius, w, h, nc);
    // check for errors
    // TODO (6.1)
	CUDA_CHECK;
}

void computeConvolutionSharedConstantMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (6.1) specify suitable block size
    dim3 grid = computeGrid3D(block, w, h, nc);

    // TODO (6.1) calculate shared memory size
	size_t smBytes = ((block.x + 2*kradius) * (block.y + 2*kradius) * block.z) * sizeof(float);

	// TODO (6.3) write constant kernel
	cudaMemcpyToSymbol(constKernel, kernel, (2*kradius + 1)*(2*kradius + 1) * sizeof(float));

    // run cuda kernel
    // TODO (6.1) execute kernel for convolution using global memory
	computeConvolutionSharedConstantMemKernel <<<grid, block, smBytes>>> (imgOut, imgIn, kradius, w, h, nc);
    // check for errors
    // TODO (6.1)
	CUDA_CHECK;
}


void computeConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (5.4) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (5.4) execute kernel for convolution using global memory
	for (size_t c = 0; c < nc; c++)
	{
		computeConvolutionGlobalMemKernel <<<grid, block>>> (imgOut, imgIn, kernel, kradius, w, h, c); 
	}
    // check for errors
    // TODO (5.4)
	CUDA_CHECK;
}

void computeConvolution3D(float *gridOut, const float *gridIn, const float* kernel, int kradius, int w, int h, int d)
{
    if (!gridOut || !gridIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 8, 3);
    dim3 grid = computeGrid3D(block, w, h, d);

    // run cuda kernel
    // execute kernel for convolution using global memory
	computeConvolution3DGlobalMemKernel <<<grid, block>>> (gridOut, gridIn, kernel, kradius, w, h, d); 
    // check for errors
	CUDA_CHECK;
}
