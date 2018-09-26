#include "optimizer.cuh"

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Geometry>
#include <cuda_runtime.h>
#include "convolution.cuh"
#include "divergence.cuh"
#include "helper.cuh"
#include <opencv2/highgui/highgui.hpp>

Optimizer::Optimizer(TSDFVolume* tsdfGlobal, float* initialDeformation, const float alpha, const float wk, const float ws, const size_t gridW, const size_t gridH, const size_t gridD) :
	m_tsdfGlobal(tsdfGlobal),
    m_deformationField(initialDeformation),
    m_alpha(alpha),
	m_wk(wk),
	m_ws(ws),
	m_gridW(gridW), 
	m_gridH(gridH),
	m_gridD(gridD)
{
    allocateMemoryInDevice();
	copyArraysToDevice();
}

void Optimizer::allocateMemoryInDevice()
{
	// Allocate deformation field
	cudaMalloc(&m_d_deformationField, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	// Allocate kernels
	cudaMalloc(&m_d_kernelDx, (27) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_kernelDy, (27) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_kernelDz, (27) * sizeof(float)); CUDA_CHECK;
	// Allocate gradients
	cudaMalloc(&m_d_dx, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dy, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dz, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	// Allocate divergence
	cudaMalloc(&m_d_div, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
}

void Optimizer::copyArraysToDevice()
{
	cudaMemcpy(m_d_kernelDx, m_kernelDxCentralDiff, (27) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_kernelDy, m_kernelDyCentralDiff, (27) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_kernelDz, m_kernelDzCentralDiff, (27) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
}

Optimizer::~Optimizer()
{
	cudaFree(m_d_deformationField); CUDA_CHECK;
	cudaFree(m_d_kernelDx); CUDA_CHECK;
	cudaFree(m_d_kernelDy); CUDA_CHECK;
	cudaFree(m_d_kernelDz); CUDA_CHECK;
	cudaFree(m_d_dx); CUDA_CHECK;
	cudaFree(m_d_dy); CUDA_CHECK;
	cudaFree(m_d_dz); CUDA_CHECK;
	cudaFree(m_d_div); CUDA_CHECK;
}

void Optimizer::optimize(float* optimDeformation, TSDFVolume* tsdfLive)
{
	// Initialize variables
	float currentMaxVectorUpdate = 1000000.0;

	// TODO: compute gradient of tsdfLive

	// TODO: compute hessian of tsdfLive

	do
	{
		// Copy necessary arrays from host to device
		cudaMemcpy(m_d_deformationField, m_deformationField, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

		// TODO: compute laplacians of the deformation field
		
		// TODO: compute divergence of deformation field
		computeDivergence(m_d_div, m_d_deformationField, m_kernelDxCentralDiff, m_kernelDyCentralDiff, m_kernelDzCentralDiff, 1, m_gridW, m_gridH, m_gridD);
		// TODO: interpolate on gradient of tsdfLive
		
		// TODO: interpolate on hessian of tsdfLive

		// TODO: compute dEdata term

		// TODO: compute dEkilling term

		// TODO: compute dElevel_set term

		// TODO compute dEnon_rigid

		// TODO: update new state of the deformation field

	} while (currentMaxVectorUpdate > MAX_VECTOR_UPDATE_THRESHOLD);
}

void Optimizer::computeDivergence(float* divOut, const float* deformationIn, const float *kernelDx, const float *kernelDy, const float *kernelDz, int kradius, int w, int h, int d)
{

	// Compute gradients for the deformation field
	computeConvolution3D(m_d_dx, deformationIn, kernelDx, kradius, w, h, d);
    computeConvolution3D(m_d_dy, deformationIn, kernelDy, kradius, w, h, d);
	computeConvolution3D(m_d_dz, deformationIn, kernelDz, kradius, w, h, d);
	// Sum the three gradient components
	computeDivergence3DCuda(divOut, m_d_dx, m_d_dy, m_d_dz, w, h, d);
}
