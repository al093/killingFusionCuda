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

Optimizer::Optimizer(TSDFVolume* tsdfGlobal, float* initialDeformationU, float* initialDeformationV, float* initialDeformationW, const float alpha, const float wk, const float ws, const size_t gridW, const size_t gridH, const size_t gridD) :
	m_tsdfGlobal(tsdfGlobal),
    m_deformationFieldU(initialDeformationU),
    m_deformationFieldV(initialDeformationV),
    m_deformationFieldW(initialDeformationW),
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
	cudaMalloc(&m_d_deformationFieldU, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_deformationFieldV, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_deformationFieldW, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	// Allocate kernels
	cudaMalloc(&m_d_kernelDx, (27) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_kernelDy, (27) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_kernelDz, (27) * sizeof(float)); CUDA_CHECK;
	// Allocate SDF live gradients
	cudaMalloc(&m_d_sdfDx, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_sdfDy, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_sdfDz, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	// Allocate deformation field gradients
	cudaMalloc(&m_d_dx, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dy, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dz, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	/* Testing convolution
	cudaMalloc(&m_d_dx, (3*3*3) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dy, (3*3*3) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dz, (3*3*3) * sizeof(float)); CUDA_CHECK;*/
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
	cudaFree(m_d_deformationFieldU); CUDA_CHECK;
	cudaFree(m_d_deformationFieldV); CUDA_CHECK;
	cudaFree(m_d_deformationFieldW); CUDA_CHECK;
	cudaFree(m_d_kernelDx); CUDA_CHECK;
	cudaFree(m_d_kernelDy); CUDA_CHECK;
	cudaFree(m_d_kernelDz); CUDA_CHECK;
	cudaFree(m_d_dx); CUDA_CHECK;
	cudaFree(m_d_dy); CUDA_CHECK;
	cudaFree(m_d_dz); CUDA_CHECK;
	cudaFree(m_d_div); CUDA_CHECK;
}

void Optimizer::optimize(float* optimDeformationU, float* optimDeformationV, float* optimDeformationW, TSDFVolume* tsdfLive)
{
	// Initialize variables
	float currentMaxVectorUpdate = 0.01;
	float* tsdfLiveGrid = tsdfLive->ptrTsdf();

	// TODO: compute gradient of tsdfLive
	computeGradient(m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, tsdfLiveGrid, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);

	// TODO: compute hessian of tsdfLive
	
	do
	{
		/* Test convolution
		{
			m_deformationFieldU = new float[27];
			for (int i = 0; i < 27; i++)
			{
				m_deformationFieldU[i] = 0.0;
			}
			m_deformationFieldU[4] = 1.0;
			m_deformationFieldV = new float[27];
			for (int i = 0; i < 27; i++)
			{
				m_deformationFieldV[i] = 0.0;
			}
			m_deformationFieldV[7] = 1.0;
			m_deformationFieldW = new float[27];
			for (int i = 0; i < 27; i++)
			{
				m_deformationFieldW[i] = 0.0;
			}
			m_deformationFieldW[22] = 1.0;
		}

		cudaMemcpy(m_d_deformationFieldU, m_deformationFieldU, (3 * 3 * 3) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMemcpy(m_d_deformationFieldV, m_deformationFieldV, (3 * 3 * 3) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMemcpy(m_d_deformationFieldW, m_deformationFieldW, (3 * 3 * 3) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		*/
		// Copy necessary arrays from host to device
		cudaMemcpy(m_d_deformationFieldU, m_deformationFieldU, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMemcpy(m_d_deformationFieldV, m_deformationFieldV, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMemcpy(m_d_deformationFieldW, m_deformationFieldW, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

		// TODO: compute laplacians of the deformation field
		
		// TODO: compute divergence of deformation field
		computeDivergence(m_d_div, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
		// TODO: interpolate on gradient of tsdfLive
		
		// TODO: interpolate on hessian of tsdfLive

		// TODO: compute dEdata term

		// TODO: compute dEkilling term

		// TODO: compute dElevel_set term

		// TODO compute dEnon_rigid

		// TODO: update new state of the deformation field

	} while (currentMaxVectorUpdate > MAX_VECTOR_UPDATE_THRESHOLD);
}

void Optimizer::test(float* optimDeformationU, float* optimDeformationV, float* optimDeformationW, TSDFVolume* tsdfLive)
{
	// Initialize variables
	float currentMaxVectorUpdate = 0.01;
	float* tsdfLiveGrid = new float[m_gridW*m_gridH*m_gridD];
	/*for (int i=0; i< m_gridW*m_gridH*m_gridD; i++)
    	tsdfLiveGrid[i]= (float) tsdfLive->ptrColorR()[m_gridW*m_gridH*m_gridD];*/

	// TODO: compute gradient of tsdfLive
	computeGradient(m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, tsdfLiveGrid, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, 1);
	float* gradX = new float[m_gridW * m_gridH * m_gridD];
	cudaMemcpy(gradX, m_d_sdfDx, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;	
	float* sliceTSDF = new float[m_gridW*m_gridH];
	float* sliceGradX = new float[m_gridW*m_gridH];
	getSlice(sliceTSDF, tsdfLiveGrid, 128);
	getSlice(sliceGradX, gradX, 128);

	cv::Mat m_tsdf(m_gridH, m_gridW, CV_32F);	
	cv::Mat m_grad_X(m_gridH, m_gridW, CV_32F);

	convertLayeredToMat(m_tsdf, tsdfLiveGrid);
	convertLayeredToMat(m_grad_X, gradX);

	showImage("TSDF", m_tsdf, 100, 100);
	showImage("Grad X", m_grad_X, 100+40, 100);

	// TODO: compute hessian of tsdfLive
	
	do
	{
		/* Test convolution
		{
			m_deformationFieldU = new float[27];
			for (int i = 0; i < 27; i++)
			{
				m_deformationFieldU[i] = 0.0;
			}
			m_deformationFieldU[4] = 1.0;
			m_deformationFieldV = new float[27];
			for (int i = 0; i < 27; i++)
			{
				m_deformationFieldV[i] = 0.0;
			}
			m_deformationFieldV[7] = 1.0;
			m_deformationFieldW = new float[27];
			for (int i = 0; i < 27; i++)
			{
				m_deformationFieldW[i] = 0.0;
			}
			m_deformationFieldW[22] = 1.0;
		}

		cudaMemcpy(m_d_deformationFieldU, m_deformationFieldU, (3 * 3 * 3) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMemcpy(m_d_deformationFieldV, m_deformationFieldV, (3 * 3 * 3) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMemcpy(m_d_deformationFieldW, m_deformationFieldW, (3 * 3 * 3) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		*/
		// Copy necessary arrays from host to device
		cudaMemcpy(m_d_deformationFieldU, m_deformationFieldU, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMemcpy(m_d_deformationFieldV, m_deformationFieldV, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMemcpy(m_d_deformationFieldW, m_deformationFieldW, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

		// TODO: compute laplacians of the deformation field
		
		// TODO: compute divergence of deformation field
		computeDivergence(m_d_div, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
		// TODO: interpolate on gradient of tsdfLive
		
		// TODO: interpolate on hessian of tsdfLive

		// TODO: compute dEdata term

		// TODO: compute dEkilling term

		// TODO: compute dElevel_set term

		// TODO compute dEnon_rigid

		// TODO: update new state of the deformation field

	} while (currentMaxVectorUpdate > MAX_VECTOR_UPDATE_THRESHOLD);
}

void Optimizer::getSlice(float* sliceOut, const float* gridIn, size_t sliceInd)
{
	for(int i = 0; i < m_gridW*m_gridH; i++)
	{
		sliceOut[i] = gridIn[i + (m_gridW*m_gridH) * sliceInd];
	}
}

void Optimizer::computeGradient(float* gradOutX, float* gradOutY, float* gradOutZ, const float* tsdfLive, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
	// Compute gradients for SDF live grid
	computeConvolution3D(gradOutX, tsdfLive, kernelDx, kradius, w, h, d);
    computeConvolution3D(gradOutY, tsdfLive, kernelDy, kradius, w, h, d);
	computeConvolution3D(gradOutZ, tsdfLive, kernelDz, kradius, w, h, d);
}

void Optimizer::computeDivergence(float* divOut, const float* deformationInU, const float* deformationInV, const float* deformationInW, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
	// Compute gradients for the deformation field
	computeConvolution3D(m_d_dx, deformationInU, kernelDx, kradius, w, h, d);
    computeConvolution3D(m_d_dy, deformationInV, kernelDy, kradius, w, h, d);
	computeConvolution3D(m_d_dz, deformationInW, kernelDz, kradius, w, h, d);

	/* Test convolution
	{
		std::cout << "Sizes: " << w << ", " << h << ", " << d << std::endl;
		float* out_dx = new float[27];
		float* out_dy = new float[27];
		float* out_dz = new float[27];
		cudaMemcpy(out_dx, m_d_dx, (h * w * d) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
		for (int i = 0; i < 27; i++)
		{
			std::cout << "Dx[" << i << "]: " << out_dx[i] << std::endl;
		}
		cudaMemcpy(out_dy, m_d_dy, (h * w * d) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
		for (int i = 0; i < 27; i++)
		{
			std::cout << "Dy[" << i << "]: " << out_dy[i] << std::endl;
		}
		cudaMemcpy(out_dz, m_d_dz, (h * w * d) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
		for (int i = 0; i < 27; i++)
		{
			std::cout << "Dz[" << i << "]: " << out_dz[i] << std::endl;
		}
	}*/
	// Sum the three gradient components
	computeDivergence3DCuda(divOut, m_d_dx, m_d_dy, m_d_dz, w, h, d);
	/* Test divergence
	{
		float* out_div = new float[27];
		cudaMemcpy(out_div, divOut, (3 * 3 * 3) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
		for (int i = 0; i < 27; i++)
		{
			std::cout << "Div[" << i << "]: " << out_div[i] << std::endl;
		}
	}*/
}
