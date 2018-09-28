#include "optimizer.cuh"

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Geometry>
#include <cuda_runtime.h>
#include "convolution.cuh"
#include "divergence.cuh"
#include "helper.cuh"
//#include "energyDerivatives.cuh"
#include "interpolator.cuh"
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
	// Allocate current TSDF
	cudaMalloc(&m_d_tsdfLive, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
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
	cudaMalloc(&m_d_dux, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_duy, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_duz, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	/*cudaMalloc(&m_d_dvx, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dvy, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dvz, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dzx, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dzy, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dzz, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;*/
	// Allocate deformation field gradients
	cudaMalloc(&m_d_du, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dv, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dw, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	// Allocate hessian
	cudaMalloc(&m_d_hessXX, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessXY, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessXZ, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessYY, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessYZ, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessZZ, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	/* Testing convolution
	cudaMalloc(&m_d_dx, (3*3*3) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dy, (3*3*3) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_dz, (3*3*3) * sizeof(float)); CUDA_CHECK;*/
	// Allocate divergence
	cudaMalloc(&m_d_div, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	// Allocate laplacian
	cudaMalloc(&m_d_lapu, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
}

void Optimizer::copyArraysToDevice()
{
	cudaMemcpy(m_d_kernelDx, m_kernelDxCentralDiff, (27) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_kernelDy, m_kernelDyCentralDiff, (27) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_kernelDz, m_kernelDzCentralDiff, (27) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	//Deformation
	cudaMemcpy(m_d_deformationFieldU, m_deformationFieldU, (m_gridW * m_gridH * m_gridD)  * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_deformationFieldV, m_deformationFieldV, (m_gridW * m_gridH * m_gridD)  * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_deformationFieldW, m_deformationFieldW, (m_gridW * m_gridH * m_gridD)  * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
}

Optimizer::~Optimizer()
{
	cudaFree(m_d_deformationFieldU); CUDA_CHECK;
	cudaFree(m_d_deformationFieldV); CUDA_CHECK;
	cudaFree(m_d_deformationFieldW); CUDA_CHECK;
	cudaFree(m_d_kernelDx); CUDA_CHECK;
	cudaFree(m_d_kernelDy); CUDA_CHECK;
	cudaFree(m_d_kernelDz); CUDA_CHECK;
	cudaFree(m_d_du); CUDA_CHECK;
	cudaFree(m_d_dv); CUDA_CHECK;
	cudaFree(m_d_dw); CUDA_CHECK;
	cudaFree(m_d_div); CUDA_CHECK;
}

void Optimizer::optimize(TSDFVolume* tsdfLive)
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

void Optimizer::test(TSDFVolume* tsdfLive)
{
	// Initialize variables
	float currentMaxVectorUpdate = 0.01;
	float* tsdfLiveGrid = tsdfLive->ptrTsdf();
	/*for (int i=0; i< m_gridW*m_gridH*m_gridD; i++)
    	tsdfLiveGrid[i]= (float) tsdfLive->ptrColorR()[m_gridW*m_gridH*m_gridD];*/

	for (int i = 0; i < 20; i++)
	{
		std::cout << m_deformationFieldU[i] << std::endl;
	}
	
	


	// TODO: compute gradient of tsdfLive
	cudaMemcpy(m_d_tsdfLive, tsdfLiveGrid, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	computeGradient(m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, m_d_tsdfLive, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
	computeLapacian(m_d_lapu, m_d_tsdfLive, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
	computeHessian(m_d_hessXX, m_d_hessXY, m_d_hessXZ, m_d_hessYY, m_d_hessYZ, m_d_hessZZ, m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
	float* tsdfLiveGridDef = new float[m_gridW * m_gridH * m_gridD];
	float* gradX = new float[m_gridW * m_gridH * m_gridD];
	float* lapU = new float[m_gridW * m_gridH * m_gridD];
	float* hessXX = new float[m_gridW * m_gridH * m_gridD];
	cudaMemcpy(gradX, m_d_sdfDx, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
	cudaMemcpy(lapU, m_d_lapu, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
	cudaMemcpy(hessXX, m_d_hessXX, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
	// Interpolate
	float* d_tsdfDef = NULL;
	cudaMalloc(&d_tsdfDef, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	Interpolator *interptsdf = new Interpolator(tsdfLiveGrid, m_gridW, m_gridH, m_gridD);
	interptsdf->interpolate3D(d_tsdfDef, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);


	//uploadToTextureMemory(tsdfLiveGrid, m_gridW, m_gridH, m_gridD);
	//test3dInterpolation(d_tsdfDef, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
	cudaDeviceSynchronize();
	//freeTextureMemory();
	cudaMemcpy(tsdfLiveGridDef, d_tsdfDef, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

	float* sliceTSDF = new float[m_gridW*m_gridH];
	float* sliceGradX = new float[m_gridW*m_gridH];
	float* sliceLapU = new float[m_gridW*m_gridH];
	float* sliceHessXX = new float[m_gridW*m_gridH];
	float* sliceTSDFDef = new float[m_gridW*m_gridH];

	cv::Mat m_tsdf(m_gridH, m_gridW, CV_32F);	
	cv::Mat m_grad_X(m_gridH, m_gridW, CV_32F);
	cv::Mat m_lapU(m_gridH, m_gridW, CV_32F);
	cv::Mat m_hessXX(m_gridH, m_gridW, CV_32F);
	cv::Mat m_tsdfDef(m_gridH, m_gridW, CV_32F);
	/*for (int i = 0; i < 256; i++)
	{
	getSlice(sliceTSDF, tsdfLiveGrid, i);
	getSlice(sliceGradX, gradX, i);



	convertLayeredToMat(m_tsdf, sliceTSDF);
	convertLayeredToMat(m_grad_X, sliceGradX);
	double min, max;
	cv::minMaxLoc(m_tsdf, &min, &max);
	std::cout << "Slice[ " << i << "]. Min: " << min << ". Max: " << max << std::endl;
	}*/
	getSlice(sliceTSDFDef, tsdfLiveGridDef, 128);
	getSlice(sliceTSDF, tsdfLiveGrid, 128);
	getSlice(sliceGradX, gradX, 128);
	getSlice(sliceLapU, lapU, 128);
	getSlice(sliceHessXX, hessXX, 128);
	convertLayeredToMat(m_tsdf, sliceTSDF);
	convertLayeredToMat(m_grad_X, sliceGradX);
	convertLayeredToMat(m_lapU, sliceLapU);
	convertLayeredToMat(m_hessXX, sliceHessXX);
	convertLayeredToMat(m_tsdfDef, sliceTSDFDef);
	double min, max, minGrad, maxGrad, minLap, maxLap, minHessXX, maxHessXX;
	cv::minMaxLoc(m_tsdf, &min, &max);
	cv::minMaxLoc(m_grad_X, &minGrad, &maxGrad);
	cv::minMaxLoc(m_lapU, &minLap, &maxLap);
	cv::minMaxLoc(m_hessXX, &minHessXX, &maxHessXX);
	std::cout << "Min: " << min << ". Max: " << max << std::endl;
	showImage("TSDF", (m_tsdf - min) / (max - min), 100, 100);
	showImage("TSDF Deform", (m_tsdfDef - min) / (max - min), 100, 100);
	showImage("Grad X", (m_grad_X - minGrad) / (maxGrad - minGrad), 100+40, 100);
	showImage("Laplacian U", m_lapU, 100+40, 100); //(m_lapU - minLap) / (maxLap - minLap)
	showImage("Hessian XX", (m_hessXX - minHessXX) / (maxHessXX - minHessXX), 100+40, 100);
	cv::waitKey();

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

void Optimizer::computeGradient(float* gradOutX, float* gradOutY, float* gradOutZ, const float* gridIn, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
	// Compute gradients for SDF live grid
	computeConvolution3D(gradOutX, gridIn, kernelDx, kradius, w, h, d);
    computeConvolution3D(gradOutY, gridIn, kernelDy, kradius, w, h, d);
	computeConvolution3D(gradOutZ, gridIn, kernelDz, kradius, w, h, d);
}

void Optimizer::computeDivergence(float* divOut, const float* gridInU, const float* gridInV, const float* gridInW, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
	// Compute gradients for the deformation field
	computeConvolution3D(m_d_du, gridInU, kernelDx, kradius, w, h, d);
    computeConvolution3D(m_d_dv, gridInV, kernelDy, kradius, w, h, d);
	computeConvolution3D(m_d_dw, gridInW, kernelDz, kradius, w, h, d);

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
	computeDivergence3DCuda(divOut, m_d_du, m_d_dv, m_d_dw, w, h, d);
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

void Optimizer::computeLapacian(float* lapOut, const float* deformationIn, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
	computeGradient(m_d_dux, m_d_duy, m_d_duz, deformationIn, kernelDx, kernelDy, kernelDz, 1, w, h, d);
	computeDivergence(lapOut, m_d_dux, m_d_duy, m_d_duz, kernelDx, kernelDy, kernelDz, 1, w, h, d);
}

void Optimizer::computeHessian(float* hessOutXX, float* hessOutXY, float* hessOutXZ, float* hessOutYY, float* hessOutYZ, float* hessOutZZ, const float* gradX, const float* gradY, const float* gradZ, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
	computeConvolution3D(hessOutXX, gradX, kernelDx, kradius, w, h, d);
    computeConvolution3D(hessOutXY, gradX, kernelDy, kradius, w, h, d);
	computeConvolution3D(hessOutXZ, gradZ, kernelDx, kradius, w, h, d);
	computeConvolution3D(hessOutYY, gradY, kernelDy, kradius, w, h, d);
    computeConvolution3D(hessOutYZ, gradZ, kernelDy, kradius, w, h, d);
	computeConvolution3D(hessOutZZ, gradZ, kernelDz, kradius, w, h, d);
}