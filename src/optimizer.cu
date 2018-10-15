// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Optimizer object that performs non-rigid optimization
// ########################################################################
#include "optimizer.cuh"

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Geometry>
#include <cuda_runtime.h>
#include "gradient.cuh"
#include "divergence.cuh"
#include "helper.cuh"
#include "energy.cuh"
#include "energyDerivatives.cuh"
#include "gridOperations.cuh"
#include "interpolator.cuh"
#include "reduction.cuh"
#include "magnitude.cuh"
#include "visualization.cuh"
#include "marching_cubes.h"
#include "color.h"

#include <opencv2/highgui/highgui.hpp>
#include "cublas_v2.h"

Color::Modifier red(Color::FG_RED);
Color::Modifier green(Color::FG_GREEN);
Color::Modifier def(Color::FG_DEFAULT);

Optimizer::Optimizer(TSDFVolume* tsdfGlobal, float* initialDeformationU, float* initialDeformationV, float* initialDeformationW, 
                     const float alpha, const float wk, const float ws, const float gamma, const size_t maxIterations,
                     const float voxelSize, const float tsdfGradScale,
                     const bool debugMode, const size_t gridW, const size_t gridH, const size_t gridD) :
	m_tsdfGlobal(tsdfGlobal),
    m_deformationFieldU(initialDeformationU),
    m_deformationFieldV(initialDeformationV),
    m_deformationFieldW(initialDeformationW),
    m_alpha(alpha),
	m_wk(wk),
	m_ws(ws),
	m_gamma(gamma),
	m_maxIterations(maxIterations),
	m_voxelSize(voxelSize),
	m_debugMode(debugMode),
	m_gridW(gridW), 
	m_gridH(gridH),
    m_gridD(gridD),
    m_tsdfGradScale(tsdfGradScale)
{
	m_maxVectorUpdateThreshold = 0.1 / (m_voxelSize * 1000.0);
    allocateMemoryInDevice();
	copyArraysToDevice();

	// Create cublas handle
    cublasCreate(&m_handle);
}

void Optimizer::allocateMemoryInDevice()
{
	// Allocate current TSDF
	cudaMalloc(&m_d_tsdfGlobal, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_tsdfLive, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_tsdfGlobalWeights, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_tsdfLiveWeights, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;

	// Allocate deformation field
	cudaMalloc(&m_d_deformationFieldU, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_deformationFieldV, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_deformationFieldW, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;

	// Allocate SDF live gradients
	cudaMalloc(&m_d_sdfDx, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_sdfDy, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_sdfDz, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;

	// Allocate deformation field gradients
	cudaMalloc(&m_d_dux, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_duy, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_duz, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&m_d_dvx, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_dvy, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_dvz, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_dwx, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_dwy, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_dwz, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;

	// Allocate hessian
	cudaMalloc(&m_d_hessXX, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessXY, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessXZ, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessYY, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessYZ, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessZZ, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;

	// Allocate divergence
	cudaMalloc(&m_d_div, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_divX, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_divY, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_divZ, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;

	// Allocate laplacian
    cudaMalloc(&m_d_lapU, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_lapV, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_lapW, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;

	// Allocate interpolated grids
	cudaMalloc(&m_d_tsdfLiveDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_sdfDxDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_sdfDyDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_sdfDzDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessXXDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessXYDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessXZDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessYYDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessYZDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&m_d_hessZZDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_tsdfLiveWeightsDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;

	// Allocate magnitude grid
	cudaMalloc(&m_d_magnitude, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;

    // Allocate for mask - only near surface regions deformation is to be calculated.
    cudaMalloc(&m_d_mask, (m_gridW * m_gridH * m_gridD) * sizeof(bool)); CUDA_CHECK;
    cudaMemset(m_d_mask, true, (m_gridW * m_gridH * m_gridD) * sizeof(bool)); CUDA_CHECK;

    // Allocate and initialize to zero the memory for the gradients of energy
    cudaMalloc(&m_d_energyDu, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_energyDv, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_energyDw, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMemset(m_d_energyDu, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMemset(m_d_energyDv, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMemset(m_d_energyDw, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    
    // Allocate gradients used by the compute divergence function
    cudaMalloc(&m_d_du, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_dv, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_dw, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    
    cudaMalloc(&m_d_dfx, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_dfy, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_dfz, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
}

void Optimizer::copyArraysToDevice()
{
	// TSDF Global (not working by now)
	cudaMemcpy(m_d_tsdfGlobal, m_tsdfGlobal->ptrTsdf(), (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_tsdfGlobalWeights, m_tsdfGlobal->ptrTsdfWeights(), (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	
	// Deformation fields
	cudaMemcpy(m_d_deformationFieldU, m_deformationFieldU, (m_gridW * m_gridH * m_gridD)  * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_deformationFieldV, m_deformationFieldV, (m_gridW * m_gridH * m_gridD)  * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_deformationFieldW, m_deformationFieldW, (m_gridW * m_gridH * m_gridD)  * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
}

Optimizer::~Optimizer()
{
    // Free current TSDF
	cudaFree(m_d_tsdfGlobal); CUDA_CHECK;
	cudaFree(m_d_tsdfLive); CUDA_CHECK;
	cudaFree(m_d_tsdfGlobalWeights); CUDA_CHECK;
	cudaFree(m_d_tsdfLiveWeights); CUDA_CHECK;

	// Free deformation field
	cudaFree(m_d_deformationFieldU); CUDA_CHECK;
	cudaFree(m_d_deformationFieldV); CUDA_CHECK;
	cudaFree(m_d_deformationFieldW); CUDA_CHECK;

	// Free SDF live gradients
	cudaFree(m_d_sdfDx); CUDA_CHECK;
	cudaFree(m_d_sdfDy); CUDA_CHECK;
	cudaFree(m_d_sdfDz); CUDA_CHECK;

	// Free deformation field gradients
	cudaFree(m_d_dux); CUDA_CHECK;
	cudaFree(m_d_duy); CUDA_CHECK;
	cudaFree(m_d_duz); CUDA_CHECK;
    cudaFree(m_d_dvx); CUDA_CHECK;
    cudaFree(m_d_dvy); CUDA_CHECK;
    cudaFree(m_d_dvz); CUDA_CHECK;
    cudaFree(m_d_dwx); CUDA_CHECK;
    cudaFree(m_d_dwy); CUDA_CHECK;
    cudaFree(m_d_dwz); CUDA_CHECK;

	// Free hessian
	cudaFree(m_d_hessXX); CUDA_CHECK;
	cudaFree(m_d_hessXY); CUDA_CHECK;
	cudaFree(m_d_hessXZ); CUDA_CHECK;
	cudaFree(m_d_hessYY); CUDA_CHECK;
	cudaFree(m_d_hessYZ); CUDA_CHECK;
	cudaFree(m_d_hessZZ); CUDA_CHECK;

	// Free divergence
	cudaFree(m_d_div); CUDA_CHECK;
    cudaFree(m_d_divX); CUDA_CHECK;
    cudaFree(m_d_divY); CUDA_CHECK;
    cudaFree(m_d_divZ); CUDA_CHECK;

	// Free laplacian
    cudaFree(m_d_lapU); CUDA_CHECK;
    cudaFree(m_d_lapV); CUDA_CHECK;
    cudaFree(m_d_lapW); CUDA_CHECK;

	// Free interpolated grids
	cudaFree(m_d_tsdfLiveDeform); CUDA_CHECK;
	cudaFree(m_d_sdfDxDeform); CUDA_CHECK;
	cudaFree(m_d_sdfDyDeform); CUDA_CHECK;
	cudaFree(m_d_sdfDzDeform); CUDA_CHECK;
	cudaFree(m_d_hessXXDeform); CUDA_CHECK;
	cudaFree(m_d_hessXYDeform); CUDA_CHECK;
	cudaFree(m_d_hessXZDeform); CUDA_CHECK;
	cudaFree(m_d_hessYYDeform); CUDA_CHECK;
	cudaFree(m_d_hessYZDeform); CUDA_CHECK;
	cudaFree(m_d_hessZZDeform); CUDA_CHECK;
    cudaFree(m_d_tsdfLiveWeightsDeform); CUDA_CHECK;

	// Free magnitude grid
	cudaFree(m_d_magnitude); CUDA_CHECK;

    // Free mask - only near surface regions deformation is to be calculated
    cudaFree(m_d_mask); CUDA_CHECK;

    // Free gradients of energy
    cudaFree(m_d_energyDu); CUDA_CHECK;
    cudaFree(m_d_energyDv); CUDA_CHECK;
    cudaFree(m_d_energyDw); CUDA_CHECK;
    
    // Free gradients used by the compute divergence function
    cudaFree(m_d_du); CUDA_CHECK;
    cudaFree(m_d_dv); CUDA_CHECK;
    cudaFree(m_d_dw); CUDA_CHECK;
    
    cudaFree(m_d_dfx); CUDA_CHECK;
    cudaFree(m_d_dfy); CUDA_CHECK;
    cudaFree(m_d_dfz); CUDA_CHECK;

    // Destroy cublas handle
    cublasDestroy(m_handle);
}

void Optimizer::getTSDFGlobalPtr(float* tsdfGlobalPtr)
{
	cudaMemcpy(tsdfGlobalPtr, m_d_tsdfGlobal, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
}

void Optimizer::getTSDFGlobalWeightsPtr(float* tsdfGlobalWeightsPtr)
{
	cudaMemcpy(tsdfGlobalWeightsPtr, m_d_tsdfGlobalWeights, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
}

void Optimizer::printTimes()
{
	std::cout << std::endl << "Mean optimization time per frame: " << (m_timeFramesOptimized / (float) m_nFramesOptimized) * 1000 << "ms" << std::endl;
	std::cout << "Mean running times (ms): " << std::endl;
	std::cout << "- computeGradient: " << 1000*m_timeComputeGradient / (float)m_nComputeGradient << " (" << m_nComputeGradient << " iterations)" <<std::endl;
	std::cout << "- computeHessian: " << 1000*m_timeComputeHessian / (float)m_nComputeHessian << " (" << m_nComputeHessian << " iterations)" <<std::endl;
	std::cout << "- computeLaplacian: " << 1000*m_timeComputeLaplacian / (float)m_nComputeLaplacian << " (" << m_nComputeLaplacian << " iterations)" <<std::endl;
	std::cout << "- computeDivergence: " << 1000*m_timeComputeDivergence / (float)m_nComputeDivergence << " (" << m_nComputeDivergence << " iterations)" <<std::endl;
	std::cout << "- computeDataTermDerivative: " << 1000*m_timeComputeDataTermDerivative / (float)m_nComputeDataTermDerivative << " (" << m_nComputeDataTermDerivative << " iterations)" <<std::endl;
	std::cout << "- computeLevelSetDerivative: " << 1000*m_timeComputeLevelSetDerivative / (float)m_nComputeLevelSetDerivative << " (" << m_nComputeLevelSetDerivative << " iterations)" <<std::endl;
	std::cout << "- computeMotionRegularizerDerivative: " << 1000*m_timeComputeMotionRegularizerDerivative / (float)m_nComputeMotionRegularizerDerivative << " (" << m_nComputeMotionRegularizerDerivative << " iterations)" <<std::endl;
	std::cout << "- addArray: " << 1000*m_timeAddArray / (float)m_nAddArray << " (" << m_nAddArray << " iterations)" <<std::endl;
	std::cout << "- computeMagnitude: " << 1000*m_timeComputeMagnitude / (float)m_nComputeMagnitude << " (" << m_nComputeMagnitude << " iterations)" <<std::endl;
	std::cout << "- findAbsMax: " << 1000*m_timeFindAbsMax / (float)m_nFindAbsMax << " (" << m_nFindAbsMax << " iterations)" <<std::endl;
	std::cout << "- addWeightedArray: " << 1000*m_timeAddWeightedArray / (float)m_nAddWeightedArray << " (" << m_nAddWeightedArray << " iterations)" <<std::endl;
	std::cout << "- interpolation: " << 1000*m_interpolation / (float)m_nInterpolation << " (" << m_nInterpolation << " iterations)" <<std::endl;
}

void Optimizer::optimize(TSDFVolume* tsdfLive)
{
	Timer frameTimer;
	frameTimer.start();
	// Initialize variables
	float currentMaxVectorUpdate;
	float* tsdfLiveGrid = tsdfLive->ptrTsdf();
	float* tsdfLiveWeights = tsdfLive->ptrTsdfWeights();
	Timer timer;
	// Copy arrays from host to device
	cudaMemcpy(m_d_tsdfLive, tsdfLiveGrid, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_tsdfLiveWeights, tsdfLiveWeights, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

	// Compute gradient of tsdfLive
    timer.start();
	computeGradient(m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, m_d_tsdfLive, m_gridW, m_gridH, m_gridD);
	timer.end();
    m_timeComputeGradient += timer.get();
    m_nComputeGradient += 1;

	// Compute hessian of tsdfLive
	timer.start();
	computeHessian(m_d_hessXX, m_d_hessXY, m_d_hessXZ, m_d_hessYY, m_d_hessYZ, m_d_hessZZ, m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, m_gridW, m_gridH, m_gridD);
	timer.end();
    m_timeComputeHessian += timer.get();
    m_nComputeHessian += 1;
	// Create interpolators
	Interpolator *interpTSDFLive = new Interpolator(m_d_tsdfLive, m_gridW, m_gridH, m_gridD);
	Interpolator *interpTSDFDx = new Interpolator(m_d_sdfDx, m_gridW, m_gridH, m_gridD);
	Interpolator *interpTSDFDy = new Interpolator(m_d_sdfDy, m_gridW, m_gridH, m_gridD);
	Interpolator *interpTSDFDz = new Interpolator(m_d_sdfDz, m_gridW, m_gridH, m_gridD);
	Interpolator *interpHessXX = new Interpolator(m_d_hessXX, m_gridW, m_gridH, m_gridD);
	Interpolator *interpHessXY = new Interpolator(m_d_hessXY, m_gridW, m_gridH, m_gridD);
	Interpolator *interpHessXZ = new Interpolator(m_d_hessXZ, m_gridW, m_gridH, m_gridD);
	Interpolator *interpHessYY = new Interpolator(m_d_hessYY, m_gridW, m_gridH, m_gridD);
	Interpolator *interpHessYZ = new Interpolator(m_d_hessYZ, m_gridW, m_gridH, m_gridD);
	Interpolator *interpHessZZ = new Interpolator(m_d_hessZZ, m_gridW, m_gridH, m_gridD);
	
    Interpolator *interpLiveWeights = new Interpolator(m_d_tsdfLiveWeights, m_gridW, m_gridH, m_gridD);

	/* Energy Computation (wrt the paper): 	PSI is m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW
											PHIglobal is m_d_tsdfGlobal			PHIn is m_d_tsdfLive
											GRADIENT OF PHIn IS: m_d_sdfDx, m_d_sdfDy, m_d_sdfDz
											HESSIAN OF PHIn IS: m_d_hessXX, m_d_hessXY, m_d_hessXZ, m_d_hessYY, m_d_hessYZ, m_d_hessZZ
	*/

    //used while debug mode to save the energies for plotting
    float dataEnergyArr[1000] = {0.0};
    float killingEnergyArr[1000] = {0.0};
    float levelSetEnergyArr[1000] = {0.0};
    float totalEnergyArr[1000] = {0.0};
    
    if (m_debugMode) std::cout<< "Deforming SDF..." << std::endl;

    // Compute current mask
    computeMask(m_d_mask, m_d_tsdfLiveDeform, m_gridW, m_gridH, m_gridD);

    size_t itr = 0;

    do
	{
        itr = itr + 1;
        if(m_debugMode) std::cout << std::endl << "GD itr num: " << itr;
        
		// Interpolate TSDF Live Frame
        timer.start();
        interpTSDFLive->interpolate3D(m_d_tsdfLiveDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_interpolation += timer.get();
        m_nInterpolation += 1;
        
        // Interpolate the gradient of Phi_n wrt the psi
        interpTSDFDx->interpolate3D(m_d_sdfDxDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        interpTSDFDy->interpolate3D(m_d_sdfDyDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        interpTSDFDz->interpolate3D(m_d_sdfDzDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        
        // Interpolate the hessian wrt the psi
        interpHessXX->interpolate3D(m_d_hessXXDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        interpHessXY->interpolate3D(m_d_hessXYDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        interpHessXZ->interpolate3D(m_d_hessXZDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        interpHessYY->interpolate3D(m_d_hessYYDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        interpHessYZ->interpolate3D(m_d_hessYZDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        interpHessZZ->interpolate3D(m_d_hessZZDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);

        // Set m_d_energyDu/v/w initially to zero
        cudaMemset(m_d_energyDu, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        cudaMemset(m_d_energyDv, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        cudaMemset(m_d_energyDw, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        
        // Compute data term derivatives
        timer.start();
        computeDataTermDerivative(m_d_energyDu, m_d_energyDv, m_d_energyDw,
                                  m_d_tsdfLiveDeform, m_d_tsdfGlobal, m_d_mask,
                                  m_d_sdfDxDeform, m_d_sdfDyDeform, m_d_sdfDzDeform,
                                  m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeDataTermDerivative += timer.get();
    	m_nComputeDataTermDerivative += 1;

        // Add the derivatives from the levelSet Derivatives with a scalar constant wk
       	timer.start();
        computeLevelSetDerivative(m_d_energyDu, m_d_energyDv, m_d_energyDw,
                                  m_d_hessXXDeform, m_d_hessXYDeform, m_d_hessXZDeform,
                                  m_d_hessYYDeform, m_d_hessYZDeform, m_d_hessZZDeform,
                                  m_d_sdfDxDeform, m_d_sdfDyDeform, m_d_sdfDzDeform,
                                  m_d_mask, m_ws, m_tsdfGradScale, m_voxelSize,
                                  m_gridW, m_gridH, m_gridD);
		timer.end();
        m_timeComputeLevelSetDerivative += timer.get();
    	m_nComputeLevelSetDerivative += 1;

        // Compute laplacians of the deformation field
        timer.start();
        computeLaplacian(m_d_lapU, m_d_dux, 0, m_d_deformationFieldU, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeLaplacian += timer.get();
    	m_nComputeLaplacian += 1;
    	timer.start();
        computeLaplacian(m_d_lapV, m_d_dvy, 1, m_d_deformationFieldV, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeLaplacian += timer.get();
    	m_nComputeLaplacian += 1;
    	timer.start();
        computeLaplacian(m_d_lapW, m_d_dwz, 2, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeLaplacian += timer.get();
    	m_nComputeLaplacian += 1;

        // Compute divergence of the deformation field
        timer.start();
        sumGradients(m_d_div, m_d_dux, m_d_dvy, m_d_dwz, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeDivergence += timer.get();
    	m_nComputeDivergence += 1;

    	// Compute the gradients of the divergence of deformation field
    	timer.start();
        computeGradient(m_d_divX, m_d_divY, m_d_divZ, m_d_div, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeGradient += timer.get();
    	m_nComputeGradient += 1;
        
        // Compute motion regularizer derivative
        timer.start();
        computeMotionRegularizerDerivative(m_d_energyDu, m_d_energyDv, m_d_energyDw,
                                           m_d_lapU, m_d_lapV, m_d_lapW,
                                           m_d_divX, m_d_divY, m_d_divZ,
                                           m_d_mask, m_wk, m_gamma,
                                           m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeMotionRegularizerDerivative += timer.get();
    	m_nComputeMotionRegularizerDerivative += 1;

        if (m_debugMode)
        {
        	// Compute all energy terms
            float dataEnergy = 0.0;
            computeDataEnergy(&dataEnergy, m_d_tsdfLiveDeform, m_d_tsdfGlobal, m_d_mask, m_gridW, m_gridH, m_gridD);
            std::cout<< "| Data Term Energy: " << dataEnergy;
            dataEnergyArr[itr] = dataEnergy;

            float levelSetEnergy = 0.0;
            computeLevelSetEnergy(&levelSetEnergy, m_d_sdfDxDeform, m_d_sdfDyDeform, m_d_sdfDzDeform,
                                                   m_d_mask, m_ws, m_tsdfGradScale, m_voxelSize,
                                                   m_gridW, m_gridH, m_gridD);
            std::cout<< "| Level Set Energy: " << levelSetEnergy;
            levelSetEnergyArr[itr] = levelSetEnergy;
            
            // Find the energy of the Killing term
            computeGradient(m_d_dux, m_d_duy, m_d_duz, m_d_deformationFieldU, m_gridW, m_gridH, m_gridD);
            computeGradient(m_d_dvx, m_d_dvy, m_d_dvz, m_d_deformationFieldV, m_gridW, m_gridH, m_gridD);
            computeGradient(m_d_dwx, m_d_dwy, m_d_dwz, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);

            float killingEnergy = 0.0;
            computeKillingEnergy(&killingEnergy, m_gamma,
                                  m_d_dux, m_d_duy, m_d_duz,
                                  m_d_dvx, m_d_dvy, m_d_dvz,
                                  m_d_dwx, m_d_dwy, m_d_dwz,
                                  m_d_mask, m_wk,
                                  m_gridW, m_gridH, m_gridD);
            std::cout<< "| Killing Energy: " << killingEnergy;
            killingEnergyArr[itr] = killingEnergy;

            totalEnergyArr[itr] = dataEnergyArr[itr] + levelSetEnergyArr[itr] + killingEnergyArr[itr];
            std::cout<< "| Total Energy: " << totalEnergyArr[itr];
        }

        // Interpolate current frame weights
        interpLiveWeights->interpolate3D(m_d_tsdfLiveWeightsDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);

        // Update new state of the deformation field
        timer.start();
        addArray(m_d_deformationFieldU, m_d_energyDu, -1.0*m_alpha, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeAddArray += timer.get();
    	m_nAddArray += 1;
    	timer.start();
        addArray(m_d_deformationFieldV, m_d_energyDv, -1.0*m_alpha, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeAddArray += timer.get();
    	m_nAddArray += 1;
    	timer.start();
        addArray(m_d_deformationFieldW, m_d_energyDw, -1.0*m_alpha, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeAddArray += timer.get();
    	m_nAddArray += 1;

        // Compute currentMaxVectorUpdate
		currentMaxVectorUpdate = 0.0;
		timer.start();
		computeMagnitude(m_d_magnitude, m_d_energyDu, m_d_energyDv, m_d_energyDw, m_gridW, m_gridH, m_gridD);
		timer.end();
        m_timeComputeMagnitude += timer.get();
    	m_nComputeMagnitude += 1;
		timer.start();
        findAbsMax(m_handle, &currentMaxVectorUpdate, m_d_magnitude, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeFindAbsMax += timer.get();
    	m_nFindAbsMax += 1;

        if(m_debugMode)
        {
        	std::cout << "| Last Max update: " << m_alpha * currentMaxVectorUpdate << std::endl;
        }

	} while ((m_alpha * currentMaxVectorUpdate) > m_maxVectorUpdateThreshold && itr < m_maxIterations);

	// Update TSDF Global using a weighted averaging scheme
	interpTSDFLive->interpolate3D(m_d_tsdfLiveDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
    interpLiveWeights->interpolate3D(m_d_tsdfLiveWeightsDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);

    timer.start();
    addWeightedArray(m_d_tsdfGlobal, m_d_tsdfGlobalWeights, m_d_tsdfGlobal, m_d_tsdfLiveDeform, m_d_tsdfGlobalWeights, m_d_tsdfLiveWeightsDeform, m_gridW, m_gridH, m_gridD);
    timer.end();
    m_timeAddWeightedArray += timer.get();
    m_nAddWeightedArray += 1;

    // Print wether convergence was achieved
    frameTimer.end();
    float currentFrameOptimizationTime = frameTimer.get();
    m_timeFramesOptimized += currentFrameOptimizationTime;
    m_nFramesOptimized += 1;
    if((m_alpha * currentMaxVectorUpdate) <= m_maxVectorUpdateThreshold)
    {
        std::cout << green << " GD Converged" << def << " (" << itr << " iterations, " << currentFrameOptimizationTime*1000.0f << "ms). Last max update: " << (m_alpha * currentMaxVectorUpdate) << std::endl;
    }
    else
    {
        std::cout << red << " GD Not Converged" << def << " (" << itr << " iterations, " << currentFrameOptimizationTime*1000.0f  << "ms). Last max update: " << (m_alpha * currentMaxVectorUpdate) << std::endl;
    }

    if(m_debugMode)
    {
        
        //plot the total and individual energy decay
        plotEnergy(dataEnergyArr, levelSetEnergyArr, killingEnergyArr, totalEnergyArr, itr,
                        "./bin/result/dataEnergy.txt", "./bin/result/levelSetEnergy.txt", "./bin/result/killingEnergy.txt", "./bin/result/totalEnergy.txt", "Energy",
                        tsdfLive->getFrameNumber(), m_gridW, m_gridH, m_gridD);
                        
        Vec3i volDim(m_gridW, m_gridH, m_gridD);
        Vec3f volSize(m_gridW*m_voxelSize, m_gridH*m_voxelSize, m_gridD*m_voxelSize);

        MarchingCubes mc(volDim, volSize);
        float* sdf = (float*)calloc(m_gridW*m_gridH*m_gridD, sizeof(float));
        float* sdfWeights = (float*)calloc(m_gridW*m_gridH*m_gridD, sizeof(float));

        cudaMemcpy(sdf, m_d_tsdfLiveDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(sdfWeights, m_d_tsdfLiveWeightsDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

        mc.computeIsoSurface(sdf, sdfWeights, tsdfLive->ptrColorR(), tsdfLive->ptrColorG(), tsdfLive->ptrColorB());
        std::string meshFilename = "./bin/result/mesh_warped_" + std::to_string(tsdfLive->getFrameNumber()) + ".ply";

        if (!mc.savePly(meshFilename))
        {
            std::cerr << "Could not save mesh!" << std::endl;
        }

        cudaMemcpy(sdf, m_d_tsdfLive, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(sdfWeights, m_d_tsdfLiveWeights, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

        mc.computeIsoSurface(sdf, sdfWeights, tsdfLive->ptrColorR(), tsdfLive->ptrColorG(), tsdfLive->ptrColorB());
        meshFilename = "./bin/result/mesh_original_" + std::to_string(tsdfLive->getFrameNumber()) + ".ply";
        if (!mc.savePly(meshFilename))
        {
            std::cerr << "Could not save mesh!" << std::endl;
        }

        delete[] sdf;
        delete[] sdfWeights;

        // Plot slices
        plotSlice(m_d_tsdfLive, m_gridD / 2, 2, "TSDF Live slice", 100, 100, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfGlobal, m_gridD / 2, 2, "TSDF Global slice", 100 + 4*m_gridW, 100, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfLiveDeform, m_gridD / 2, 2, "Warped TSDF Live", 100 + 8*m_gridW, 100, m_gridW, m_gridH, m_gridD);
        /*
        // Additional example plots
        plotSlice(m_d_tsdfLive, 50, 1, "Live TSDF Y", 100, 100 + 8*m_gridH, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfLive, m_gridW / 2, 0, "Live TSDF X", 100 + 8*m_gridW, 100 + 8*m_gridH, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfLiveDeform, m_gridW / 2, 0, "LiveDeform TSDF X", 100 + 4*m_gridW, 100 + 8*m_gridH, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfLive, m_gridW / 2, 0, "Live TSDF X", 100 + 8*m_gridW, 100 + 8*m_gridH, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfLiveDeform, m_gridW / 2, 0, "LiveDeform TSDF X", 100 + 4*m_gridW, 100 + 8*m_gridH, m_gridW, m_gridH, m_gridD);
        */
        // Plot the deformation for only one slice along the Z axis, so currently the W deformation field is not used
        //if(tsdfLive->getFrameNumber() >= 343)
        //{
        //    for (size_t slice = 0; slice < 80; ++slice)
        //    {
                plotVectorField(m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_d_tsdfLive, m_gridD/2,
                                "./bin/result/u.txt", "./bin/result/v.txt", "./bin/result/w.txt", "./bin/result/weights.txt", "Deformation_Field",
                                tsdfLive->getFrameNumber()/**100+slice*/, m_gridW, m_gridH, m_gridD);
        //    }
        //}
        cv::waitKey(30);
    }

	delete interpTSDFLive;
	delete interpTSDFDx;
	delete interpTSDFDy;
	delete interpTSDFDz;
	delete interpHessXX;
	delete interpHessXY;
	delete interpHessXZ;
	delete interpHessYY;
	delete interpHessYZ;
	delete interpHessZZ;
    delete interpLiveWeights;
}

void Optimizer::testIntermediateSteps(TSDFVolume* tsdfLive)
{
	// Initialize variable
	float* tsdfLiveGrid = tsdfLive->ptrTsdf();
	cudaMemcpy(m_d_tsdfLive, tsdfLiveGrid, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

	// Compute intermediate steps
	computeGradient(m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, m_d_tsdfLive, m_gridW, m_gridH, m_gridD);
	computeLaplacian(m_d_lapU, m_d_dux, 0, m_d_tsdfLive, m_gridW, m_gridH, m_gridD);
	computeHessian(m_d_hessXX, m_d_hessXY, m_d_hessXZ, m_d_hessYY, m_d_hessYZ, m_d_hessZZ, m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, m_gridW, m_gridH, m_gridD);
	
	// Perform interpolation
	float* d_tsdfDef = NULL;
	cudaMalloc(&d_tsdfDef, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	Interpolator *interptsdf = new Interpolator(m_d_tsdfLive, m_gridW, m_gridH, m_gridD);
	interptsdf->interpolate3D(d_tsdfDef, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);

	// Plot slices
	plotSlice(m_d_tsdfLive, m_gridD / 2, 2, "TSDF", 100 + 4*m_gridW, 100, m_gridW, m_gridH, m_gridD);
	plotSlice(d_tsdfDef, m_gridD / 2, 2, "TSDF Deform", 100 + 8*m_gridW, 100, m_gridW, m_gridH, m_gridD);
	plotSlice(m_d_sdfDx, m_gridD / 2, 2, "Grad X", 100 + 12*m_gridW, 100, m_gridW, m_gridH, m_gridD);
	plotSlice(m_d_lapU, m_gridD / 2, 2, "Laplacian U", 100 + m_gridW, 100 + 4*m_gridW, m_gridW, m_gridH, m_gridD);
	plotSlice(m_d_hessXX, m_gridD / 2, 2, "Hessian XX", 100 + 4*m_gridW, 100 + 4*m_gridW, m_gridW, m_gridH, m_gridD);
	cv::waitKey();

	// Destroy used object
	delete interptsdf;
}


void Optimizer::computeGradient(float* gradOutX, float* gradOutY, float* gradOutZ, const float* gridIn, int w, int h, int d)
{
	// Compute gradients for SDF live grid
	computeGradient3DX(gradOutX, gridIn, w, h, d);
	computeGradient3DY(gradOutY, gridIn, w, h, d);
	computeGradient3DZ(gradOutZ, gridIn, w, h, d);
}

void Optimizer::computeDivergence(float* divOut, const float* gridInU, const float* gridInV, const float* gridInW, int w, int h, int d)
{
	// Compute gradients for the deformation field
	computeGradient3DX(m_d_du, gridInU, w, h, d);
	computeGradient3DY(m_d_dv, gridInV, w, h, d);
	computeGradient3DZ(m_d_dw, gridInW, w, h, d);

	// Sum the three gradient components
	computeDivergence3D(divOut, m_d_du, m_d_dv, m_d_dw, w, h, d);
}

void Optimizer::sumGradients(float* divOut, const float* duxIn, const float* dvyIn, const float* dwzIn, int w, int h, int d)
{
	// Sum the three gradient components
	computeDivergence3D(divOut, duxIn, dvyIn, dwzIn, w, h, d);
}

void Optimizer::computeLaplacian(float* lapOut, float* dOut, const size_t dOutComponent, const float* deformationIn, int w, int h, int d)
{
	if (dOutComponent == 0)
	{
		computeGradient3DX(dOut, deformationIn, w, h, d);
		computeGradient3DY(m_d_dfy, deformationIn, w, h, d);
		computeGradient3DZ(m_d_dfz, deformationIn, w, h, d);
		computeDivergence(lapOut, dOut, m_d_dfy, m_d_dfz, w, h, d);
	}
	else if (dOutComponent == 1)
	{
		computeGradient3DX(m_d_dfx, deformationIn, w, h, d);
		computeGradient3DY(dOut, deformationIn, w, h, d);
		computeGradient3DZ(m_d_dfz, deformationIn, w, h, d);
		computeDivergence(lapOut, m_d_dfx, dOut, m_d_dfz, w, h, d);
	}
	else
	{
		computeGradient3DX(m_d_dfx, deformationIn, w, h, d);
		computeGradient3DY(m_d_dfy, deformationIn, w, h, d);
		computeGradient3DZ(dOut, deformationIn, w, h, d);
		computeDivergence(lapOut, m_d_dfx, m_d_dfy, dOut, w, h, d);
	}
}

void Optimizer::computeHessian(float* hessOutXX, float* hessOutXY, float* hessOutXZ, float* hessOutYY, float* hessOutYZ, float* hessOutZZ, const float* gradX, const float* gradY, const float* gradZ, int w, int h, int d)
{
	computeGradient3DX(hessOutXX, gradX, w, h, d);
    computeGradient3DY(hessOutXY, gradX, w, h, d);
	computeGradient3DX(hessOutXZ, gradZ, w, h, d);
	computeGradient3DY(hessOutYY, gradY, w, h, d);
    computeGradient3DY(hessOutYZ, gradZ, w, h, d);
	computeGradient3DZ(hessOutZZ, gradZ, w, h, d);
}
