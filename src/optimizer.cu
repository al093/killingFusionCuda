#include "optimizer.cuh"

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Geometry>
#include <cuda_runtime.h>
#include "gradient.cuh"
#include "convolution.cuh"
#include "divergence.cuh"
#include "helper.cuh"
#include "energy.cuh"
#include "energyDerivatives.cuh"
#include "interpolator.cuh"
#include "reduction.cuh"
#include "magnitude.cuh"
#include "visualization.cuh"
#include "marching_cubes.h"
#include "color.h"

#include <opencv2/highgui/highgui.hpp>

Color::Modifier red(Color::FG_RED);
Color::Modifier green(Color::FG_GREEN);
Color::Modifier def(Color::FG_DEFAULT);

Optimizer::Optimizer(TSDFVolume* tsdfGlobal, float* initialDeformationU, float* initialDeformationV, float* initialDeformationW, 
                     const float alpha, const float wk, const float ws, const size_t maxIterations, const float voxelSize,
                     const bool debugMode, const size_t gridW, const size_t gridH, const size_t gridD) :
	m_tsdfGlobal(tsdfGlobal),
    m_deformationFieldU(initialDeformationU),
    m_deformationFieldV(initialDeformationV),
    m_deformationFieldW(initialDeformationW),
    m_alpha(alpha),
	m_wk(wk),
	m_ws(ws),
	m_maxIterations(maxIterations),
	m_voxelSize(voxelSize),
	m_debugMode(debugMode),
	m_gridW(gridW), 
	m_gridH(gridH),
	m_gridD(gridD)
{
	m_maxVectorUpdateThreshold = 0.1 / (m_voxelSize * 1000.0);
    allocateMemoryInDevice();
	copyArraysToDevice();
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
    cudaMemset(m_d_mask, 0, (m_gridW * m_gridH * m_gridD) * sizeof(bool)); CUDA_CHECK;

    // Allocate and initialize to zero the memory for the gradients of energy
    cudaMalloc(&m_d_energyDu, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_energyDv, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_energyDw, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMemset(m_d_energyDu, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMemset(m_d_energyDv, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMemset(m_d_energyDw, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    
    //used by the compute Divergence function
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
	// Derivative kernels
	cudaMemcpy(m_d_kernelDx, m_kernelDxCentralDiff, (27) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_kernelDy, m_kernelDyCentralDiff, (27) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_kernelDz, m_kernelDzCentralDiff, (27) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	// Deformation fields
	cudaMemcpy(m_d_deformationFieldU, m_deformationFieldU, (m_gridW * m_gridH * m_gridD)  * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_deformationFieldV, m_deformationFieldV, (m_gridW * m_gridH * m_gridD)  * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_deformationFieldW, m_deformationFieldW, (m_gridW * m_gridH * m_gridD)  * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
}

Optimizer::~Optimizer()
{
    //TODO Really Important for our GRADES! Free all the allocated memory.
	cudaFree(m_d_deformationFieldU); CUDA_CHECK;
	cudaFree(m_d_deformationFieldV); CUDA_CHECK;
	cudaFree(m_d_deformationFieldW); CUDA_CHECK;
	cudaFree(m_d_kernelDx); CUDA_CHECK;
	cudaFree(m_d_kernelDy); CUDA_CHECK;
	cudaFree(m_d_kernelDz); CUDA_CHECK;
	cudaFree(m_d_div); CUDA_CHECK;
    cudaFree(m_d_dvx); CUDA_CHECK;
    cudaFree(m_d_dvy); CUDA_CHECK;
    cudaFree(m_d_dvz); CUDA_CHECK;
    cudaFree(m_d_dwx); CUDA_CHECK;
    cudaFree(m_d_dwy); CUDA_CHECK;
    cudaFree(m_d_dwz); CUDA_CHECK;
    cudaFree(m_d_du); CUDA_CHECK;
    cudaFree(m_d_dv); CUDA_CHECK;
    cudaFree(m_d_dw); CUDA_CHECK;
    cudaFree(m_d_dfx); CUDA_CHECK;
    cudaFree(m_d_dfy); CUDA_CHECK;
    cudaFree(m_d_dfz); CUDA_CHECK;
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
	std::cout << "Mean running times (ms): " << std::endl;
	std::cout << "- computeGradient: " << 1000*m_timeComputeGradient / (float)m_nComputeGradient << " (" << m_nComputeGradient << " iterations)" <<std::endl;
	std::cout << "- computeHessian: " << 1000*m_timeComputeHessian / (float)m_nComputeHessian << " (" << m_nComputeHessian << " iterations)" <<std::endl;
	std::cout << "- computeLaplacian: " << 1000*m_timeComputeLaplacian / (float)m_nComputeLaplacian << " (" << m_nComputeLaplacian << " iterations)" <<std::endl;
	std::cout << "- computeDivergence: " << 1000*m_timeComputeDivergence / (float)m_nComputeDivergence << " (" << m_nComputeDivergence << " iterations)" <<std::endl;
	std::cout << "- computeDataTermDeritavie: " << 1000*m_timeComputeDataTermDerivative / (float)m_nComputeDataTermDerivative << " (" << m_nComputeDataTermDerivative << " iterations)" <<std::endl;
	std::cout << "- computeLevelSetDerivative: " << 1000*m_timeComputeLevelSetDerivative / (float)m_nComputeLevelSetDerivative << " (" << m_nComputeLevelSetDerivative << " iterations)" <<std::endl;
	std::cout << "- computeMotionRegularizerDerivatie: " << 1000*m_timeComputeMotionRegularizerDerivative / (float)m_nComputeMotionRegularizerDerivative << " (" << m_nComputeMotionRegularizerDerivative << " iterations)" <<std::endl;
	std::cout << "- addArray: " << 1000*m_timeAddArray / (float)m_nAddArray << " (" << m_nAddArray << " iterations)" <<std::endl;
	std::cout << "- computeMagnitude: " << 1000*m_timeComputeMagnitude / (float)m_nComputeMagnitude << " (" << m_nComputeMagnitude << " iterations)" <<std::endl;
	std::cout << "- findAbsMax: " << 1000*m_timeFindAbsMax / (float)m_nFindAbsMax << " (" << m_nFindAbsMax << " iterations)" <<std::endl;
	std::cout << "- addWeightedArray: " << 1000*m_timeAddWeightedArray / (float)m_nAddWeightedArray << " (" << m_nAddWeightedArray << " iterations)" <<std::endl;
}

void Optimizer::optimize(TSDFVolume* tsdfLive)
{
	// Initialize variables
	float currentMaxVectorUpdate = 0.01;
	float* tsdfLiveGrid = tsdfLive->ptrTsdf();
	float* tsdfLiveWeights = tsdfLive->ptrTsdfWeights();
	Timer timer;
	// Copy arrays from host to device
	cudaMemcpy(m_d_tsdfLive, tsdfLiveGrid, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_tsdfLiveWeights, tsdfLiveWeights, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

	// Compute gradient of tsdfLive
    timer.start();
	computeGradient(m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, m_d_tsdfLive, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
	timer.end();
    m_timeComputeGradient += timer.get();
    m_nComputeGradient += 1;


    // Compute the mask for this tsdfLive


	// Compute hessian of tsdfLive
	timer.start();
	computeHessian(m_d_hessXX, m_d_hessXY, m_d_hessXZ, m_d_hessYY, m_d_hessYZ, m_d_hessZZ, m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
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

	/* IN ORDER TO COMPUTE ENERGIES: 	PSI is m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW
										PHIglobal is m_d_tsdfGlobal			PHIn is m_d_tsdfLive
										GRADIENT OF PHIn IS: m_d_sdfDx, m_d_sdfDy, m_d_sdfDz
										HESSIAN OF PHIn IS: m_d_hessXX, m_d_hessXY, m_d_hessXZ, m_d_hessYY, m_d_hessYZ, m_d_hessZZ
	*/

    // Compute the mask for this tsdfLive
    computeMask(m_d_mask, m_d_tsdfLive, m_gridW, m_gridH, m_gridD);

    if (m_debugMode) std::cout<< "Deforming SDF..." << std::endl;
    size_t itr = 0;

    do
	{
        ++itr;
        if(m_debugMode) std::cout << "\nGD itr num: " << itr;
        
        // Interpolate TSDF Live Frame
        interpTSDFLive->interpolate3D(m_d_tsdfLiveDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);

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

        // Set m_d_energyDu/v/w intially to zero
        cudaMemset(m_d_energyDu, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        cudaMemset(m_d_energyDv, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        cudaMemset(m_d_energyDw, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        
        // Compute data term derivatives
        timer.start();
        computeDataTermDerivative(m_d_energyDu, m_d_energyDv, m_d_energyDw,
                                  m_d_tsdfLiveDeform, m_d_tsdfGlobal,
                                  m_d_sdfDxDeform, m_d_sdfDyDeform, m_d_sdfDzDeform, m_d_mask,
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
                                  m_d_mask, m_ws,
                                  m_gridW, m_gridH, m_gridD);
		timer.end();
        m_timeComputeLevelSetDerivative += timer.get();
    	m_nComputeLevelSetDerivative += 1;

        //compute laplacians of the deformation field
        timer.start();
        computeLapacian(m_d_lapU, m_d_deformationFieldU, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeLaplacian += timer.get();
    	m_nComputeLaplacian += 1;
    	timer.start();
        computeLapacian(m_d_lapV, m_d_deformationFieldV, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeLaplacian += timer.get();
    	m_nComputeLaplacian += 1;
    	timer.start();
        computeLapacian(m_d_lapW, m_d_deformationFieldW, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeLaplacian += timer.get();
    	m_nComputeLaplacian += 1;

        //compute divergence and then the gradients of the divergence of deformation field
        timer.start();
        computeDivergence(m_d_div, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeDivergence += timer.get();
    	m_nComputeDivergence += 1;
    	timer.start();
        computeGradient(m_d_divX, m_d_divY, m_d_divZ, m_d_div, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeComputeGradient += timer.get();
    	m_nComputeGradient += 1;

        float gamma = 0.05;
        
        //compute motion regularizer derivative
        timer.start();
        computeMotionRegularizerDerivative(m_d_energyDu, m_d_energyDv, m_d_energyDw,
                                           m_d_lapU, m_d_lapV, m_d_lapW,
                                           m_d_divX, m_d_divY, m_d_divZ,
                                           m_d_mask, m_wk, gamma,
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

            float levelSetEnergy = 0.0;
            computeLevelSetEnergy(&levelSetEnergy, m_d_sdfDxDeform, m_d_sdfDyDeform, m_d_sdfDzDeform, m_d_mask, m_gridW, m_gridH, m_gridD);
            std::cout<< "| Level Set Energy: " << levelSetEnergy;

            // Find the energy of the Killing term
            computeGradient(m_d_dux, m_d_duy, m_d_duz, m_d_deformationFieldU, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
            computeGradient(m_d_dvx, m_d_dvy, m_d_dvz, m_d_deformationFieldV, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
            computeGradient(m_d_dwx, m_d_dwy, m_d_dwz, m_d_deformationFieldW, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);

            float killingEnergy = 0.0;
            computeKillingEnergy(&killingEnergy, gamma,
                                  m_d_dux, m_d_duy, m_d_duz,
                                  m_d_dvx, m_d_dvy, m_d_dvz,
                                  m_d_dwx, m_d_dwy, m_d_dwz,
                                  m_d_mask,
                                  m_gridW, m_gridH, m_gridD);
            std::cout<< "| Killing Energy: " << killingEnergy;
        }

        //multiply by weights
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
        findAbsMax(&currentMaxVectorUpdate, m_d_magnitude, m_gridW, m_gridH, m_gridD);
        timer.end();
        m_timeFindAbsMax += timer.get();
        m_nFindAbsMax += 1;

        if(m_debugMode) std::cout<<"| Max update Norm: " << m_alpha * currentMaxVectorUpdate << std::endl;

    } while ((m_alpha * currentMaxVectorUpdate) > m_maxVectorUpdateThreshold && itr < m_maxIterations);

    //print if convergence was achieved
    if((m_alpha * currentMaxVectorUpdate) <= m_maxVectorUpdateThreshold )
        std::cout<< green << " GD Converged" << def << " Last max delta norm: " << (m_alpha * currentMaxVectorUpdate);
    else
        std::cout<< red << " GD Not Converged" << def << " Last max delta norm: " << (m_alpha * currentMaxVectorUpdate);;

    // Update TSDF Global using a weighted averaging scheme
    interpTSDFLive->interpolate3D(m_d_tsdfLiveDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
    interpLiveWeights->interpolate3D(m_d_tsdfLiveWeightsDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);

    timer.start();
    addWeightedArray(m_d_tsdfGlobal, m_d_tsdfGlobalWeights, m_d_tsdfGlobal, m_d_tsdfLiveDeform, m_d_tsdfGlobalWeights, m_d_tsdfLiveWeightsDeform, m_gridW, m_gridH, m_gridD);
    timer.end();
    m_timeAddWeightedArray += timer.get();
    m_nAddWeightedArray += 1;

    
    if(m_debugMode)
    {
        //this will store the deformed mesh and the original mesh for every frame
        Vec3i volDim(m_gridW, m_gridH, m_gridD);
        Vec3f volSize(m_gridW*m_voxelSize, m_gridH*m_voxelSize, m_gridD*m_voxelSize);
        MarchingCubes mc(volDim, volSize);
        
        float* sdf = (float*)calloc(m_gridW*m_gridH*m_gridD, sizeof(float));
        float* sdfWeights = (float*)calloc(m_gridW*m_gridH*m_gridD, sizeof(float));
        cudaMemcpy(sdf, m_d_tsdfLiveDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(sdfWeights, m_d_tsdfLiveWeightsDeform, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        mc.computeIsoSurface(sdf, sdfWeights, tsdfLive->ptrColorR(), tsdfLive->ptrColorG(), tsdfLive->ptrColorB());
        std::string meshFilename = "./bin/result/mesh_warped_" + std::to_string(tsdfLive->getFrameNumber()) + ".ply";
        if (!mc.savePly(meshFilename)) std::cerr << "Could not save mesh!" << std::endl;

        cudaMemcpy(sdf, m_d_tsdfLive, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(sdfWeights, m_d_tsdfLiveWeights, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        mc.computeIsoSurface(sdf, sdfWeights, tsdfLive->ptrColorR(), tsdfLive->ptrColorG(), tsdfLive->ptrColorB());
        meshFilename = "./bin/result/mesh_original_" + std::to_string(tsdfLive->getFrameNumber()) + ".ply";
        if (!mc.savePly(meshFilename)) std::cerr << "Could not save mesh!" << std::endl;

        delete[] sdf;
        delete[] sdfWeights;

        plotSlice(m_d_tsdfLive, m_gridD / 2, 2, "TSDF Live slice", 100, 100, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfGlobal, m_gridD / 2, 2, "TSDF Global slice", 100 + 4*m_gridW, 100, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfLiveDeform, m_gridD / 2, 2, "Warped TSDF Live", 100 + 8*m_gridW, 100, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfLiveWeights, m_gridD / 2, 2, "Live weights", 100, 100 + 4*m_gridH, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfGlobalWeights, m_gridD / 2, 2, "Global weights", 100 + 4*m_gridW, 100 + 4*m_gridH, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfLive, 50, 1, "Live TSDF Y", 100, 100 + 8*m_gridH, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfLive, m_gridW / 2 + 5, 0, "Live TSDF X", 100, 100 + 12*m_gridH, m_gridW, m_gridH, m_gridD);
        
        //plots the deformation for only one slice (the middle one) along the Z axis, so currently the W deformation field is not used.
        plotVectorField(m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_d_tsdfLive, m_gridD / 2,
                        "./bin/result/u.txt", "./bin/result/v.txt", "./bin/result/w.txt", "./bin/result/weights.txt", "deformation_field",
                        tsdfLive->getFrameNumber(), m_gridW, m_gridH, m_gridD);

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

void Optimizer::optimizeTest(TSDFVolume* tsdfLive)
{
//Dummy Test function.
}


void Optimizer::computeGradient(float* gradOutX, float* gradOutY, float* gradOutZ, const float* gridIn, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
	// Compute gradients for SDF live grid
	computeGradient3DX(gradOutX, gridIn, w, h, d);
	computeGradient3DY(gradOutY, gridIn, w, h, d);
	computeGradient3DZ(gradOutZ, gridIn, w, h, d);
}

void Optimizer::computeDivergence(float* divOut, const float* gridInU, const float* gridInV, const float* gridInW, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
    // Compute gradients for the deformation field
    computeGradient3DX(m_d_du, gridInU, w, h, d);
    computeGradient3DY(m_d_dv, gridInV, w, h, d);
    computeGradient3DZ(m_d_dw, gridInW, w, h, d);

    // Sum the three gradient components
    computeDivergence3DCuda(divOut, m_d_du, m_d_dv, m_d_dw, w, h, d);
}

void Optimizer::computeLapacian(float* lapOut, const float* deformationIn, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
    computeGradient3DX(m_d_dfx, deformationIn, w, h, d);
    computeGradient3DY(m_d_dfy, deformationIn, w, h, d);
    computeGradient3DZ(m_d_dfz, deformationIn, w, h, d);
    computeDivergence(lapOut, m_d_dux, m_d_duy, m_d_duz, kernelDx, kernelDy, kernelDz, 1, w, h, d);
}

void Optimizer::computeHessian(float* hessOutXX, float* hessOutXY, float* hessOutXZ, float* hessOutYY, float* hessOutYZ, float* hessOutZZ, const float* gradX, const float* gradY, const float* gradZ, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
	computeGradient3DX(hessOutXX, gradX, w, h, d);
    computeGradient3DY(hessOutXY, gradX, w, h, d);
	computeGradient3DX(hessOutXZ, gradZ, w, h, d);
	computeGradient3DY(hessOutYY, gradY, w, h, d);
    computeGradient3DY(hessOutYZ, gradZ, w, h, d);
	computeGradient3DZ(hessOutZZ, gradZ, w, h, d);
}
