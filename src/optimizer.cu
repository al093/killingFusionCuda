#include "optimizer.cuh"

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Geometry>
#include <cuda_runtime.h>
#include "convolution.cuh"
#include "divergence.cuh"
#include "helper.cuh"
#include "energy.cuh"
#include "energyDerivatives.cuh"
#include "interpolator.cuh"
#include "reduction.cuh"
#include "magnitude.cuh"
#include "visualization.cuh"

#include <opencv2/highgui/highgui.hpp>

Optimizer::Optimizer(TSDFVolume* tsdfGlobal, float* initialDeformationU, float* initialDeformationV, float* initialDeformationW, 
                     const float alpha, const float wk, const float ws, const size_t maxIterations, const size_t gridW, 
                     const size_t gridH, const size_t gridD, const bool debugMode) :
	m_tsdfGlobal(tsdfGlobal),
    m_deformationFieldU(initialDeformationU),
    m_deformationFieldV(initialDeformationV),
    m_deformationFieldW(initialDeformationW),
    m_alpha(alpha),
	m_wk(wk),
	m_ws(ws),
	m_maxIterations(maxIterations),
	m_gridW(gridW), 
	m_gridH(gridH),
	m_gridD(gridD),
    m_debugMode(debugMode)
{
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

    // Allocate and initialize to zero the memory for the gradients of energy
    cudaMalloc(&m_d_energyDu, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_energyDv, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_energyDw, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMemset(m_d_energyDu, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMemset(m_d_energyDv, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMemset(m_d_energyDw, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
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
}

void Optimizer::getTSDFGlobalPtr(float* tsdfGlobalPtr)
{
	cudaMemcpy(tsdfGlobalPtr, m_d_tsdfGlobal, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
}

void Optimizer::getTSDFGlobalWeightsPtr(float* tsdfGlobalWeightsPtr)
{
	cudaMemcpy(tsdfGlobalWeightsPtr, m_d_tsdfGlobalWeights, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
}

void Optimizer::optimize(TSDFVolume* tsdfLive)
{
	// Initialize variables
	float currentMaxVectorUpdate = 0.01;
	float* tsdfLiveGrid = tsdfLive->ptrTsdf();
	float* tsdfLiveWeights = tsdfLive->ptrTsdfWeights();
	// Copy arrays from host to device
	cudaMemcpy(m_d_tsdfLive, tsdfLiveGrid, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(m_d_tsdfLiveWeights, tsdfLiveWeights, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

	// Compute gradient of tsdfLive
	computeGradient(m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, m_d_tsdfLive, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);

	// Compute hessian of tsdfLive
	computeHessian(m_d_hessXX, m_d_hessXY, m_d_hessXZ, m_d_hessYY, m_d_hessYZ, m_d_hessZZ, m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);

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

    if (m_debugMode)
    {
        std::cout<< "\nDeforming SDF...\n";
    }
    int itr = 0;
	do
	{
        std::cout<<"\nGD itr num: " << itr++;
        
		// Interpolate TSDF Live Frame (EXAMPLE: HOW TO INTERPOLATE PHIn DEFORMED BY PSI)
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
        computeDataTermDerivative(m_d_energyDu, m_d_energyDv, m_d_energyDw,
                                  m_d_tsdfLiveDeform, m_d_tsdfGlobal,
                                  m_d_sdfDxDeform, m_d_sdfDyDeform, m_d_sdfDzDeform,
                                  m_gridW, m_gridH, m_gridD);

        // Add the derivatives from the levelSet Derivatives with a scalar constant wk
        computeLevelSetDerivative(m_d_energyDu, m_d_energyDv, m_d_energyDw,
                                  m_d_hessXXDeform, m_d_hessXYDeform, m_d_hessXZDeform,
                                  m_d_hessYYDeform, m_d_hessYZDeform, m_d_hessZZDeform,
                                  m_d_sdfDxDeform, m_d_sdfDyDeform, m_d_sdfDzDeform,
                                  m_ws,
                                  m_gridW, m_gridH, m_gridD);

        //compute laplacians of the deformation field
        computeLapacian(m_d_lapU, m_d_deformationFieldU, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
        computeLapacian(m_d_lapV, m_d_deformationFieldV, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
        computeLapacian(m_d_lapW, m_d_deformationFieldW, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);

        //compute divergence and then the gradients of the divergence of deformation field
        computeDivergence(m_d_div, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
        computeGradient(m_d_divX, m_d_divY, m_d_divZ, m_d_div, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
        
        //TODO make gamma (the killing field weight) a member variable
        float gamma = 0.1;
        
        //compute motion regularizer derivative
        computeMotionRegularizerDerivative(m_d_energyDu, m_d_energyDv, m_d_energyDw,
                                           m_d_lapU, m_d_lapV, m_d_lapW,
                                           m_d_divX, m_d_divY, m_d_divZ,
                                           m_wk, gamma,
                                           m_gridW, m_gridH, m_gridD);
        if (m_debugMode)
        {
        	// Compute all energy derivatives split
        	/*interpLiveWeights->interpolate3D(m_d_tsdfLiveWeightsDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        	interpTSDFLive->interpolate3D(m_d_tsdfLiveDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        	plotSlice(m_d_tsdfLiveDeform, m_gridD / 2, "TSDF Live slice", 100, 100, m_gridW, m_gridH, m_gridD);
        	cv::waitKey(30);
        	// Gradients
        	plotVectorField(m_d_sdfDxDeform, m_d_sdfDyDeform, m_d_sdfDzDeform, m_d_tsdfLiveWeightsDeform, 40, 
        				    "./bin/result/gradX.txt", "./bin/result/gradY.txt", "./bin/result/gradZ.txt", "./bin/result/weights.txt", "gradients",
        					m_gridW, m_gridH, m_gridD);
        	// Data energy derivative
        	float* d_energyDatalDu = NULL, * d_energyDatalDv = NULL, * d_energyDatalDw = NULL;
        	cudaMalloc(&d_energyDatalDu, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
			cudaMalloc(&d_energyDatalDv, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
			cudaMalloc(&d_energyDatalDw, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	cudaMemset(d_energyDatalDu, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	cudaMemset(d_energyDatalDv, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	cudaMemset(d_energyDatalDw, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	computeDataTermDerivative(d_energyDatalDu, d_energyDatalDv, d_energyDatalDw,
                                  m_d_tsdfLiveDeform, m_d_tsdfGlobal,
                                  m_d_sdfDxDeform, m_d_sdfDyDeform, m_d_sdfDzDeform,
                                  m_gridW, m_gridH, m_gridD);
			plotVectorField(d_energyDatalDu, d_energyDatalDv, d_energyDatalDw, m_d_tsdfLiveWeightsDeform, 40, 
        				    "./bin/result/dDataU.txt", "./bin/result/dDataV.txt", "./bin/result/dDataW.txt", "./bin/result/weights.txt", "d_data",
        					m_gridW, m_gridH, m_gridD);

        	cudaFree(d_energyDatalDu);
        	cudaFree(d_energyDatalDv);
        	cudaFree(d_energyDatalDw);    
        	// Level set energy derivative
        	float* d_energyLevelDu = NULL, * d_energyLevelDv = NULL, * d_energyLevelDw = NULL;
        	cudaMalloc(&d_energyLevelDu, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
			cudaMalloc(&d_energyLevelDv, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
			cudaMalloc(&d_energyLevelDw, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	cudaMemset(d_energyLevelDu, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	cudaMemset(d_energyLevelDv, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	cudaMemset(d_energyLevelDw, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	computeLevelSetDerivative(d_energyLevelDu, d_energyLevelDv, d_energyLevelDw,
                                  m_d_hessXXDeform, m_d_hessXYDeform, m_d_hessXZDeform,
                                  m_d_hessYYDeform, m_d_hessYZDeform, m_d_hessZZDeform,
                                  m_d_sdfDxDeform, m_d_sdfDyDeform, m_d_sdfDzDeform,
                                  m_ws,
                                  m_gridW, m_gridH, m_gridD);
        	plotVectorField(d_energyLevelDu, d_energyLevelDv, d_energyLevelDw, m_d_tsdfLiveWeightsDeform, 40, 
        				    "./bin/result/dLevelU.txt", "./bin/result/dLevelV.txt", "./bin/result/dLevelW.txt", "./bin/result/weights.txt", "d_level",
        					m_gridW, m_gridH, m_gridD);
        	cudaFree(d_energyLevelDu);
        	cudaFree(d_energyLevelDv);
        	cudaFree(d_energyLevelDw);      
        	// Killing energy derivative
        	float* d_energyKillingDu = NULL, * d_energyKillingDv = NULL, * d_energyKillingDw = NULL;
        	cudaMalloc(&d_energyKillingDu, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
			cudaMalloc(&d_energyKillingDv, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
			cudaMalloc(&d_energyKillingDw, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	cudaMemset(d_energyKillingDu, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	cudaMemset(d_energyKillingDv, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	cudaMemset(d_energyKillingDw, 0, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
        	computeMotionRegularizerDerivative(d_energyKillingDu, d_energyKillingDv, d_energyKillingDw,
                                           m_d_lapU, m_d_lapV, m_d_lapW,
                                           m_d_divX, m_d_divY, m_d_divZ,
                                           m_wk, gamma,
                                           m_gridW, m_gridH, m_gridD);
        	plotVectorField(d_energyKillingDu, d_energyKillingDv, d_energyKillingDw, m_d_tsdfLiveWeightsDeform, 40, 
        				    "./bin/result/dKillingU.txt", "./bin/result/dKillingV.txt", "./bin/result/dKillingW.txt", "./bin/result/weights.txt", "d_killing",
        					m_gridW, m_gridH, m_gridD);
        	cudaFree(d_energyKillingDu);
        	cudaFree(d_energyKillingDv);
        	cudaFree(d_energyKillingDw);*/


        	// Compute all energy terms
            float dataEnergy = 0.0;
            computeDataEnergy(&dataEnergy, m_d_tsdfLiveDeform, m_d_tsdfGlobal, m_gridW, m_gridH, m_gridD);
            std::cout<< "| Data Term Energy: " << dataEnergy;

            float levelSetEnergy = 0.0;
            computeLevelSetEnergy(&levelSetEnergy, m_d_sdfDxDeform, m_d_sdfDyDeform, m_d_sdfDzDeform, m_gridW, m_gridH, m_gridD);
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
                                  m_gridW, m_gridH, m_gridD);
            std::cout<< "| Killing Energy: " << killingEnergy;
        }

        //multiply by weights
        interpLiveWeights->interpolate3D(m_d_tsdfLiveWeightsDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
        //multiplyArrays(m_d_energyDu, m_d_energyDu, m_d_tsdfLiveWeightsDeform, m_gridW, m_gridH, m_gridD);
        //multiplyArrays(m_d_energyDv, m_d_energyDv, m_d_tsdfLiveWeightsDeform, m_gridW, m_gridH, m_gridD);
        //multiplyArrays(m_d_energyDw, m_d_energyDw, m_d_tsdfLiveWeightsDeform, m_gridW, m_gridH, m_gridD);

        // Update new state of the deformation field
        addArray(m_d_deformationFieldU, m_d_energyDu, -1.0*m_alpha, m_gridW, m_gridH, m_gridD);
        addArray(m_d_deformationFieldV, m_d_energyDv, -1.0*m_alpha, m_gridW, m_gridH, m_gridD);
        addArray(m_d_deformationFieldW, m_d_energyDw, -1.0*m_alpha, m_gridW, m_gridH, m_gridD);

        // Compute currentMaxVectorUpdate
		currentMaxVectorUpdate = 0.0;
		computeMagnitude(m_d_magnitude, m_d_energyDu, m_d_energyDv, m_d_energyDw, m_gridW, m_gridH, m_gridD);
		//thresholdArray(m_d_magnitude, m_d_tsdfLiveWeightsDeform, 0.5, m_gridW, m_gridH, m_gridD);  // TEST
        findAbsMax(&currentMaxVectorUpdate, m_d_magnitude, m_gridW, m_gridH, m_gridD);

        std::cout<<"| Abs Max update: " << m_alpha * currentMaxVectorUpdate << std::endl;
        //if(m_debugMode) plotDeformation(m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, 40, m_gridW, m_gridH, m_gridD);


	} while ((m_alpha * currentMaxVectorUpdate) > MAX_VECTOR_UPDATE_THRESHOLD && itr < m_maxIterations);
	
	// TEST
	interpLiveWeights->interpolate3D(m_d_tsdfLiveWeightsDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
    /*thresholdArray(m_d_deformationFieldU, m_d_tsdfLiveWeightsDeform, 0.5, m_gridW, m_gridH, m_gridD);
    thresholdArray(m_d_deformationFieldV, m_d_tsdfLiveWeightsDeform, 0.5, m_gridW, m_gridH, m_gridD);
    thresholdArray(m_d_deformationFieldW, m_d_tsdfLiveWeightsDeform, 0.5, m_gridW, m_gridH, m_gridD);*/
    // END-TEST


	// Update TSDF Global using a weighted averaging scheme
	interpTSDFLive->interpolate3D(m_d_tsdfLiveDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);
    interpLiveWeights->interpolate3D(m_d_tsdfLiveWeightsDeform, m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_gridW, m_gridH, m_gridD);

    //thresholdArray(m_d_tsdfLiveDeform, m_d_tsdfLiveWeightsDeform, 0.5, m_gridW, m_gridH, m_gridD);
    addWeightedArray(m_d_tsdfGlobal, m_d_tsdfGlobalWeights, m_d_tsdfGlobal, m_d_tsdfLiveDeform, m_d_tsdfGlobalWeights, m_d_tsdfLiveWeightsDeform, m_gridW, m_gridH, m_gridD);

    if (m_debugMode)
    {
        plotSlice(m_d_tsdfLive, m_gridD / 2, "TSDF Live slice", 100, 100, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfGlobal, m_gridD / 2, "TSDF Global slice", 100 + 4*m_gridW, 100, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfLiveDeform, m_gridD / 2, "Warped TSDF Live", 100 + 8*m_gridW, 100, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfLiveWeights, m_gridD / 2, "Live weights", 100, 100 + 4*m_gridH, m_gridW, m_gridH, m_gridD);
        plotSlice(m_d_tsdfGlobalWeights, m_gridD / 2, "Global weights", 100 + 4*m_gridW, 100 + 4*m_gridH, m_gridW, m_gridH, m_gridD);
        //plots the deformation for only one slice along the Z axis, so currently the W deformation fild is not used.
        plotVectorField(m_d_deformationFieldU, m_d_deformationFieldV, m_d_deformationFieldW, m_d_tsdfLiveWeightsDeform, 40, 
        				"./bin/result/u.txt", "./bin/result/v.txt", "./bin/result/w.txt", "./bin/result/weights.txt", "deformation_field",
        				m_gridW, m_gridH, m_gridD);
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
	computeLapacian(m_d_lapU, m_d_tsdfLive, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
	computeHessian(m_d_hessXX, m_d_hessXY, m_d_hessXZ, m_d_hessYY, m_d_hessYZ, m_d_hessZZ, m_d_sdfDx, m_d_sdfDy, m_d_sdfDz, m_d_kernelDx, m_d_kernelDy, m_d_kernelDz, 1, m_gridW, m_gridH, m_gridD);
	float* tsdfLiveGridDef = new float[m_gridW * m_gridH * m_gridD];
	float* gradX = new float[m_gridW * m_gridH * m_gridD];
	float* lapU = new float[m_gridW * m_gridH * m_gridD];
	float* hessXX = new float[m_gridW * m_gridH * m_gridD];
	cudaMemcpy(gradX, m_d_sdfDx, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
	cudaMemcpy(lapU, m_d_lapU, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
	cudaMemcpy(hessXX, m_d_hessXX, (m_gridW * m_gridH * m_gridD) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
	// Interpolate
	float* d_tsdfDef = NULL;
	cudaMalloc(&d_tsdfDef, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
	Interpolator *interptsdf = new Interpolator(m_d_tsdfLive, m_gridW, m_gridH, m_gridD);
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
	getSlice(sliceTSDFDef, tsdfLiveGridDef, 128, m_gridW, m_gridH);
	getSlice(sliceTSDF, tsdfLiveGrid, 128, m_gridW, m_gridH);
	getSlice(sliceGradX, gradX, 128, m_gridW, m_gridH);
	getSlice(sliceLapU, lapU, 128, m_gridW, m_gridH);
	getSlice(sliceHessXX, hessXX, 128, m_gridW, m_gridH);
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


void Optimizer::computeGradient(float* gradOutX, float* gradOutY, float* gradOutZ, const float* gridIn, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
	// Compute gradients for SDF live grid
	computeConvolution3D(gradOutX, gridIn, kernelDx, kradius, w, h, d);
    computeConvolution3D(gradOutY, gridIn, kernelDy, kradius, w, h, d);
	computeConvolution3D(gradOutZ, gridIn, kernelDz, kradius, w, h, d);
}

void Optimizer::computeDivergence(float* divOut, const float* gridInU, const float* gridInV, const float* gridInW, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
    cudaMalloc(&m_d_du, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_dv, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&m_d_dw, (m_gridW * m_gridH * m_gridD) * sizeof(float)); CUDA_CHECK;

	// Compute gradients for the deformation field
	computeConvolution3D(m_d_du, gridInU, kernelDx, kradius, w, h, d);
    computeConvolution3D(m_d_dv, gridInV, kernelDy, kradius, w, h, d);
	computeConvolution3D(m_d_dw, gridInW, kernelDz, kradius, w, h, d);

	// Sum the three gradient components
	computeDivergence3DCuda(divOut, m_d_du, m_d_dv, m_d_dw, w, h, d);

    cudaFree(m_d_du); CUDA_CHECK;
    cudaFree(m_d_dv); CUDA_CHECK;
    cudaFree(m_d_dw); CUDA_CHECK;
}

void Optimizer::computeLapacian(float* lapOut, const float* deformationIn, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d)
{
    //todo since this function uses m_d_dux/y/z as temporary storage variables, but their names collide with the the derivatives of the dux/y/z
    //their naming must be changed
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
