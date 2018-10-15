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
#include "mat.h"
#include "tsdf_volume.h"

#include <opencv2/core/core.hpp>
#include "cublas_v2.h"

class Optimizer
{
public:

    Optimizer(TSDFVolume* tsdfGlobal, float* initialDeformationU, float* initialDeformationV, float* initialDeformationW, const float alpha,
              const float wk, const float ws, const float gamma, const size_t maxIterations, const float voxelSize, const float tsdfGradScale,
              const bool debugMode,const size_t gridW, const size_t gridH, const size_t gridD);
    ~Optimizer();

	void getTSDFGlobalPtr(float* tsdfGlobalPtr);
	void getTSDFGlobalWeightsPtr(float* tsdfGlobalWeightsPtr);
	void printTimes();

	void optimize(TSDFVolume* tsdfLive);
	void testIntermediateSteps(TSDFVolume* tsdfLive);
	

protected:
	void allocateMemoryInDevice();
	void copyArraysToDevice();
	void computeGradient(float* gradOutX, float* gradOutY, float* gradOutZ, const float* gridIn, int w, int h, int d);
	void computeDivergence(float* divOut, const float* gridInDu, const float* gridInDV, const float* gridInDW, int w, int h, int d);
	void sumGradients(float* divOut, const float* duxIn, const float* dvyIn, const float* dwzIn, int w, int h, int d);
	void computeLaplacian(float* lapOut, float* dOut, const size_t dOutComponent, const float* deformationIn, int w, int h, int d);
	void computeHessian(float* hessOutXX, float* hessOutXY, float* hessOutXZ, float* hessOutYY, float* hessOutYZ, float* hessOutZZ, const float* gradX, const float* gradY, const float* gradZ, int w, int h, int d);

    TSDFVolume* m_tsdfGlobal;
    
    const float m_alpha;
	const float m_wk;
	const float m_ws;
	const float m_gamma;
	const size_t m_maxIterations;
	const bool m_debugMode;
	const size_t m_gridW, m_gridH, m_gridD;
    const float m_voxelSize;
    const float m_tsdfGradScale;
	float m_maxVectorUpdateThreshold; 			// Threshold: 0.1mm
	cublasHandle_t m_handle;

	// Timing variables
	float m_timeComputeGradient = 0.0f, m_timeComputeHessian = 0.0f, m_timeComputeLaplacian = 0.0f, m_timeComputeDivergence = 0.0f,
		   m_timeComputeDataTermDerivative = 0.0f, m_timeComputeLevelSetDerivative = 0.0f, m_timeComputeMotionRegularizerDerivative = 0.0f,
           m_timeAddArray = 0.0f, m_timeComputeMagnitude = 0.0f, m_timeFindAbsMax = 0.0f, m_timeAddWeightedArray = 0.0f, m_timeFramesOptimized = 0.0f, m_interpolation = 0.0f;
	size_t m_nComputeGradient = 0, m_nComputeHessian = 0, m_nComputeLaplacian = 0, m_nComputeDivergence = 0,
		   m_nComputeDataTermDerivative = 0, m_nComputeLevelSetDerivative = 0, m_nComputeMotionRegularizerDerivative = 0,
		   m_nAddArray = 0, m_nComputeMagnitude = 0, m_nFindAbsMax = 0, m_nAddWeightedArray = 0, m_nFramesOptimized = 0, m_nInterpolation = 0;

    // TSDFs and weights
	float* m_d_tsdfGlobal = NULL;
	float* m_d_tsdfLive = NULL;
	float* m_d_tsdfGlobalWeights = NULL;
	float* m_d_tsdfLiveWeights = NULL;

	// Deformation field
	float* m_deformationFieldU = NULL;
	float* m_deformationFieldV = NULL;
	float* m_deformationFieldW = NULL;
	float* m_d_deformationFieldU = NULL;
	float* m_d_deformationFieldV = NULL;
	float* m_d_deformationFieldW = NULL;

	// Gradients of live SDF
	float* m_d_sdfDx = NULL;
	float* m_d_sdfDy = NULL;
	float* m_d_sdfDz = NULL;

	// Deformation field gradients
	float* m_d_dux = NULL;
	float* m_d_duy = NULL;
	float* m_d_duz = NULL;
    float* m_d_dvx = NULL;
    float* m_d_dvy = NULL;
    float* m_d_dvz = NULL;
    float* m_d_dwx = NULL;
    float* m_d_dwy = NULL;
    float* m_d_dwz = NULL;

	// Hessian
	float* m_d_hessXX = NULL;
	float* m_d_hessXY = NULL;
	float* m_d_hessXZ = NULL;
	float* m_d_hessYY = NULL;
	float* m_d_hessYZ = NULL;
	float* m_d_hessZZ = NULL;

	// Divergence
	float* m_d_div = NULL;
    float* m_d_divX = NULL;
    float* m_d_divY = NULL;
    float* m_d_divZ = NULL;

	// Laplacian
	float* m_d_lapU = NULL;
    float* m_d_lapV = NULL;
    float* m_d_lapW = NULL;

	// Interpolated grids
	float* m_d_tsdfLiveDeform = NULL;
	float* m_d_sdfDxDeform = NULL;
	float* m_d_sdfDyDeform = NULL;
	float* m_d_sdfDzDeform = NULL;
	float* m_d_hessXXDeform = NULL;
	float* m_d_hessXYDeform = NULL;
	float* m_d_hessXZDeform = NULL;
	float* m_d_hessYYDeform = NULL;
	float* m_d_hessYZDeform = NULL;
	float* m_d_hessZZDeform = NULL;
	float* m_d_tsdfLiveWeightsDeform = NULL;

	// Magnitude grid
	float* m_d_magnitude = NULL;

    // Mask - only near surface regions deformation is to be calculated.
    // Not used 
    bool* m_d_mask = NULL;

    // Gradients of energy
    float* m_d_energyDu = NULL;
    float* m_d_energyDv = NULL;
    float* m_d_energyDw = NULL;

    // Gradients used by to compute the divergence function
	float* m_d_du = NULL;
	float* m_d_dv = NULL;
	float* m_d_dw = NULL;
    float* m_d_dfx = NULL;
    float* m_d_dfy = NULL;
    float* m_d_dfz = NULL;
};
