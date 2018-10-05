#include <opencv2/core/core.hpp>
#include "mat.h"
#include "tsdf_volume.h"



const float MAX_VECTOR_UPDATE_THRESHOLD = 0.1 / 8.0; // Threshold: 0.1mm -> (/ 4mm per voxel, / alpha)
const float m_kernelDxCentralDiff[27] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, -0.5f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const float m_kernelDyCentralDiff[27] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const float m_kernelDzCentralDiff[27] = {0.0f, 0.0f, 0.0f, 0.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f};

class Optimizer
{
public:

    Optimizer(TSDFVolume* tsdfGlobal, float* initialDeformationU, float* initialDeformationV, float* initialDeformationW, const float alpha,
    		  const float wk, const float ws, const size_t maxIterations, const size_t gridW, const size_t gridH, const size_t gridD, const bool debugMode);
    ~Optimizer();

	void getTSDFGlobalPtr(float* tsdfGlobalPtr);
	void getTSDFGlobalWeightsPtr(float* tsdfGlobalWeightsPtr);
	void printTimes();

	void optimize(TSDFVolume* tsdfLive);
	void optimizeTest(TSDFVolume* tsdfLive);
	void test(TSDFVolume* tsdfLive);
	

protected:
	void allocateMemoryInDevice();
	void copyArraysToDevice();
	void computeGradient(float* gradOutX, float* gradOutY, float* gradOutZ, const float* gridIn, const float *kernelDx, const float *kernelDy, const float *kernelDz, int kradius, int w, int h, int d);
	void computeDivergence(float* divOut, const float* gridInU, const float* gridInV, const float* gridInW, const float *kernelDx, const float *kernelDy, const float *kernelDz, int kradius, int w, int h, int d);
	void computeLapacian(float* lapOut, const float* deformationIn, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d);
	void computeHessian(float* hessOutXX, float* hessOutXY, float* hessOutXZ, float* hessOutYY, float* hessOutYZ, float* hessOutZZ, const float* gradX, const float* gradY, const float* gradZ, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d);

    TSDFVolume* m_tsdfGlobal;
    float* m_d_tsdfGlobal = NULL, * m_d_tsdfLive = NULL, * m_d_tsdfGlobalWeights = NULL, * m_d_tsdfLiveWeights = NULL;
    float* m_deformationFieldU, * m_deformationFieldV, * m_deformationFieldW;
	float* m_d_deformationFieldU, * m_d_deformationFieldV, * m_d_deformationFieldW;
    float m_alpha;
	float m_wk;
	float m_ws;
	size_t m_maxIterations;
	bool m_debugMode;
	const size_t m_gridW, m_gridH, m_gridD;

	// Timing variables
	float m_timeComputeGradient = 0.0f, m_timeComputeHessian = 0.0f, m_timeComputeLaplacian = 0.0f, m_timeComputeDivergence = 0.0f,
		   m_timeComputeDataTermDerivative = 0.0f, m_timeComputeLevelSetDerivative = 0.0f, m_timeComputeMotionRegularizerDerivative = 0.0f,
		   m_timeAddArray = 0.0f, m_timeComputeMagnitude = 0.0f, m_timeFindAbsMax = 0.0f, m_timeAddWeightedArray = 0.0f;
	size_t m_nComputeGradient = 0, m_nComputeHessian = 0, m_nComputeLaplacian = 0, m_nComputeDivergence = 0,
		   m_nComputeDataTermDerivative = 0, m_nComputeLevelSetDerivative = 0, m_nComputeMotionRegularizerDerivative = 0,
		   m_nAddArray = 0, m_nComputeMagnitude = 0, m_nFindAbsMax = 0, m_nAddWeightedArray = 0;

	// Kernel variables
	float* m_d_kernelDx = NULL;
	float* m_d_kernelDy = NULL;
	float* m_d_kernelDz = NULL;
	// Gradients of live SDF
	float* m_d_sdfDx = NULL;
	float* m_d_sdfDy = NULL;
	float* m_d_sdfDz = NULL;

	// Gradients of deformation field
    //TODO these variables may not be needed during normal runtimes, only needed for debugging
    //but the laplacian function uses m_d_dux/y/z as temp variables!!
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

	float* m_d_du = NULL;
	float* m_d_dv = NULL;
	float* m_d_dw = NULL;
	float* m_d_div = NULL;
    float* m_d_divX = NULL;
    float* m_d_divY = NULL;
    float* m_d_divZ = NULL;
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
    
    float* m_d_energyDu = NULL;
    float* m_d_energyDv = NULL;
    float* m_d_energyDw = NULL;

	// Magnitude
    float* m_d_magnitude = NULL;

    bool* m_d_mask = NULL;
};

