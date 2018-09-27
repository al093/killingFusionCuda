#include <opencv2/core/core.hpp>
#include "mat.h"
#include "tsdf_volume.h"



const float MAX_VECTOR_UPDATE_THRESHOLD = 0.1;
const float m_kernelDxCentralDiff[27] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.5f, 0.0f, -0.5f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const float m_kernelDyCentralDiff[27] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.5f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const float m_kernelDzCentralDiff[27] = {0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f};


class Optimizer
{
public:

    Optimizer(TSDFVolume* tsdfGlobal, float* initialDeformationU, float* initialDeformationV, float* initialDeformationW, const float alpha, const float wk, const float ws, const size_t gridW, const size_t gridH, const size_t gridD);
    ~Optimizer();

	void optimize(float* optimDeformationU, float* optimDeformationV, float* optimDeformationW, TSDFVolume* tsdfLive);
	void test(float* optimDeformationU, float* optimDeformationV, float* optimDeformationW, TSDFVolume* tsdfLive);

protected:
	void allocateMemoryInDevice();
	void copyArraysToDevice();
	void computeGradient(float* gradOutX, float* gradOutY, float* gradOutZ, const float* gridIn, const float *kernelDx, const float *kernelDy, const float *kernelDz, int kradius, int w, int h, int d);
	void computeDivergence(float* divOut, const float* gridInU, const float* gridInV, const float* gridInW, const float *kernelDx, const float *kernelDy, const float *kernelDz, int kradius, int w, int h, int d);
	void computeLapacian(float* lapOut, const float* deformationIn, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d);
	void computeHessian(float* hessOutXX, float* hessOutXY, float* hessOutXZ, float* hessOutYY, float* hessOutYZ, float* hessOutZZ, const float* gradX, const float* gradY, const float* gradZ, const float* kernelDx, const float* kernelDy, const float* kernelDz, int kradius, int w, int h, int d);
	void getSlice(float* sliceOut, const float* gridIn, size_t sliceInd);

    TSDFVolume* m_tsdfGlobal;
    float* m_d_tsdfLive = NULL;
    float* m_deformationFieldU, * m_deformationFieldV, * m_deformationFieldW;
	float* m_d_deformationFieldU, * m_d_deformationFieldV, * m_d_deformationFieldW;
    float m_alpha;
	float m_wk;
	float m_ws;
	const size_t m_gridW, m_gridH, m_gridD;

	float* m_d_kernelDx = NULL;
	float* m_d_kernelDy = NULL;
	float* m_d_kernelDz = NULL;
	// Gradients of live SDF
	float* m_d_sdfDx = NULL;
	float* m_d_sdfDy = NULL;
	float* m_d_sdfDz = NULL;

	// Gradients of deformation field
	float* m_d_dux = NULL;
	float* m_d_duy = NULL;
	float* m_d_duz = NULL;

	// Hessian
	float* m_d_hessXX = NULL;
	float* m_d_hessXY = NULL;
	float* m_d_hessXZ = NULL;
	float* m_d_hessYY = NULL;
	float* m_d_hessYZ = NULL;
	float* m_d_hessZZ = NULL;

	/*float* m_d_dvx = NULL;
	float* m_d_dvy = NULL;
	float* m_d_dvz = NULL;
	float* m_d_dwx = NULL;
	float* m_d_dwy = NULL;
	float* m_d_dwz = NULL;*/
	float* m_d_du = NULL;
	float* m_d_dv = NULL;
	float* m_d_dw = NULL;
	float* m_d_div = NULL;
	float* m_d_lapu = NULL;
};

