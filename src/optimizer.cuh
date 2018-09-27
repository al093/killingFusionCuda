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
	void computeGradient(float* gradOutX, float* gradOutY, float* gradOutZ, const float* tsdfLive, const float *kernelDx, const float *kernelDy, const float *kernelDz, int kradius, int w, int h, int d);
	void computeDivergence(float* divOut, const float* deformationInU, const float* deformationInV, const float* deformationInW, const float *kernelDx, const float *kernelDy, const float *kernelDz, int kradius, int w, int h, int d);
	void getSlice(float* sliceOut, const float* gridIn, size_t sliceInd);

    TSDFVolume* m_tsdfGlobal;
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
	float* m_d_dx = NULL;
	float* m_d_dy = NULL;
	float* m_d_dz = NULL;
	float* m_d_div = NULL;
};

