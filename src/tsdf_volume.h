#ifndef TSDF_VOLUME_H
#define TSDF_VOLUME_H

#include <opencv2/core/core.hpp>
#include "mat.h"

class TSDFVolume
{
public:

    TSDFVolume(const Vec3i &dimensions, const Vec3f &size, const Mat3f &K, const float truncationDistance, size_t frameNumber);
    ~TSDFVolume();

    bool init();

    void release();

    void bbox(Vec3 &min, Vec3 &max) const;

    void setSize(const Vec3f &size);
    Vec3f size() const;

    Vec3i dimensions() const;

    Vec3f voxelToWorld(const Vec3i &voxel) const;
    Vec3f voxelToWorld(int i, int j, int k) const;
    Vec3f voxelToWorld(const Vec3f &voxel) const;

    Vec3i worldToVoxel(const Vec3f &pt) const;
    Vec3f worldToVoxelF(const Vec3f &pt) const;

    void integrate(const Mat4f &pose, const cv::Mat &color, const cv::Mat &vertexMap, const cv::Mat &normals = cv::Mat());

    bool load(const std::string &filename);
    bool save(const std::string &filename);

    size_t getFrameNumber() {return m_frameNumber;}

    float truncate(float sdf) const;

    float interpolate3(float x, float y, float z) const;

    void setDelta(float delta) { m_delta = delta; m_deltaInv = 1.0f / m_delta; }
    float delta() const { return m_delta; }

    float* ptrTsdf() { return m_tsdf; }
    float* ptrTsdfWeights() { return m_weights; }
    unsigned char* ptrColorR() { return m_colorR; }
    unsigned char* ptrColorG() { return m_colorG; }
    unsigned char* ptrColorB() { return m_colorB; }

    Vec3f surfaceNormal(int i, int j, int k);

protected:
    float interpolate3voxel(float x, float y, float z) const;

    Vec3i m_dim;
    size_t m_gridSize;
    Vec3f m_size;
    Vec3f m_voxelSize;

    size_t m_frameNumber;
    float* m_tsdf;
    float* m_weights;
    unsigned char* m_colorR;
    unsigned char* m_colorG;
    unsigned char* m_colorB;
    float* m_weightsColor;
    Mat3f m_K;

    float m_delta;
    float m_deltaInv;
    };

#endif
