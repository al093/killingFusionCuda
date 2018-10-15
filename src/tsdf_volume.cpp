#include "tsdf_volume.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Geometry>
#include <opencv2/highgui/highgui.hpp>



TSDFVolume::TSDFVolume(const Vec3i &dimensions, const Vec3f &size, const Mat3f &K, const float truncationDistance, size_t frameNumber) :
    m_dim(dimensions),
    m_gridSize(m_dim[0] * m_dim[1] * m_dim[2]),
    m_size(size),
    m_voxelSize(Vec3f(m_size.cwiseQuotient(m_dim.cast<float>()))),
    m_tsdf(0),
    m_weights(0),
    m_colorR(0),
    m_colorG(0),
    m_colorB(0),
    m_weightsColor(0),
    m_delta(truncationDistance),
    m_deltaInv(1.0f / truncationDistance),
    m_K(K),
    m_frameNumber(frameNumber)
{
    init();
}


TSDFVolume::~TSDFVolume()
{
    release();
}


bool TSDFVolume::init()
{
    try
    {
        // allocate and initialize memory
        m_tsdf = new float[m_gridSize];
        std::fill_n(m_tsdf, m_gridSize, -1.0f);

        m_weights = new float[m_gridSize];
        std::fill_n(m_weights, m_gridSize, 0.0f);

        unsigned char defaultColor = 127; //0;
        m_colorR = new unsigned char[m_gridSize];
        std::fill_n(m_colorR, m_gridSize, defaultColor);
        m_colorG = new unsigned char[m_gridSize];
        std::fill_n(m_colorG, m_gridSize, defaultColor);
        m_colorB = new unsigned char[m_gridSize];
        std::fill_n(m_colorB, m_gridSize, defaultColor);

        m_weightsColor = new float[m_gridSize];
        std::fill_n(m_weightsColor, m_gridSize, 0.0f);
    }
    catch (...)
    {
        release();
        return false;
    }
    return true;
}


void TSDFVolume::release()
{
    delete[] m_tsdf;
    m_tsdf = 0;
    delete[] m_weights;
    m_weights = 0;
    delete[] m_colorR;
    m_colorR = 0;
    delete[] m_colorG;
    m_colorG = 0;
    delete[] m_colorB;
    m_colorB = 0;
    delete[] m_weightsColor;
    m_weightsColor = 0;
}


float TSDFVolume::truncate(float sdf) const
{
    float tsdf = sdf;
    if (tsdf > m_delta)
        tsdf = m_delta;
    else if (tsdf < -m_delta)
        tsdf = -m_delta;
    return tsdf * m_deltaInv;   // normalize tsdf value to interval [-1.0,...,1.0]
}


void TSDFVolume::integrate(const Mat4f &pose, const cv::Mat &color, const cv::Mat &depth, const cv::Mat &normals)
{
    Mat3f R = pose.topLeftCorner(3,3);
    Vec3f t = pose.topRightCorner(3,1);

    const float* ptrDepth = (float*)depth.data;
    unsigned char* ptrColor = 0;
    if (!color.empty())
        ptrColor = (unsigned char*)color.data;
    int h = depth.rows;
    int w = depth.cols;
    float fx = m_K(0, 0);
    float fy = m_K(1, 1);
    float cx = m_K(0, 2);
    float cy = m_K(1, 2);

    // project each voxel into output image
    for (size_t z = 0; z < m_dim[2]; ++z)
    {
        for (size_t y = 0; y < m_dim[1]; ++y)
        {
            for (size_t x = 0; x < m_dim[0]; ++x)
            {
                size_t off = z*m_dim[0]*m_dim[1] + y*m_dim[0] + x;

                // transform voxel into view
                Vec3i vx(x, y, z);
                Vec3f pt = voxelToWorld(vx);
                Vec3f ptTf = R * pt + t;

                // check if voxel is behind the camera
                float distVoxel = ptTf[2];
                if (distVoxel <= 0.0f)
                    continue;

                // project point into camera and round to nearest integer to avoid smearing out of object boundaries to the background
                float zInv = 1.0f / ptTf[2];
                float uf = (fx * ptTf[0] * zInv) + cx;
                float vf = (fy * ptTf[1] * zInv) + cy;
                int u = static_cast<int>(uf + 0.5f);
                int v = static_cast<int>(vf + 0.5f);

                if (u >= 0 && u < w && v >= 0 && v < h)
                {
                    size_t idx = v*w + u;
                    size_t idx3 = idx * 3;
                    float depthIn = ptrDepth[idx];
                    if (depthIn == 0.0f || std::isnan(depthIn))
                    {
                        continue;
                    }

                    // compute tsdf value
                    float sdfVal = ptTf[2] - depthIn;

                    // truncate tsdf value
                    float tsdfVal = truncate(sdfVal);

                    // compute weight
                    float wTsdf = 0.0f;
                    if (sdfVal <= 0)//-m_delta)
                    {
                        wTsdf = 1.0f;
                    }
                    else if (sdfVal <= m_delta)
                    {
                        // constant weighting function
                        // wTsdf = 1.0f;
                        // linear weighting function
                        //  wTsdf = (tsdfVal + 1.0f) / 2.0f;
                        wTsdf = 1.0f - tsdfVal;
                    }

                    float wTsdfNew = 0.0f;
                    if (wTsdf > 0.0f)
                    {
                        // average weight
                        float wTsdfOld = m_weights[off];
                        float tsdfOld = m_tsdf[off];
                        // average tsdf value
                        wTsdfNew = wTsdfOld + wTsdf;
                        float tsdfNew = (tsdfOld * wTsdfOld + tsdfVal * wTsdf) / wTsdfNew;
                        m_tsdf[off] = tsdfNew;

                        m_weights[off] = wTsdfNew;
                    }

#if 1
                    // separate weighting for colors
                    float wColorOld = m_weightsColor[off];
                    if (sdfVal <= m_delta || wColorOld == 0.0f)
                    {
                        float wColor = wTsdf;
                        float wColorNew;
                        Vec3b c(ptrColor[idx3], ptrColor[idx3+1], ptrColor[idx3+2]);
                        unsigned char cR = c[2];
                        unsigned char cG = c[1];
                        unsigned char cB = c[0];
                        if (wColorOld == 0.0f)
                        {
                            wColorNew = wColor;
                        }
                        else
                        {
                            // compute color averaging weights
                            wColorNew = wColorOld + wColor;
                            float w0 = wColorOld / wColorNew;
                            float w1 = 1.0f - w0;
                            // compute average color
                            cR = (unsigned char)(m_colorR[off] * w0 + cR * w1);
                            cG = (unsigned char)(m_colorG[off] * w0 + cG * w1);
                            cB = (unsigned char)(m_colorB[off] * w0 + cB * w1);
                        }
                        m_weightsColor[off] = wColorNew;

                        // voxel color
                        m_colorR[off] = cR;
                        m_colorG[off] = cG;
                        m_colorB[off] = cB;
                    }
#endif
                }
            }
        }
    }
}


float TSDFVolume::interpolate3(float x, float y, float z) const
{
    Vec3f pt(x, y, z);
    Vec3f voxel = (pt + m_size.cast<float>()*0.5f).cwiseQuotient(m_voxelSize);
    return interpolate3voxel(voxel[0], voxel[1], voxel[2]);
}


float TSDFVolume::interpolate3voxel(float x, float y, float z) const
{
    Vec3f voxel(x, y, z);
    if (voxel[0] < 0.0f || voxel[0] > m_dim[0] - 1 ||
            voxel[1] < 0.0f || voxel[1] > m_dim[1] - 1 ||
            voxel[2] < 0.0f || voxel[2] > m_dim[2] - 1)
        return -1.0f;

    // tri-linear interpolation
    const int x0 = static_cast<int>(voxel[0]);
    const int y0 = static_cast<int>(voxel[1]);
    const int z0 = static_cast<int>(voxel[2]);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;

    if (x1 >= m_dim[0] || y1 >= m_dim[1] || z1 >= m_dim[2])
        return m_tsdf[z0 * m_dim[0] * m_dim[1] + y0 * m_dim[0] + x0];

    const float xd = voxel[0] - x0;
    const float yd = voxel[1] - y0;
    const float zd = voxel[2] - z0;

    const float sdf000 = m_tsdf[z0 * m_dim[0] * m_dim[1] + y0 * m_dim[0] + x0];
    const float sdf010 = m_tsdf[z0 * m_dim[0] * m_dim[1] + y1 * m_dim[0] + x0];
    const float sdf001 = m_tsdf[z1 * m_dim[0] * m_dim[1] + y0 * m_dim[0] + x0];
    const float sdf011 = m_tsdf[z1 * m_dim[0] * m_dim[1] + y1 * m_dim[0] + x0];
    const float sdf100 = m_tsdf[z0 * m_dim[0] * m_dim[1] + y0 * m_dim[0] + x1];
    const float sdf110 = m_tsdf[z0 * m_dim[0] * m_dim[1] + y1 * m_dim[0] + x1];
    const float sdf101 = m_tsdf[z1 * m_dim[0] * m_dim[1] + y0 * m_dim[0] + x1];
    const float sdf111 = m_tsdf[z1 * m_dim[0] * m_dim[1] + y1 * m_dim[0] + x1];

    const float c00 = sdf000 * (1.0f - xd) + sdf100 * xd;
    const float c10 = sdf010 * (1.0f - xd) + sdf110 * xd;
    const float c01 = sdf001 * (1.0f - xd) + sdf101 * xd;
    const float c11 = sdf011 * (1.0f - xd) + sdf111 * xd;

    const float c0 = c00 * (1.0f - yd) + c10 * yd;
    const float c1 = c01 * (1.0f - yd) + c11 * yd;

    return c0 * (1.0f - zd) + c1 * zd;
}


bool TSDFVolume::load(const std::string &filename)
{
    std::ifstream inFile(filename.c_str(), std::ios::binary);
    if (!inFile.is_open())
        return false;

    release();

    inFile.read((char*)m_dim.data(), sizeof(int) * 3);
    m_gridSize = m_dim[0] * m_dim[1] * m_dim[2];
    inFile.read((char*)m_size.data(), sizeof(double) * 3);
    inFile.read((char*)&m_delta, sizeof(float));
    m_deltaInv = 1.0f / m_delta;

    bool ok = init();

    inFile.read((char*)m_tsdf, sizeof(float) * m_gridSize);
    inFile.read((char*)m_weights, sizeof(float) * m_gridSize);

    inFile.read((char*)m_colorR, sizeof(unsigned char) * m_gridSize);
    inFile.read((char*)m_colorG, sizeof(unsigned char) * m_gridSize);
    inFile.read((char*)m_colorB, sizeof(unsigned char) * m_gridSize);
    inFile.read((char*)m_weightsColor, sizeof(float) * m_gridSize);

    inFile.close();

    return ok;
}


bool TSDFVolume::save(const std::string &filename)
{
    std::ofstream outFile(filename.c_str(), std::ios::binary);
    if (!outFile.is_open())
        return false;

    outFile.write((const char*)m_dim.data(), sizeof(int) * 3);
    outFile.write((const char*)m_size.data(), sizeof(double) * 3);
    outFile.write((const char*)&m_delta, sizeof(float));
    outFile.write((const char*)m_tsdf, sizeof(float) * m_gridSize);
    outFile.write((const char*)m_weights, sizeof(float) * m_gridSize);

    outFile.write((const char*)m_colorR, sizeof(unsigned char) * m_gridSize);
    outFile.write((const char*)m_colorG, sizeof(unsigned char) * m_gridSize);
    outFile.write((const char*)m_colorB, sizeof(unsigned char) * m_gridSize);
    outFile.write((const char*)m_weightsColor, sizeof(float) * m_gridSize);

    outFile.close();
}


Vec3f TSDFVolume::surfaceNormal(int i, int j, int k)
{
    Vec3f n = Vec3f::Zero();
    if (i < 0 || j < 0 || k < 0 ||
            i >= m_dim[0]-1 || j >= m_dim[1]-1 || k >= m_dim[2]-1)
        return n;

    size_t idx0 = k * m_dim[0] * m_dim[1] + j * m_dim[0] + i;
    size_t idx1 = k * m_dim[0] * m_dim[1] + j * m_dim[0] + (i+1);
    size_t idx2 = k * m_dim[0] * m_dim[1] + (j+1) * m_dim[0] + i;
    size_t idx3 = (k+1) * m_dim[0] * m_dim[1] + j * m_dim[0] + i;

    float sdf0 = m_tsdf[idx0];
    n[0] = m_tsdf[idx1] - sdf0;
    n[1] = m_tsdf[idx2] - sdf0;
    n[2] = m_tsdf[idx3] - sdf0;
    if (n.norm() != 0.0f)
        n.normalize();
    return n;
}



void TSDFVolume::bbox(Vec3 &min, Vec3 &max) const
{
    min = Vec3(-m_size[0]*0.5, -m_size[1]*0.5, -m_size[2]*0.5);
    max = Vec3(m_size[0]*0.5, m_size[1]*0.5, m_size[2]*0.5);
}


void TSDFVolume::setSize(const Vec3f &size)
{
    m_size = size;
    m_voxelSize = m_size.cwiseQuotient(m_dim.cast<float>());
}


Vec3f TSDFVolume::size() const
{
    return m_size;
}


Vec3i TSDFVolume::dimensions() const
{
    return m_dim;
}


Vec3f TSDFVolume::voxelToWorld(const Vec3i &voxel) const
{
    Vec3f pt = voxel.cast<float>().cwiseProduct(m_voxelSize) - m_size*0.5f;
    return pt;
}


Vec3f TSDFVolume::voxelToWorld(int i, int j, int k) const
{
    return voxelToWorld(Vec3i(i, j, k));
}


Vec3f TSDFVolume::voxelToWorld(const Vec3f &voxel) const
{
    Vec3f pt = voxel.cwiseProduct(m_voxelSize) - m_size*0.5f;
    return pt;
}


Vec3i TSDFVolume::worldToVoxel(const Vec3f &pt) const
{
    Vec3f voxelSizeInv(1.0 / m_voxelSize[0], 1.0 / m_voxelSize[1], 1.0 / m_voxelSize[2]);
    Vec3f voxelF = (pt + 0.5f * m_size).cwiseProduct(voxelSizeInv);
    Vec3i voxelIdx = voxelF.cast<int>();
    return voxelIdx;
}


Vec3f TSDFVolume::worldToVoxelF(const Vec3f &pt) const
{
    Vec3f voxelSizeInv(1.0 / m_voxelSize[0], 1.0 / m_voxelSize[1], 1.0 / m_voxelSize[2]);
    Vec3f voxelF = (pt + 0.5f * m_size).cwiseProduct(voxelSizeInv);
    return voxelF;
}
