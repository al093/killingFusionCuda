#include <iostream>
#include <vector>

#include "mat.h"

#include <cuda_runtime.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "helper.cuh"
#include "dataset.h"
#include "tsdf_volume.h"
#include "marching_cubes.h"
#include "optimizer.cuh"


#define STR1(x)  #x
#define STR(x)  STR1(x)


typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;


bool depthToVertexMap(const Mat3f &K, const cv::Mat &depth, cv::Mat &vertexMap)
{
    if (depth.type() != CV_32FC1 || depth.empty())
        return false;

    int w = depth.cols;
    int h = depth.rows;
    vertexMap = cv::Mat::zeros(h, w, CV_32FC3);
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxInv = 1.0f / fx;
    float fyInv = 1.0f / fy;
    float* ptrVert = (float*)vertexMap.data;

    const float* ptrDepth = (const float*)depth.data;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float depthMeter = ptrDepth[y*w + x];
            float x0 = (float(x) - cx) * fxInv;
            float y0 = (float(y) - cy) * fyInv;

            size_t off = (y*w + x) * 3;
            ptrVert[off] = x0 * depthMeter;
            ptrVert[off+1] = y0 * depthMeter;
            ptrVert[off+2] = depthMeter;
        }
    }

    return true;
}


Vec3f centroid(const cv::Mat &vertexMap)
{
    Vec3f centroid(0.0, 0.0, 0.0);

    size_t cnt = 0;
    for (int y = 0; y < vertexMap.rows; ++y)
    {
        for (int x = 0; x < vertexMap.cols; ++x)
        {
            cv::Vec3f pt = vertexMap.at<cv::Vec3f>(y, x);
            if (pt.val[2] > 0.0)
            {
                Vec3f pt3(pt.val[0], pt.val[1], pt.val[2]);
                centroid += pt3;
                ++cnt;
            }
        }
    }
    centroid /= float(cnt);

    return centroid;
}


int main(int argc, char *argv[])
{
    // default input sequence in folder
    std::string dataFolder = std::string(STR(KILLINGFUSION_SOURCE_DIR)) + "/data/";

    // parse command line parameters
    const char *params = {
        "{i|input| |input rgb-d sequence}"
        "{f|frames|10000|number of frames to process (0=all)}"
        "{n|iterations|100|max number of GD iterations}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input sequence
    // download from http://campar.in.tum.de/personal/slavcheva/deformable-dataset/index.html
    std::string inputSequence = cmd.get<std::string>("input");
    if (inputSequence.empty())
    {
        inputSequence = dataFolder;
        //std::cerr << "No input sequence specified!" << std::endl;
        //return 1;
    }
    std::cout << "input sequence: " << inputSequence << std::endl;
    // number of frames to process
    size_t frames = (size_t)cmd.get<int>("frames");
    std::cout << "# frames: " << frames << std::endl;
    // max number of GD iterations
    size_t iterations = (size_t)cmd.get<int>("iterations");
    std::cout << "iterations: " << iterations << std::endl;

    // initialize cuda context
    cudaDeviceSynchronize(); CUDA_CHECK;

    // load camera intrinsics
    Eigen::Matrix3f K;
    if (!loadIntrinsics(inputSequence + "/intrinsics_kinect1.txt", K))
    {
        std::cerr << "No intrinsics file found!" << std::endl;
        return 1;
    }
    std::cout << "K: " << std::endl << K << std::endl;

    // create tsdf volume
	size_t gridW = 256, gridH = 256, gridD = 256;
	float alpha = 0.1, wk = 0.5, ws = 0.2;
    Vec3i volDim(gridW, gridH, gridD);
    Vec3f volSize(1.0f, 1.0f, 1.0f);
    TSDFVolume* tsdfGlobal = new TSDFVolume(volDim, volSize, K);
	TSDFVolume* tsdfLive = new TSDFVolume(volDim, volSize, K);
	float* deformationU = new float[gridW*gridH*gridD];
	float* deformationV = new float[gridW*gridH*gridD];
	float* deformationW = new float[gridW*gridH*gridD];
	std::memset(deformationU, 0, (gridW*gridH*gridD)*sizeof(float));
	std::memset(deformationV, 0, (gridW*gridH*gridD)*sizeof(float));
	std::memset(deformationW, 0, (gridW*gridH*gridD)*sizeof(float));
	Optimizer* optimizer = new Optimizer(tsdfGlobal, deformationU, deformationV, deformationW, alpha, wk, ws, gridW, gridH, gridD);

    // create windows
    cv::namedWindow("color");
    cv::namedWindow("depth");
    cv::namedWindow("mask");

    // process frames
    Mat4f poseVolume = Mat4f::Identity();
    cv::Mat color, depth, mask;
    for (size_t i = 0; i < frames; ++i)
    {
        // load input frame
        if (!loadFrame(inputSequence, i, color, depth, mask))
        {
            //std::cerr << "Frame " << i << " could not be loaded!" << std::endl;
            //return 1;
            break;
        }

        // filter depth values outside of mask
        filterDepth(mask, depth);

        // show input images
        cv::imshow("color", color);
        cv::imshow("depth", depth);
        cv::imshow("mask", mask);
        cv::waitKey();

        // get initial volume pose from centroid of first depth map
        if (i == 0)
        {
            // initial pose for volume by computing centroid of first depth/vertex map
            cv::Mat vertMap;
            depthToVertexMap(K, depth, vertMap);
            Vec3f transCentroid = centroid(vertMap);
            poseVolume.topRightCorner<3,1>() = transCentroid;
            std::cout << "pose centroid" << std::endl << poseVolume << std::endl;
			tsdfGlobal->integrate(poseVolume, color, depth);
        }
		else
		{
            // initial pose for volume by computing centroid of first depth/vertex map
            cv::Mat vertMap;
            depthToVertexMap(K, depth, vertMap);
            Vec3f transCentroid = centroid(vertMap);
            poseVolume.topRightCorner<3,1>() = transCentroid;
            std::cout << "pose centroid" << std::endl << poseVolume << std::endl;
			// integrate frame into tsdf volume
        	tsdfLive->integrate(poseVolume, color, depth);

			// TODO: perform optimization
			//optimizer->optimize(deformationU, deformationV, deformationW, tsdfLive);
			// TODO: update global model
			
		}
        // integrate frame into tsdf volume
        
    }

    // extract mesh using marching cubes
    std::cout << "Extracting mesh..." << std::endl;
    MarchingCubes mc(volDim, volSize);
    mc.computeIsoSurface(tsdfGlobal->ptrTsdf(), tsdfGlobal->ptrTsdfWeights(), tsdfGlobal->ptrColorR(), tsdfGlobal->ptrColorG(), tsdfGlobal->ptrColorB());

    // save mesh
    std::cout << "Saving mesh..." << std::endl;
    const std::string meshFilename = inputSequence + "/mesh.ply";
    if (!mc.savePly(meshFilename))
    {
        std::cerr << "Could not save mesh!" << std::endl;
    }

    // clean up
    delete tsdfGlobal;
	delete tsdfLive;
	delete optimizer;
    cv::destroyAllWindows();

    return 0;
}
