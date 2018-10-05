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
        "{a|alpha|0.1|Gradient Descent step size}"
        "{b|begin|0|First frame id to begin with}"
        "{d|debug|false|Debug mode}"
        "{wk|wk|0.5|Killing term weight}"
        "{ws|ws|0.1|Level set weight}"
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

    // GD step size
    float alpha = (float)cmd.get<float>("alpha");
    std::cout << "Gradient Descent step: " << alpha << std::endl;

    // First frame id
    size_t firstFrameId = (size_t)cmd.get<int>("begin");
    std::cout << "First frame of sequence: " << firstFrameId << std::endl;

    // Debug mode
    bool debugMode = (bool)cmd.get<bool>("debug");
    std::cout << "Debug mode: " << debugMode << std::endl;

    // Killing term weight
    float wk = (float)cmd.get<float>("wk");
    std::cout << "w_k: " << wk << std::endl;

    // Level Set term weight
    float ws = (float)cmd.get<float>("ws");
    std::cout << "w_s: " << ws << std::endl;

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
    size_t gridW = 80, gridH = 80, gridD = 80;
	float voxelSize = 0.008; 		// Voxel size in m
    Vec3i volDim(gridW, gridH, gridD);
    Vec3f volSize(gridW*voxelSize, gridH*voxelSize, gridD*voxelSize);
    TSDFVolume* tsdfGlobal = new TSDFVolume(volDim, volSize, K, 0);
    TSDFVolume* tsdfLive;
    //initialize the deformation to zero initially
    float* deformationU = (float*)calloc(gridW*gridH*gridD, sizeof(float));
	float* deformationV = (float*)calloc(gridW*gridH*gridD, sizeof(float));
	float* deformationW = (float*)calloc(gridW*gridH*gridD, sizeof(float));

    for (size_t i = 0; i < gridW*gridH*gridD; i++)
    {
        deformationU[i] = 0.0f;
        deformationV[i] = 0.0f; 
        deformationW[i] = 0.0f;   
    }
    
	Optimizer* optimizer;

    // create windows
    cv::namedWindow("color");
    cv::namedWindow("depth");
    cv::namedWindow("mask");

    // process frames
    Mat4f poseVolume = Mat4f::Identity();
    cv::Mat color, depth, mask;
    for (size_t i = firstFrameId; i < frames; ++i)
    {
        tsdfLive = new TSDFVolume(volDim, volSize, K, i);
        std::cout << std::endl << " Loading Frame: " << i << std::endl;

        // load input frame
        if (!loadFrame(inputSequence, i, color, depth, mask))
        {
            std::cerr << " ->Frame " << i << " could not be loaded!" << std::endl;
            break;
        }

        // filter depth values outside of mask
        filterDepth(mask, depth);

        // show input images
        cv::imshow("color", color);
        cv::imshow("depth", depth);
        cv::imshow("mask", mask);
        if (debugMode)
        {
            cv::waitKey(30);
        }
        

        // get initial volume pose from centroid of first depth map
        if (i == firstFrameId)
        {
            // initial pose for volume by computing centroid of first depth/vertex map
            cv::Mat vertMap;
            depthToVertexMap(K, depth, vertMap);
            Vec3f transCentroid = centroid(vertMap);
            poseVolume.topRightCorner<3,1>() = transCentroid;
            std::cout << "pose centroid" << std::endl << poseVolume << std::endl;
			tsdfGlobal->integrate(poseVolume, color, depth);
			optimizer = new Optimizer(tsdfGlobal, deformationU, deformationV, deformationW, alpha, wk, ws, iterations, gridW, gridH, gridD, debugMode);

			MarchingCubes mc(volDim, volSize);
    		mc.computeIsoSurface(tsdfGlobal->ptrTsdf(), tsdfGlobal->ptrTsdfWeights(), tsdfGlobal->ptrColorR(), tsdfGlobal->ptrColorG(), tsdfGlobal->ptrColorB());
			const std::string meshFilename = inputSequence + "/mesh.ply";
			if (!mc.savePly(meshFilename))
			{
				std::cerr << "Could not save mesh!" << std::endl;
			}
        }
		else
		{
            // initial pose for volume by computing centroid of first depth/vertex map
            /*cv::Mat vertMap;
            depthToVertexMap(K, depth, vertMap);
            Vec3f transCentroid = centroid(vertMap);
            poseVolume.topRightCorner<3,1>() = transCentroid;
            std::cout << "pose centroid" << std::endl << poseVolume << std::endl;*/
			// integrate frame into tsdf volume
        	tsdfLive->integrate(poseVolume, color, depth);

			// TODO: perform optimization
			optimizer->optimize(tsdfLive);
		}
        delete tsdfLive;
    }
    optimizer->printTimes();
    // extract mesh using marching cubes
    std::cout << "Extracting mesh..." << std::endl;
	float* tsdfGlobalAccumulated = (float*)calloc(gridW*gridH*gridD, sizeof(float));
	float* tsdfGlobalWeightsAccumulated = (float*)calloc(gridW*gridH*gridD, sizeof(float));
	optimizer->getTSDFGlobalPtr(tsdfGlobalAccumulated);
	optimizer->getTSDFGlobalWeightsPtr(tsdfGlobalWeightsAccumulated);
	MarchingCubes mcAcc(volDim, volSize);
	mcAcc.computeIsoSurface(tsdfGlobalAccumulated, tsdfGlobalWeightsAccumulated, tsdfGlobal->ptrColorR(), tsdfGlobal->ptrColorG(), tsdfGlobal->ptrColorB());
    // save mesh
    std::cout << "Saving mesh..." << std::endl;
	const std::string meshAccFilename = inputSequence + "/meshAcc.ply";
	if (!mcAcc.savePly(meshAccFilename))
    {
        std::cerr << "Could not save accumulated mesh!" << std::endl;
    }

    // clean up
    delete tsdfGlobal;
	//delete tsdfLive;
	delete optimizer;
    cv::destroyAllWindows();

    return 0;
}
