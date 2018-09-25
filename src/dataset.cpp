#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


bool loadIntrinsics(const std::string &intrinsicsFile, Eigen::Matrix3f &K)
{
    if (intrinsicsFile.empty())
        return false;

    std::ifstream intrIn(intrinsicsFile.c_str());
    if (!intrIn.is_open())
        return false;

    //camera intrinsics
    K = Eigen::Matrix3f::Identity();
    float fVal = 0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            intrIn >> fVal;
            kMat(i, j) = fVal;
        }
    }
    intrIn.close();

    return true;
}


bool loadFrame(const std::string &folder, size_t index, cv::Mat &color, cv::Mat &depth, cv::Mat &mask)
{
    // build postfix
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << index;
    std::string id = ss.str() + ".png";

    // load color
    std::string colorFile = folder + "/color_" + id;
    color = cv::imread(colorFile);
    if (color.empty())
        return false;

    // load depth
    std::string depthFile = folder + "/depth_" + id;
    //fill/read 16 bit depth image
    cv::Mat depthIn = cv::imread(depthFile, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    if (depthIn.empty())
        return false;
    depthIn.convertTo(depth, CV_32FC1, (1.0 / 1000.0));

    // load mask
    std::string maskFile = folder + "/omask_" + id;
    mask = cv::imread(maskFile);

    return true;
}


#endif
