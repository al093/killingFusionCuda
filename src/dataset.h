#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>
#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

bool loadIntrinsics(const std::string &intrinsicsFile, Eigen::Matrix3f &K);

bool loadFrame(const std::string &folder, size_t index, cv::Mat &color, cv::Mat &depth, cv::Mat &mask);

void filterDepth(const cv::Mat &mask, cv::Mat &depth);

#endif
