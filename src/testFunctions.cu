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
#include "convolution.cuh"


#define STR1(x)  #x
#define STR(x)  STR1(x)


typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;

// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <string>

const float kernelDxCentralDiff[27] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.5f, 0.0f, -0.5f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const float kernelDyCentralDiff[27] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.5f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const float kernelDzCentralDiff[27] = {0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
									   0.0f, 0.0f, 0.0f, 0.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f};

int main(int argc,char **argv)
{
    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{b|bw|false|load input image as grayscale/black-white}"
        "{s|sigma|3.0|sigma}"
        "{r|repeats|1|number of computation repetitions}"
        "{c|cpu|false|compute on CPU}"
        "{m|mem|0|memory: 0=global, 1=shared, 2=texture, 3=shared+constant}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // number of computation repetitions to get a better run time measurement
    size_t repeats = (size_t)cmd.get<int>("repeats");
    // load the input image as grayscale
    bool gray = cmd.get<bool>("bw");
    // compute on CPU
    bool cpu = cmd.get<bool>("cpu");
    std::cout << "mode: " << (cpu ? "CPU" : "GPU") << std::endl;
    float sigma = cmd.get<float>("sigma");
    std::cout << "sigma: " << sigma << std::endl;
    size_t memory = (size_t)cmd.get<int>("mem");
    if (memory == 1)
        std::cout << "memory: shared" << std::endl;
    else if (memory == 2)
        std::cout << "memory: texture" << std::endl;
	else if (memory == 3)
        std::cout << "memory: shared+constant" << std::endl;
    else
        std::cout << "memory: global" << std::endl;

    // init camera
    bool useCam = inputImage.empty();
    cv::VideoCapture camera;
    if (useCam && !openCamera(camera, 0))
    {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    // read input frame
    cv::Mat mIn;
    if (useCam)
    {
        // read in first frame to get the dimensions
        camera >> mIn;
    }
    else
    {
        // load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
        mIn = cv::imread(inputImage.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
    }
    // check
    if (mIn.empty())
    {
        std::cerr << "ERROR: Could not retrieve frame " << inputImage << std::endl;
        return 1;
    }
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn, CV_32F);

    // init kernel
    int kradius = ceil(3 * sigma);    // TODO (5.1) calculate kernel radius using sigma
    std::cout << "kradius: " << kradius << std::endl;
    int k_diameter = 2*kradius + 1;     // TODO (5.1) calculate kernel diameter from radius
    int kn = k_diameter*k_diameter;
    float *kernel = new float[kn];    // TODO (5.1) allocate array
    // TODO (5.1) implement createConvolutionKernel() in convolution.cu
    createConvolutionKernel(kernel, kradius, sigma);

    cv::Mat mKernel(k_diameter,k_diameter,CV_32FC1);
	cv::Mat mKernelResized(k_diameter,k_diameter,CV_32FC1);
    {
    	// TODO (5.2) fill mKernel for visualization
		convertLayeredToMat(mKernelResized, kernel);
		cv::normalize(mKernelResized, mKernelResized, 1.0, 0);
		cv::resize(mKernelResized, mKernelResized, cv::Size(), 10, 10);
    	showImage("Resized Kernel", mKernelResized, 100, 100);
    }

    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    std::cout << "Image: " << w << " x " << h << std::endl;

    // initialize CUDA context
    cudaDeviceSynchronize();  CUDA_CHECK;

    // ### Set the output image format
    cv::Mat mOut(h,w,mIn.type());  // grayscale or color depending on input image, nc layers

    // ### Allocate arrays
    // allocate raw input image array
    float *imgIn = new float[h * w * nc];    // TODO allocate array
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[h * w * nc];   // TODO allocate array

    // allocate arrays on GPU
    float *d_imgIn = NULL;
    float *d_imgOut = NULL;
    float *d_kernel = NULL;
    // TODO alloc cuda memory for device arrays
	cudaMalloc(&d_imgIn, (h * w * nc) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_imgOut, (h * w * nc) * sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_kernel, (h * w * nc) * sizeof(float)); CUDA_CHECK;

    do
    {
		Timer timer;
        timer.start();
        // convert range of each channel to [0,1]
        mIn /= 255.0f;
        // init raw input image array (and convert to layered)
        convertMatToLayered (imgIn, mIn);

		float* imgColor = new float[w*h*nc];
        convertMatToLayered (imgColor, mIn);
		float* d_imgIn = NULL;
		float* d_kernelDx = NULL;
		float* d_kernelDy = NULL;
		float* d_gradX = NULL;
		float* d_gradY = NULL;
		float* gradX = new float[w*h*nc];
		float* gradY = new float[w*h*nc];
		cv::Mat mColor_grad_X(h,w,mIn.type());
		cv::Mat mColor_grad_Y(h,w,mIn.type());
		cudaMalloc(&d_imgIn, (w*h*nc) * sizeof(float)); CUDA_CHECK;
		cudaMalloc(&d_gradX, (w*h*nc) * sizeof(float)); CUDA_CHECK;
		cudaMalloc(&d_gradY, (w*h*nc) * sizeof(float)); CUDA_CHECK;
		cudaMalloc(&d_kernelDx, (27) * sizeof(float)); CUDA_CHECK;
		cudaMalloc(&d_kernelDy, (27) * sizeof(float)); CUDA_CHECK;

		cudaMemcpy(d_imgIn, imgColor, (w*h*nc) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMemcpy(d_kernelDx, kernelDxCentralDiff, (27) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMemcpy(d_kernelDy, kernelDyCentralDiff, (27) * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		computeConvolution3D(d_gradX, d_imgIn, d_kernelDx, 1, w, h, nc);
    	computeConvolution3D(d_gradY, d_imgIn, d_kernelDy, 1, w, h, nc);
		cudaMemcpy(gradX, d_gradX, (w*h*nc) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
		cudaMemcpy(gradY, d_gradY, (w*h*nc) * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
		convertLayeredToMat(mColor_grad_X, gradX);
		convertLayeredToMat(mColor_grad_Y, gradY);

		showImage("Grad X", mColor_grad_X*10, 100+w+40, 100);
		showImage("Grad Y", mColor_grad_Y*10, 100+w+40, 100);
		timer.end();
		float t = timer.get()/repeats;
		std::cout << "time: " << t*1000 << " ms" << std::endl;

        // proceed similarly for other output images, e.g. the convolution kernel:
        if (!mKernel.empty())
			convertLayeredToMat(mKernel, kernel);
            showImage("Kernel", mKernel, 100, 50);

        if (useCam)
        {
            // wait 30ms for key input
            if (cv::waitKey(30) >= 0)
            {
                mIn.release();
            }
            else
            {
                // retrieve next frame from camera
                camera >> mIn;
                // convert to float representation (opencv loads image values as single bytes by default)
                mIn.convertTo(mIn, CV_32F);
            }
        }
    }
    while (useCam && !mIn.empty());

    if (!useCam)
    {
        cv::waitKey(0);

        // save input and result
        cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
        cv::imwrite("image_result.png",mOut*255.f);
		cv::imwrite("image_kernel_resized.png",mKernelResized*255.f);
        cv::imwrite("image_kernel.png",mKernel*255.f);
    }

    // ### Free allocated arrays
    // TODO free cuda memory of all device arrays
	cudaFree(d_imgIn); CUDA_CHECK;
	cudaFree(d_imgOut); CUDA_CHECK;
	cudaFree(d_kernel); CUDA_CHECK;
    // TODO free memory of all host arrays
	delete[] imgOut;
    delete[] imgIn;
	delete[] kernel;
    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}




		
