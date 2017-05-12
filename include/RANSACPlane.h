#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>


#define PI 3.14159265
#define SWAP(a,b) {temp=a;a=b;b=temp;}

using namespace cv;

class RANSACPlane
{
public:
	struct Parameter
	{
		int mWidth;
		int mHeight;
		int mDThreshold;
		int mNumSamples;  
		Parameter() :mWidth(0), mHeight(0), mDThreshold(1), mNumSamples(300){}
		Parameter(int width,int height,int dThreshold,int numSamples) :mWidth(width), mHeight(height), mDThreshold(dThreshold), mNumSamples(numSamples){}
	};

	struct RansacROI
	{
		int startX;
		int startY;
		int endX;
		int endY;
		RansacROI(){ startX = startY = endX = endY = 0; }
		RansacROI(int x0, int x1, int y0, int y1) :startX(x0), endX(x1), startY(y0), endY(y1){}
		RansacROI(const RansacROI& roi) :startX(roi.startX), endX(roi.endX), startY(roi.startY), endY(roi.endY){}
	};

	struct Disp
	{
		float u, v, d;
		Disp(float u, float v, float d) : u(u), v(v), d(d) {}
	};

	RANSACPlane(Parameter* param);
	RANSACPlane();
	~RANSACPlane();

	bool compute(Mat* disparity, Mat* groundMap=NULL);
	void getRoadPlane(float *plane);

	static inline uint32_t getAddressOffsetImage(const int32_t& u, const int32_t& v, const int32_t& width) 
	{
		return v*width + u;
	}

	static void drawRandomPlaneSample(std::vector<Disp> &d_list, float *plane);
	std::vector<Disp> sparseDisparityGrid(float* D, const int width, const int height, int32_t step_size);
	static bool gaussJordanElimination(float **a, const int n, float **b, int m, float eps = 1e-8);
	static void leastSquarePlane(std::vector<Disp> &d_list, std::vector<int32_t> &ind, float *plane);

private:
	RansacROI mRoi;
	cv::Mat *mDisparity;
	cv::Mat mDispFloat;
	cv::Mat *mGroundMap;
	Parameter *mParam;
	float mPlane[3];
};
