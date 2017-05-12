#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "SGMStereo.h"
#include "RANSACPlane.h"

class GroundEstimator
{
public:
	struct Parameter
	{
		float mCU;
		float mCV;
		float mBaseline;
		float mFocal;
		int   mUDispThres;   // 用于在U视差图中初步筛选障碍物区域
		float mSigma;        // 用于对V视差图进行后处理用到的参数（高斯模糊的标准差）
		float mGroundLineBias;
		float mSlopeAngle;
		Parameter() :mCU(0), mBaseline(0), mFocal(0), mCV(0), mUDispThres(10), mSigma(0.7), mGroundLineBias(-20), mSlopeAngle(10){}
		Parameter(float cu, float cv, float baseline, float focal)
			:mCV(cv), mCU(cu), mBaseline(baseline), mFocal(focal), mUDispThres(10), mSigma(0.7), mGroundLineBias(-20), mSlopeAngle(10){}
	};

	SGMStereo::Parameter *mStereoParam;
	Parameter *mGroundEstParam;
	RANSACPlane::Parameter *mRANSACParam;
	
	Mat *mLeftImg;
	Mat *mRightImg;
	Mat *mDisparity;

	void compute(Mat *_leftImg, Mat *_rightImg, Mat* _disparity);
	Mat getGroundMap(){ return mGroundMapWithPlane; }
	std::vector<float> getUpperBound(){ return mGroundUpperBound; };
	GroundEstimator(Parameter* param, SGMStereo::Parameter* stereoParam,RANSACPlane::Parameter* RANSACParam);
	~GroundEstimator();

private:
	Mat mUDispMap;
	Mat mVDispMap;
	Mat mRefinedVDisp;
	Mat mObstacleMap;
	Mat mGroundMap;
	Mat mGroundMapWithPlane;

	bool mGroundLineExist;
	std::vector<float> mGroundLine;
	std::vector<float> mGroundUpperBound;
	RANSACPlane planeEstimator;
	float mPlane[3];
	bool  mPlaneValid;

	void computeUDisparity();	  
	void removeObstacle();        // 通过UdispMap初步将障碍物区域筛选出来
	void computeVDisparity();     // 基于Udisp初步过滤之后，计算V视差空间的投影
	void refineVdispMap();        // 将vdisp进行模糊以及边缘检测
	bool groundLineFromVDisp();   // 将细化后的Vdisp进行直线检测，得到f[d] = v
	void groundMapExtraction();   // 从f[d] =v中得到图中的初步路面区域
	void groundPlaneEstimation(); // RANSAC道路平面拟合
	void groundPlaneRefinement(); // 根据RANSAC的结果来进行路面的细化
};