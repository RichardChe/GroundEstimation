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
		int   mUDispThres;   // ������U�Ӳ�ͼ�г���ɸѡ�ϰ�������
		float mSigma;        // ���ڶ�V�Ӳ�ͼ���к����õ��Ĳ�������˹ģ���ı�׼�
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
	void removeObstacle();        // ͨ��UdispMap�������ϰ�������ɸѡ����
	void computeVDisparity();     // ����Udisp��������֮�󣬼���V�Ӳ�ռ��ͶӰ
	void refineVdispMap();        // ��vdisp����ģ���Լ���Ե���
	bool groundLineFromVDisp();   // ��ϸ�����Vdisp����ֱ�߼�⣬�õ�f[d] = v
	void groundMapExtraction();   // ��f[d] =v�еõ�ͼ�еĳ���·������
	void groundPlaneEstimation(); // RANSAC��·ƽ�����
	void groundPlaneRefinement(); // ����RANSAC�Ľ��������·���ϸ��
};