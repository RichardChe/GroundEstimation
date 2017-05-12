#include "GroundEstimator.h"
#include <cstdint>
#include <iostream>
#include <vector>

GroundEstimator::GroundEstimator(Parameter* _groundParam, SGMStereo::Parameter* _stereoParam, RANSACPlane::Parameter* RANSACParam) :
mGroundEstParam(_groundParam), mStereoParam(_stereoParam), mRANSACParam(RANSACParam), planeEstimator(RANSACParam)
{

}

GroundEstimator::~GroundEstimator()
{

}

void GroundEstimator::compute(Mat *_leftImg, Mat *_rightImg, Mat* _disparity)
{
	mLeftImg = _leftImg;
	mRightImg = _rightImg;
	mDisparity = _disparity;

	computeUDisparity();
	removeObstacle();
	computeVDisparity();
	refineVdispMap();
	mGroundLineExist = groundLineFromVDisp();
	groundMapExtraction();
	//imshow("gnd", mGroundMap);
	groundPlaneEstimation();
	//imshow("gnd with plane", mGroundMapWithPlane);
	groundPlaneRefinement();
	imshow("refinement", mGroundMapWithPlane);
}

void GroundEstimator::removeObstacle()
{
	if (mObstacleMap.empty())
		mObstacleMap.create(mDisparity->rows, mDisparity->cols, CV_8UC1);

	mObstacleMap.setTo(0);

	int height = mDisparity->rows;
	int width = mDisparity->cols;

	for (int v = 0; v < height; v++)
	{
		uint8_t* pRowInDisp = mDisparity->ptr<uchar>(v);
		uint8_t* pRowInObsMap = mObstacleMap.ptr<uchar>(v);
		for (int u = 0; u < width; u++)
		{
			uint8_t currDisp = pRowInDisp[u];
			if (currDisp < mStereoParam->mMaxDisp && mUDispMap.at<ushort>(currDisp, u) > mGroundEstParam->mUDispThres)
				pRowInObsMap[u] = 255;
		}
	}
}

void GroundEstimator::computeUDisparity()
{
	if (mUDispMap.empty())
		mUDispMap.create(mStereoParam->mMaxDisp, mDisparity->cols, CV_16UC1);

	mUDispMap.setTo(0);

	int width = mDisparity->cols;
	int height = mDisparity->rows;

	for (int v = 0; v < height; v++)
	{
		uint8_t* pRowInDisp = mDisparity->ptr<uchar>(v);
		for (int u = 0; u < width; u++)
		{
			uint8_t currDisp = pRowInDisp[u];
			if (currDisp > 0 && currDisp < mStereoParam->mMaxDisp)
				mUDispMap.at<ushort>(currDisp, u)++;
		}
	}
	//Mat temp;
	//mUDispMap.convertTo(temp, CV_8UC1);
	//imshow("u", temp);
}

void GroundEstimator::computeVDisparity()
{
	if (mVDispMap.empty())
		mVDispMap.create(mDisparity->rows, mStereoParam->mMaxDisp, CV_16UC1);

	mVDispMap.setTo(0);

	int height = mDisparity->rows;
	int width = mDisparity->cols;

	for (int v = 0; v < height; v++)
	{
		uint8_t* pRowInDisp = mDisparity->ptr<uchar>(v);
		uint8_t* pRowInObsMap = mObstacleMap.ptr<uchar>(v);
		uint16_t* pRowInVDisp = mVDispMap.ptr<ushort>(v);

		for (int u = 0; u < width; u++)
		{
			uint8_t currDisp = pRowInDisp[u];
			if (currDisp > 0 && currDisp < mStereoParam->mMaxDisp && pRowInObsMap[u] == 0)
				pRowInVDisp[currDisp]++;
		}
	}
	/*
		Mat temp;
		mVDispMap.convertTo(temp, CV_8UC1);
		imshow("v", temp);*/
}

void GroundEstimator::refineVdispMap()
{
	int maxDisp = mStereoParam->mMaxDisp;
	//std::vector<float> vGroundLine(maxDisp);
	mVDispMap.convertTo(mRefinedVDisp, CV_8UC1);

	GaussianBlur(mRefinedVDisp, mRefinedVDisp, Size(5, 5), mGroundEstParam->mSigma);
	Canny(mRefinedVDisp, mRefinedVDisp, 40, 150);
}

bool GroundEstimator::groundLineFromVDisp()
{
	int maxDisp = mStereoParam->mMaxDisp;
	mGroundLine = std::vector<float>(maxDisp, 0);
	std::vector<Vec2f> lines;
	HoughLines(mRefinedVDisp, lines, 1, CV_PI / 180, 50, 0, 0);

	if (!lines.size())
	{
		std::cout << "No ground line detected in groundLineFromVDisp" << std::endl;
		return false;
	}

	float theta = lines[0][1];
	float rho = lines[0][0];

	float cosTheta = cos(theta);
	float sinTheta = sin(theta);

	float x0 = rho * cosTheta;
	float y0 = rho * sinTheta;

	float x1 = x0 + 1000 * (-sinTheta);
	float y1 = y0 + 1000 * (cosTheta);

	float k = (y1 - y0) / (x1 - x0);
	for (int currDisp = 0; currDisp < maxDisp; currDisp++)
	{
		float currV = k* (currDisp - x0) + y0;
		mGroundLine[currDisp] = currV + mGroundEstParam->mGroundLineBias;
	}

	return true;
}

void GroundEstimator::groundMapExtraction()
{
	int height = mDisparity->rows;
	int width = mDisparity->cols;
	if (mGroundMap.empty())
		mGroundMap.create(height, width, CV_8UC1);

	for (int v = 0; v < height; v++)
	{
		uint8_t* pRowInDisp = mDisparity->ptr<uchar>(v);
		uint8_t* pRowInGroundMap = mGroundMap.ptr<uchar>(v);
		for (int u = 0; u < width; u++)
		{
			int currDisp = pRowInDisp[u];
			if (currDisp > 0)
				if (v >= mGroundLine[currDisp])
					pRowInGroundMap[u] = 0;
				else
					pRowInGroundMap[u] = 255;
			else
				pRowInGroundMap[u] = 255;
		}
	}
}

void GroundEstimator::groundPlaneEstimation()
{
	int width = mDisparity->cols;
	int height = mDisparity->rows;

	if (mGroundLineExist)
		planeEstimator.compute(mDisparity, &mGroundMap);
	else
		planeEstimator.compute(mDisparity);


	planeEstimator.getRoadPlane(mPlane);
	//cout<<"plane parameter:"<<plane[0]<<" "<<plane[1]<<" "<<plane[2]<<endl;

	//judge normal vector
	float normU = mPlane[0] / sqrtf(mPlane[0] * mPlane[0] + mPlane[1] * mPlane[1] + 1);
	float normV = mPlane[1] / sqrtf(mPlane[0] * mPlane[0] + mPlane[1] * mPlane[1] + 1);
	float normD = -1 / sqrtf(mPlane[0] * mPlane[0] + mPlane[1] * mPlane[1] + 1);

	//compute cosine of the angle
	float thetaCos = -normD; //normU * 0 + normV *0 + normD * -1
	float theta = acos(thetaCos) / PI * 180;
	std::cout << "slope angle = " << theta << std::endl;

	if (mGroundMapWithPlane.empty())
		mGroundMapWithPlane.create(height, width, CV_8UC1);

	mGroundMapWithPlane.setTo(255);

	if (theta < mGroundEstParam->mSlopeAngle)
		mPlaneValid = false;
	else
	{
		mPlaneValid = true;
		for (int v = 0; v < height; v++)
		{
			uint8_t* pRowInDisp = mDisparity->ptr<uchar>(v);    //TODO：这里要根据输入调整，以后要用到float的输入
			uint8_t* pRowInGndMap = mGroundMapWithPlane.ptr<uchar>(v);
			for (int u = 0; u < width; u++)
			{
				float currd = pRowInDisp[u];
				if (currd > 0)
				{
					float expectground = mPlane[0] * float(u) + float(v)*mPlane[1] + mPlane[2];
					if (fabs(currd - expectground) < mRANSACParam->mDThreshold + 1)
						pRowInGndMap[u] = 0;
				}
			}
		}
	}
}

void GroundEstimator::groundPlaneRefinement()
{
	std::vector<std::vector<Point> > contours0;
	std::vector<Vec4i> hierarchy;
	Mat contourImg(mGroundMapWithPlane.rows, mGroundMapWithPlane.cols, CV_8UC1, Scalar(0));
	Mat contourSrc;

	if (mPlaneValid) contourSrc = 255 - mGroundMapWithPlane;
	else contourSrc = 255 - mGroundMap;

	findContours(contourSrc, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	// iterate through all the top-level contours,
	// draw each connected component with its own random color
	int idx = 0;
	int maxarea = -1;
	int maxidx = -1;
	for (; idx >= 0; idx = hierarchy[idx][0])
	{
		int currerea = contourArea(contours0[idx]);
		if (currerea > maxarea)
		{
			maxidx = idx;
			maxarea = currerea;
		}
	}

	drawContours(contourImg, contours0, maxidx, 255,-1);
	mGroundUpperBound = std::vector<float>(mGroundMapWithPlane.cols, 0);

	for (int u = 0; u < contourImg.cols; u++)
	{
		mGroundUpperBound[u] = FLT_MAX;
		for (int v = 0; v < contourImg.rows; v++)
		{
			if (contourImg.at<uchar>(v, u) == 255 && v < mGroundUpperBound[u])
				mGroundUpperBound[u] = v;
		}
	}

	for (int u = 0; u < contourImg.cols; u++)
	{
		if (mGroundUpperBound[u] != FLT_MAX)
			for (int v = contourImg.rows - 1; v >= mGroundUpperBound[u]; v--)
				contourImg.at<uchar>(v, u) = 255;
	}

	mGroundMapWithPlane = Scalar(255,255,255) - contourImg;
	//Mat temp;
	//cvtColor(mGroundMapWithPlane, temp, CV_GRAY2BGR);
	//for (int i = 0; i < contourImg.cols; i++)
	//{
	//	circle(temp, Point(i, mGroundUpperBound[i]), 2, Scalar(255, 255, 0), -1);
	//}
	//imshow("refine with line", temp);
}