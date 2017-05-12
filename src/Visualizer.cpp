#include "Visualizer.h"

Visualizer::Visualizer(GroundEstimator* groundEstimaor)
	:mGroundEstimator(groundEstimaor)
{
}

Visualizer::~Visualizer(){}

Mat Visualizer::showGroundWithImage(Scalar color)
{
	int width =  mGroundEstimator->mLeftImg->cols;
	int height = mGroundEstimator->mRightImg->rows;
	Mat result(height, width, CV_8UC3);

	Mat mask(height,width,CV_8UC3);
	Mat LeftImgBGR;
	cvtColor(*mGroundEstimator->mLeftImg, LeftImgBGR, CV_GRAY2BGR);
	mask.setTo(color, mGroundEstimator->getGroundMap() == 0);
	addWeighted(mask, 0.5, LeftImgBGR, 0.5,0,result);

	return result;
}
