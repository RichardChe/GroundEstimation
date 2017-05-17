#include "Visualizer.h"

Visualizer::Visualizer(GroundEstimator* groundEstimaor)
	:mGroundEstimator(groundEstimaor)
{
}

Visualizer::~Visualizer(){}

Mat Visualizer::showGroundWithImage(Scalar color)
{
	int width  = mGroundEstimator->getGroundMap().cols;
	int height = mGroundEstimator->getGroundMap().rows;
	Mat result(height, width, CV_8UC3);

	Mat mask(height,width,CV_8UC3);
	Mat LeftImgBGR;
	cvtColor(*mGroundEstimator->mLeftImg, LeftImgBGR, CV_GRAY2BGR);
	resize(LeftImgBGR, LeftImgBGR, Size(width, height));

	mask.setTo(color, mGroundEstimator->getGroundMap() == 0);
	addWeighted(mask, 0.3, LeftImgBGR, 0.7,0,result);

	return result;
}
