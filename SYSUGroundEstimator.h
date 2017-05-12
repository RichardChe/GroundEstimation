#ifndef ROAD_ESTIMATOR_H
#define ROAD_ESTIMATOR_H

#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

#define SLOPEANGLE 10
//#define SHOW_GRID_MAP

struct GridParam
{
	float maxZ;
	float maxX;
	float gridSize;
};

struct Calib
{
	float cv,cu,base,f,vehicle_height;
};

struct ObsParam
{
	float maxZ;
	float minY;
};

class SYSUGroundEstimator
{
public:

	//SYSUGroundEstimator(Calib _calib, ObsParam _obsparam);
	SYSUGroundEstimator(Calib _calib, ObsParam _obsparam, GridParam _gridParam);
	~SYSUGroundEstimator();

	GridParam gridParam;
	Calib calib;
	ObsParam obsParam;
	Mat gridMap_;

	bool compute(Mat left, Mat right, Mat& ground,Mat& disp);

};



#endif
