#include <opencv2/opencv.hpp>

#include "SGMStereo.h"
#include "GroundEstimator.h"
#include "RANSACPlane.h"
#include "Visualizer.h"

#include <iostream>
#include <time.h>
#include <iomanip>
#include "Config.h"

using namespace cv;
using namespace std;

int main()
{
	SGMStereo::Parameter param;
	param.mMaxDisp = 128;
	SGMStereo stereo(param);
	GroundEstimator::Parameter groundParam;
	RANSACPlane::Parameter RANSACParam;
	GroundEstimator groundEstimator(&groundParam, &param, &RANSACParam);
	Visualizer visualizer(&groundEstimator); 

	for (int i = Config::frameBegin; i < Config::frameEnd; i++)
	{
		stringstream leftFile, rightFile;
		leftFile << Config::imageDir << "/" << Config::imageBaseName[0] << setfill('0') << setw(Config::idxWidth) << i << Config::imageFormat;
		rightFile << Config::imageDir << "/" << Config::imageBaseName[1] << setfill('0') << setw(Config::idxWidth) << i << Config::imageFormat;

		Mat leftImg = imread(leftFile.str(), CV_LOAD_IMAGE_GRAYSCALE);
		Mat rightImg = imread(rightFile.str(), CV_LOAD_IMAGE_GRAYSCALE);

		clock_t beginTime, endTime;
		beginTime = clock();
		stereo.process(leftImg, rightImg);
		endTime = clock();
		cout << float(endTime - beginTime) / (float)CLOCKS_PER_SEC << endl;
		
		// in linux(gcc) this line is necessary in case of the "temporal address error"
		Mat disparity = stereo.getDisparity();
		groundEstimator.compute(&leftImg, &rightImg, &disparity);

		imshow("left", leftImg);
		imshow("disp", stereo.getDisparity());
		imshow("ground mask",visualizer.showGroundWithImage());
		
		waitKey(10);
	}

	system("pause");

	return 0;
}