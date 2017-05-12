#pragma once

#include "GroundEstimator.h"

class Visualizer
{
public:
	GroundEstimator* mGroundEstimator;
	Visualizer(GroundEstimator* groundEstimator);
	~Visualizer();
	Mat showGroundWithImage(Scalar color = Scalar(255,255,0));
};