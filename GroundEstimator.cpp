#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "cv.h"
#include "highgui.h"

#include <vector>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#include "SYSUGroundEstimator.h"
#include "SYSUStereo.h"

#include <time.h>

#define PI 3.14159265
#define SWAP(a,b) {temp=a;a=b;b=temp;}

using namespace std;
using namespace cv;

void obstacleExtraction(int* groundPoints, Mat obstacleMap, Mat x, Mat y, Mat z, Mat &extractedObs);
void drawObstacle(vector<Vec3i> uBottomUp, vector<int> segments, Mat &obstacleImage);

cv::Scalar obsColor[8] = { cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255),
cv::Scalar(255, 128, 0), cv::Scalar(128, 255, 0),
cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255),
cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0) };

class ColorEncoder
{
public:
	enum EncodingMode
	{
		ENCODING_RGB,   // Red -> Green -> Blue
		ENCODING_BGR,   // Blue -> Green -> Red

		ENCODING_RG,    // Red -> Green
		ENCODING_GR,    // Green -> Red

		ENCODING_BR,  // Blue -> Red
		ENCODING_RB,   // Red -> Blue

		ENCODING_HUE,

		ENCODING_BLRY,
		ENCODING_YRBL
	};

	enum EncodingAlg
	{
		ENCODING_SINE,   // compute color from value using sine function
		ENCODING_LINEAR  // using linear function
	};

	ColorEncoder(EncodingMode _encodingMode, float _rangeMin, float _rangeMax, EncodingAlg _alg = ENCODING_SINE);
	bool valueToColor(float value, uint8_t& r, uint8_t& g, uint8_t& b);

private:
	EncodingMode encodingMode;
	EncodingAlg encodingAlg;

	// range of value
	float rangeMax;
	float rangeMin;

};

void encodeBR(float value, float rangeMin, float rangeMax, uint8_t& r, uint8_t& g, uint8_t& b, bool inverse, ColorEncoder::EncodingAlg alg);
void encodeBGR(float value, float rangeMin, float rangeMax, uint8_t& r, uint8_t& g, uint8_t& b, bool inverse, ColorEncoder::EncodingAlg alg);
void encodeRG(float value, float rangeMin, float rangeMax, uint8_t& r, uint8_t& g, uint8_t& b, bool inverse, ColorEncoder::EncodingAlg alg);
void encodeHue(const float value, const float rangeMin, const float rangeMax, uint8_t& r, uint8_t& g, uint8_t& b, bool inverse);
void encodeBlackRedYellow(float value, float rangeMin, float rangeMax, uint8_t& r, uint8_t& g, uint8_t& b, bool inverse);

void getRgbFromHsv(const float h, const float s, const float v, uint8_t& r, uint8_t& g, uint8_t& b);

ColorEncoder::ColorEncoder(ColorEncoder::EncodingMode _encodingMode, float _rangeMin, float _rangeMax, ColorEncoder::EncodingAlg _alg)
{
	encodingMode = _encodingMode;
	rangeMin = _rangeMin;
	rangeMax = _rangeMax;
	encodingAlg = _alg;
}

bool ColorEncoder::valueToColor(float value, uint8_t& r, uint8_t& g, uint8_t& b)
{
	if (encodingMode == ENCODING_BGR)
		encodeBGR(value, rangeMin, rangeMax, r, g, b, false, encodingAlg);

	else if (encodingMode == ENCODING_RGB)
		encodeBGR(value, rangeMin, rangeMax, r, g, b, true, encodingAlg);

	else if (encodingMode == ENCODING_BR)
		encodeBR(value, rangeMin, rangeMax, r, g, b, false, encodingAlg);

	else if (encodingMode == ENCODING_RB)
		encodeBR(value, rangeMin, rangeMax, r, g, b, true, encodingAlg);

	else if (encodingMode == ENCODING_RG)
		encodeRG(value, rangeMin, rangeMax, r, g, b, false, encodingAlg);

	else if (encodingMode == ENCODING_GR)
		encodeRG(value, rangeMin, rangeMax, r, g, b, true, encodingAlg);

	else if (encodingMode == ENCODING_HUE)
		encodeHue(value, rangeMin, rangeMax, r, g, b, true);

	else if (encodingMode == ENCODING_BLRY)
		encodeBlackRedYellow(value, rangeMin, rangeMax, r, g, b, false);

	else if (encodingMode == ENCODING_YRBL)
		encodeBlackRedYellow(value, rangeMin, rangeMax, r, g, b, true);

	else
	{
		r = 0;
		g = 0;
		b = 0;
		return false;
	}

	return true;
}

void encodeBR(float value, float rangeMin, float rangeMax, uint8_t& r, uint8_t& g, uint8_t& b, bool inverse, ColorEncoder::EncodingAlg alg)
{
	float dX = (value - rangeMin) / (rangeMax - rangeMin);

	if (inverse)
		dX = 1 - dX;

	if (dX < 0.0)
	{
		r = 0;
		g = 0;
		b = 255;
	}
	else if (dX >= (.99 * .99)) //
	{
		r = 255;
		g = 0;
		b = 0;
	}
	else
	{
		if (alg == ColorEncoder::ENCODING_LINEAR)
		{
			dX = sqrt(dX);  //?
			b = int(255.0 * (-dX + 1));
			g = 0;
			r = int(255.0 * dX);
		}

		else
		{
			dX = sqrt(dX);

			int temp_r = int(255.0 * sin(dX * M_PI / 1.8));
			int temp_g = int(255.0 * sin(dX * M_PI / 0.9));
			int temp_b = int(255.0 * (1.0 - sin(dX * M_PI / 1.8)));

			temp_r = (temp_r + 255) / 2;
			temp_g = (temp_g + 255) / 2;
			temp_b = (temp_b + 255) / 2;

			r = temp_r;
			g = temp_g;
			b = temp_b;
		}
	}
}

void encodeBGR(float value, float rangeMin, float rangeMax, uint8_t& r, uint8_t& g, uint8_t& b, bool inverse, ColorEncoder::EncodingAlg alg)
{
	float dX = (value - rangeMin) / (rangeMax - rangeMin);

	if (inverse)
		dX = 1 - dX;

	if (dX < 0.0)
	{
		r = 0;
		g = 0;
		b = 255;         // Blue
	}
	else if (dX >= (.99 * .99))
	{
		r = 255;
		g = 0;
		b = 0;           // Red
	}
	else
	{
		if (alg == ColorEncoder::ENCODING_LINEAR)
		{
			float r_float = 255.0 * (dX);
			float g_float = dX > 0.5 ? (-2 * dX + 2) : 2 * dX;
			g_float *= 255;
			//float g_float = 255.0 * (2 * dX - 0.5 ) ;
			float b_float = 255.0 * (-dX + 1);

			r = uint8_t(r_float);
			g = uint8_t(g_float);
			b = uint8_t(b_float);
		}

		else
		{
			int temp_r = 255.0 * cos(dX * M_PI + M_PI);
			int temp_g = 255.0 * sin(2.0 * dX * M_PI - (M_PI / 2.0));
			int temp_b = 255.0 * cos(dX * M_PI);

			temp_r = (temp_r + 255) / 2;
			temp_g = (temp_g + 255) / 2;
			temp_b = (temp_b + 255) / 2;


			r = temp_r;
			g = temp_g;
			b = temp_b;
		}
	}
}

void encodeRG(float value, float rangeMin, float rangeMax, uint8_t& r, uint8_t& g, uint8_t& b, bool inverse, ColorEncoder::EncodingAlg alg)
{
	float dX = (value - rangeMin) / (rangeMax - rangeMin);

	if (inverse)
		dX = 1 - dX;

	b = 0;    // Blue component is always 0

	if (dX < 0.0)
	{
		r = 0;
		g = 255;         // Green
	}
	else if (dX >= 1.0)
	{
		r = 255;
		g = 0;           // Red
	}

	else
	{
		if (alg == ColorEncoder::ENCODING_LINEAR)  //Unfortunately, there is no difference in RG encoding here. :(
		{
			//r   = int(300*dX-45);
			r = int(255 * dX);
			g = int(255 * (-dX + 1));

			//r = float(r)*0.7+128*0.7;
			//g = float(g)*0.3*128*0.3;
		}
		else
		{
			r = int(255 * dX);
			g = int((-255 * dX) + 255.0);
		}
	}
}

void encodeHue(const float value, const float rangeMin, const float rangeMax, uint8_t& r, uint8_t& g, uint8_t& b, bool  inverse)
{
	float range_f = rangeMax - rangeMin;

	float val_f = 360 * (value - rangeMin) / range_f;

	if (inverse)
		val_f = 360 - val_f;

	float h = val_f;
	float s = .75f;
	float v = .8f;

	getRgbFromHsv(h, s, v, r, g, b);
}

void  getRgbFromHsv(const float h, const float s, const float v, uint8_t& r, uint8_t& g, uint8_t& b)
{
	float h_f = h / 60.f;
	int   h_i = ((int)h_f) % 6;
	float f_f = h_f - h_i;

	float p_f = v * (1.f - s);
	float q_f = v * (1.f - f_f * s);
	float t_f = v * (1.f - (1.f - f_f) * s);

	float r_f, g_f, b_f;

	switch (h_i)
	{
	case 0:
	{
			  r_f = v;
			  g_f = t_f;
			  b_f = p_f;
	}
		break;

	case 1:
	{
			  r_f = q_f;
			  g_f = v;
			  b_f = p_f;
	}
		break;

	case 2:
	{
			  r_f = p_f;
			  g_f = v;
			  b_f = t_f;
	}
		break;

	case 3:
	{
			  r_f = p_f;
			  g_f = q_f;
			  b_f = v;
	}
		break;

	case 4:
	{
			  r_f = t_f;
			  g_f = p_f;
			  b_f = v;
	}
		break;

	case 5:
	default:
	{
			   r_f = v;
			   g_f = p_f;
			   b_f = q_f;
	}
		break;
	}
	r = uint8_t(r_f * 255.f);
	g = uint8_t(g_f * 255.f);
	b = uint8_t(b_f * 255.f);

}

void encodeBlackRedYellow(float value, float rangeMin, float rangeMax, uint8_t& r, uint8_t& g, uint8_t& b, bool inverse)
{
	float dX = (value - rangeMin) / (rangeMax - rangeMin);

	if (inverse)
		dX = 1 - dX;

	float  rfloat, gfloat;


	// Encode red.
	gfloat = 255 * dX;

	// Encode red.
	rfloat = (1 - dX);
	rfloat *= rfloat;
	rfloat *= rfloat;
	rfloat = (1 - rfloat) * 255;

	//fr_color.set ( (int)r, (int)g, (int)0 );
	b = 0;
	r = (uint8_t)rfloat;
	g = (uint8_t)gfloat;
}

bool disp2Color(uint8_t dispValue, uint8_t& r, uint8_t& g, uint8_t& b, const int maxdisp)
{
	// color map
	float map[8][4] = { { 0, 0, 0, 114 }, { 0, 0, 1, 185 }, { 1, 0, 0, 114 }, { 1, 0, 1, 174 },
	{ 0, 1, 0, 114 }, { 0, 1, 1, 185 }, { 1, 1, 0, 114 }, { 1, 1, 1, 0 } };
	float sum = 0;
	for (int32_t i = 0; i < 8; i++)
		sum += map[i][3];

	float weights[8]; // relative weights
	float cumsum[8];  // cumulative weights
	cumsum[0] = 0;

	for (int32_t i = 0; i < 7; i++) {
		weights[i] = sum / map[i][3];
		cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
	}

	// get normalized value
	float val = std::min(std::max(float(dispValue) / float(maxdisp), 0.0f), 1.0f);

	// find bin
	int32_t i;
	for (i = 0; i < 7; i++)
	if (val < cumsum[i + 1])
		break;

	// compute red/green/blue values
	float   w = 1.0 - (val - cumsum[i])*weights[i];
	r = (uint8_t)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
	g = (uint8_t)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
	b = (uint8_t)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);

	return true;
}

class ObsDetector
{
public:
	Mat obstacleMap;
	Mat x;
	Mat y;
	Mat z;
	Calib calib;
	ObsParam obsparam;

	ObsDetector(Calib _calib, ObsParam _obsparam);
	void process(Mat disp, Mat groundMap);
};

struct disp
{
	float u, v, d;
	disp(float u, float v, float d) : u(u), v(v), d(d) {}
};

class RoadEstimator
{
public:

	struct RansacROI
	{
		int startX;
		int startY;
		int endX;
		int endY;
		RansacROI()
		{
			startX = 0;
			endX = 0;
			startY = 0;
			endY = 0;
		}

		RansacROI(int x0, int x1, int y0, int y1)
		{
			startX = x0;
			endX = x1;
			startY = y0;
			endY = y1;
		}

		RansacROI(const RansacROI& roi)
		{
			startX = roi.startX;
			startY = roi.startY;
			endX = roi.endX;
			endY = roi.endY;
		}
	};

	RoadEstimator();
	bool compute(float* disparity, const int width, const int height, uchar* groundMap, bool lineExist);

	void computeRoadPlaneEstimate();
	void getRoadPlane(float *plane);
private:

	RansacROI mRoi;

	cv::Mat mDisparity;

	int mWidth;
	int mHeight;
	int mNumSamples;
	float mDThreshold;
	float mPlane[3];
	uchar* mGroundMap;
};

struct GroundEstimatorParam
{
	int uDispThreshold;
	float sigmaGaussian;
	int dH;
	bool multiLine;
	float slopeAngle;
};

class GroundEstimator
{
public:
	GroundEstimator(GroundEstimatorParam param);
	RoadEstimator roadEstimator;

	~GroundEstimator()
	{
		if (groundMap)
			free(groundMap);
		if (groundMapRansac)
			free(groundMapRansac);

	}
	void create(int widht, int height, int maxDisp);
	void process(uint8_t*disp, int width, int height, int maxDisp);
	int uDispThreshold;
	float sigmaGaussian;
	int dH;
	uint8_t *groundMap;
	uint8_t *groundMapRansac;
	bool multiLine;
	bool bDetectGround;
	float slopeAngle;

};

void computeUDisparityMap(uint8_t* dispData, uint16_t* uDispData, int width, int height, int maxDisp);
void extractObstacle(uint16_t *uDispData, uint8_t* disp, uint8_t* obstacleMap, int width, int height, int maxDisp, int uDispThreshold);
void computeNewVDispMap(uint8_t *dispData, uint8_t* obstacleMap, uint16_t* vDispData, const int width, const int height, const int maxDisp);
bool extractGround(Mat vDisp8U, float sigma, int width, int height, int dH, float *vGroundOffset);
void generateGroundMap(float* vGroundOffset, uint8_t* disp, int width, int height, uint8_t *groundMap);
bool extractGroundMultiLine(Mat vDisp8U, float sigma, int width, int height, int dH, float *vGroundOffset);
void refineGroundLine(Mat vDispFiltered, float* vGround, float *vGroundOffset, int maxDisp, int dH);

GroundEstimator::GroundEstimator(GroundEstimatorParam param)
{
	uDispThreshold = param.uDispThreshold;
	sigmaGaussian = param.sigmaGaussian;
	dH = param.dH;
	multiLine = param.multiLine;
	slopeAngle = param.slopeAngle;
	groundMap = NULL;
	groundMapRansac = NULL;
}

void GroundEstimator::process(uint8_t*disp, int width, int height, int maxDisp)
{
	uint16_t *uDisp = (uint16_t*)malloc(sizeof(uint16_t)* width * maxDisp);
	uint16_t *vDisp = (uint16_t*)malloc(sizeof(uint16_t)*height * maxDisp);
	uint8_t *obstacleMap = (uint8_t*)malloc(sizeof(uint8_t)* width*height);

	//uint8_t *groundMap = (uint8_t*) malloc(sizeof(uint8_t) * width*height);
	if (groundMap == NULL)
	{
		groundMap = (uint8_t*)malloc(width*height*sizeof(uint8_t));
	}
	if (groundMapRansac == NULL)
	{
		groundMapRansac = (uint8_t*)malloc(width*height*sizeof(uint8_t));
	}

	float *vGroundOffset = (float*)malloc(height*sizeof(float)); //ÿһ��dֵ��Ӧ�� ���桪���ǵ��� �ֽ���

	computeUDisparityMap(disp, uDisp, width, height, maxDisp);
	extractObstacle(uDisp, disp, obstacleMap, width, height, maxDisp, uDispThreshold);
	computeNewVDispMap(disp, obstacleMap, vDisp, width, height, maxDisp);

	Mat vDispMat(height, maxDisp, CV_16UC1, vDisp);
	Mat vDisp8U;
	vDispMat.convertTo(vDisp8U, CV_8UC1);
	bool lineExist = false;

	if (multiLine == false)
	{
		if (extractGround(vDisp8U, sigmaGaussian, width, height, dH, vGroundOffset))
		{
			generateGroundMap(vGroundOffset, disp, width, height, groundMap);
		}
		else
		{
			memset(groundMap, 0xff, sizeof(uint8_t)*width*height);
			memset(groundMapRansac, 0xff, sizeof(uint8_t)*width*height);
		}
	}

	else
	{
		lineExist = extractGroundMultiLine(vDisp8U, sigmaGaussian, width, height, dH, vGroundOffset);
		if (lineExist)
		{
			generateGroundMap(vGroundOffset, disp, width, height, groundMap);
		}
		else
		{
			memset(groundMap, 0xff, sizeof(uint8_t)*width*height);
			memset(groundMapRansac, 0xff, sizeof(uint8_t)*width*height);
		}
	}
	//Mat groundMapMat(height,width,CV_8UC1,groundMap);
	//imshow("ground",groundMapMat);
	//waitKey();

	float *dispFlt = (float*)malloc(sizeof(float)*width*height);
	for (int i = 0; i < height*width; i++)
	{
		*(dispFlt + i) = (float)(*(disp + i));
	}

	roadEstimator.compute(dispFlt, width, height, groundMap, lineExist);
	float plane[3];
	roadEstimator.getRoadPlane(plane);
	//cout<<"plane parameter:"<<plane[0]<<" "<<plane[1]<<" "<<plane[2]<<endl;

	//judge normal vector
	float normU = plane[0] / sqrtf(plane[0] * plane[0] + plane[1] * plane[1] + 1);
	float normV = plane[1] / sqrtf(plane[0] * plane[0] + plane[1] * plane[1] + 1);
	float normD = -1 / sqrtf(plane[0] * plane[0] + plane[1] * plane[1] + 1);

	//compute cosine of the angle
	float thetaCos = -normD; //normU * 0 + normV *0 + normD * -1
	float theta = acos(thetaCos) / PI * 180;
	cout << "slope angle = " << theta << endl;

	if (theta < slopeAngle)
	{
		bDetectGround = false;
		for (int u = 0; u < width; u++)
		{
			for (int v = 0; v < height; v++)
			{
				groundMapRansac[v*width + u] = 255;
			}
		}
	}

	else
	{
		bDetectGround = true;
		for (int u = 0; u < width; u++)
		{
			for (int v = 0; v < height; v++)
			{
				float currd = dispFlt[v*width + u];
				if (currd >0)
				{
					float expectground = plane[0] * float(u) + float(v)*plane[1] + plane[2];
					if (fabs(currd - expectground) < 2)
						groundMapRansac[v*width + u] = 0;
					else
						groundMapRansac[v*width + u] = 255;
				}
				else
					groundMapRansac[v*width + u] = 255;
			}
		}
	}

	//for (int u = 0; u < width; u++)
	//{
	//	for (int v = 0; v < height; v++)
	//	{
	//		float currd  = dispFlt[v*width+u];
	//		if(currd >0)
	//		{
	//			float expectground = plane[0]*float(u) + float(v)*plane[1] + plane[2];
	//			if(fabs(currd - expectground) < 2)
	//				groundMapRansac[v*width+u] = 0;
	//			else
	//				groundMapRansac[v*width+u] = 255;
	//		}
	//		else
	//			groundMapRansac[v*width+u] = 255;
	//	}
	//}

	free(uDisp);
	free(vDisp);
	free(obstacleMap);
	free(vGroundOffset);
	free(dispFlt);

	//free(groundMap);
}

void computeUDisparityMap(uint8_t* dispData, uint16_t* uDispData, int width, int height, int maxDisp)
{
	memset(uDispData, 0, sizeof(uint16_t)*maxDisp*width);

	for (int u = 0; u < width; u++)
	{
		for (int v = 0; v < height; v++)
		{
			uint8_t currDisp = *(dispData + width * v + u);

			if (currDisp>0 && currDisp < maxDisp)
			{
				(*(uDispData + width * currDisp + u))++;
			}
		}
	}
}

void extractObstacle(uint16_t *uDispData, uint8_t* disp, uint8_t* obstacleMap, int width, int height, int maxDisp, int uDispThreshold)
{
	memset(obstacleMap, 0, sizeof(uint8_t)*width*height);

	for (int u = 0; u < width; u++)
	{
		for (int v = 0; v < height; v++)
		{
			uint8_t currDisp = *(disp + v*width + u);
			if (currDisp < maxDisp)
			{
				uint16_t currUDispValue = *(uDispData + currDisp * width + u);
				if (currUDispValue > uDispThreshold)
				{
					*(obstacleMap + v*width + u) = 255;
				}
			}
		}
	}
}

void computeNewVDispMap(uint8_t *dispData, uint8_t* obstacleMap, uint16_t* vDispData, const int width, const int height, const int maxDisp)
{
	memset(vDispData, 0, sizeof(short)*height*maxDisp);
	for (int u = 0; u < width; u++)
	{
		for (int v = 0; v < height; v++)
		{
			uint8_t currDisp = *(dispData + v *width + u);
			if (*(obstacleMap + v*width + u) == 0 && currDisp >0)
				(*(vDispData + v*maxDisp + currDisp))++;
		}
	}
}

bool extractGround(Mat vDisp8U, float sigma, int width, int height, int dH, float *vGroundOffset)
{
	int maxDisp = vDisp8U.cols;
	float *vGround = (float*)malloc(maxDisp*sizeof(float));       //ÿһ��dֵ��Ӧ��v f(d) = v

	Mat vDispFiltered = vDisp8U;
	//imshow("v disp",vDisp8U);
	//waitKey();
	GaussianBlur(vDisp8U, vDispFiltered, Size(5, 5), sigma);
	Canny(vDispFiltered, vDispFiltered, 40, 150);

	vector<Vec2f> lines;
	HoughLines(vDispFiltered, lines, 1, CV_PI / 180, 50, 0, 0);

	if (lines.size() == 0)
	{
		cout << "Ground detection failed. No ground Line..." << endl;
		//todo:prior
		//imshow("ground line",vDispFiltered);
		//waitKey();
		free(vGround);
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

	//printf("k = %.2f \n",k);
	//printf("x0 = %.2f , y0 = %.2f\n",x0,y0);

	Mat vDispColor;
	cvtColor(vDispFiltered, vDispColor, CV_GRAY2RGB);
	//imshow("ground line",vDispColor);
	//waitKey(1);

	for (int currDisp = 0; currDisp < maxDisp; currDisp++)
	{
		float currV = k* (currDisp - x0) + y0;
		vGround[currDisp] = currV;
		vGroundOffset[currDisp] = currV + dH;
		circle(vDispColor, Point(currDisp, vGround[currDisp]), 1, Scalar(255, 255, 0), 1);
		circle(vDispColor, Point(currDisp, vGroundOffset[currDisp]), 1, Scalar(255, 0, 0), 1);
	}
	//refineGroundLine(vDispFiltered,vGround,vGroundOffset,maxDisp,dH);

	for (int currDisp = 0; currDisp < maxDisp; currDisp++)
	{
		//circle(vDispColor,Point(currDisp,vGround[currDisp]),2,Scalar(255,255,0),1);
		//circle(vDispColor,Point(currDisp,vGroundOffset[currDisp]),2,Scalar(255,0,0),1);
		//circle(vDispColor,Point(currDisp,vGround[currDisp]+dH),1,Scalar(0,255,0),1);
	}

	//imshow("ground line",vDispColor);
	//waitKey();

	free(vGround);
	return true;
}

bool extractGroundMultiLine(Mat vDisp8U, float sigma, int width, int height, int dH, float *vGroundOffset)
{
	int maxDisp = vDisp8U.cols;
	float *vGround = (float*)malloc(maxDisp*sizeof(float));       //ÿһ��dֵ��Ӧ��v f(d) = v

	Mat vDispFiltered = vDisp8U;
	GaussianBlur(vDisp8U, vDispFiltered, Size(5, 5), sigma);
	Canny(vDispFiltered, vDispFiltered, 50, 300);

	Mat vDispFilteredLeft = vDispFiltered(Rect(0, 0, maxDisp / 2, height));
	Mat vDispFilteredRight = vDispFiltered(Rect(maxDisp / 2, 0, maxDisp / 2, height));

	vector<Vec2f> linesLeft;
	vector<Vec2f> linesRight;

	HoughLines(vDispFilteredLeft, linesLeft, 1, CV_PI / 180, 50, 0, 0);
	HoughLines(vDispFilteredRight, linesRight, 1, CV_PI / 180, 50, 0, 0);

	if (linesLeft.size() == 0 && linesRight.size() == 0)
	{
		cout << "Ground detection failed. No ground Line..." << endl;
		//todo:prior
		free(vGround);
		return false;
	}

	Mat vDispColor;
	cvtColor(vDispFiltered, vDispColor, CV_GRAY2RGB);

	bool enableLeftLine = true;
	bool enableRightLine = true;

	float thetaLeft, rhoLeft, cosThetaLeft, sinThetaLeft;
	float x0Left, y0Left, x1Left, y1Left, kLeft;
	float thetaRight, rhoRight, cosThetaRight, sinThetaRight;
	float x0Right, y0Right, x1Right, y1Right, kRight;

	if (linesLeft.size() == 0)
	{
		thetaLeft = thetaRight = linesRight[0][1];
		rhoLeft = rhoRight = linesRight[0][0];
		enableLeftLine = false;
	}
	else if (linesRight.size() == 0)
	{
		thetaLeft = thetaRight = linesLeft[0][1];
		rhoLeft = rhoRight = linesLeft[0][0];
		enableRightLine = false;
	}
	else
	{
		thetaLeft = linesLeft[0][1];
		rhoLeft = linesLeft[0][0];

		thetaRight = linesRight[0][1];
		rhoRight = linesRight[0][0];
	}
	cosThetaLeft = cos(thetaLeft);  sinThetaLeft = sin(thetaLeft);
	x0Left = rhoLeft * cosThetaLeft;  y0Left = rhoLeft * sinThetaLeft;
	x1Left = x0Left + 1000 * (-sinThetaLeft); y1Left = y0Left + 1000 * (cosThetaLeft);
	kLeft = (y1Left - y0Left) / (x1Left - x0Left);

	cosThetaRight = cos(thetaRight);
	sinThetaRight = sin(thetaRight);

	x0Right = rhoRight * cosThetaRight;
	y0Right = rhoRight * sinThetaRight;
	x1Right = x0Right + 1000 * (-sinThetaRight);
	y1Right = y0Right + 1000 * (cosThetaRight);

	kRight = (y1Right - y0Right) / (x1Right - x0Right);

	for (int currDisp = 0; currDisp < maxDisp / 2; currDisp++)
	{
		float currV = kLeft* (currDisp - x0Left) + y0Left;
		vGround[currDisp] = currV;
		vGroundOffset[currDisp] = currV + dH;
		circle(vDispColor, Point(currDisp, vGround[currDisp]), 1, Scalar(255, 255, 0), 2);
		circle(vDispColor, Point(currDisp, vGroundOffset[currDisp]), 1, Scalar(255, 0, 0), 2);
	}

	for (int currDisp = maxDisp / 2; currDisp < maxDisp; currDisp++)
	{
		float currV;
		if (enableRightLine)
			currV = kRight* (currDisp - maxDisp / 2 - x0Right) + y0Right;
		else
			currV = kRight* (currDisp - x0Right) + y0Right;

		vGround[currDisp] = currV;
		vGroundOffset[currDisp] = currV + dH;
		circle(vDispColor, Point(currDisp, vGround[currDisp]), 1, Scalar(255, 255, 0), 1);
		circle(vDispColor, Point(currDisp, vGroundOffset[currDisp]), 1, Scalar(255, 0, 0), 1);
	}

	return true;
}

void generateGroundMap(float* vGroundOffset, uint8_t* disp, int width, int height, uint8_t *groundMap)
{
	for (int u = 0; u < width; u++)
	{
		for (int v = 0; v < height; v++)
		{
			int currDisp = *(disp + v * width + u);
			if (currDisp >0)
			{
				if (v >= vGroundOffset[currDisp])
					//rawImg.at<uchar>(v,u) = 0;
					*(groundMap + v*width + u) = 0;
				else
					*(groundMap + v*width + u) = 255;
			}
			else
				*(groundMap + v*width + u) = 255;
		}
	}
}

void refineGroundLine(Mat vDispFiltered, float* vGround, float *vGroundOffset, int maxDisp, int dH)  //stick to the pt (highest, near dhpxs)
{
	cout << "maxdisp:" << maxDisp << endl;

	uint8_t* vDispFilteredData = vDispFiltered.data;
	int height = vDispFiltered.rows;
	int width = vDispFiltered.cols;

	//int *find  = (int*) malloc(sizeof(int) * maxDisp);
	//memset(find,0,sizeof(int)*maxDisp);

	for (int currDisp = 0; currDisp < maxDisp; currDisp++)
	{
		float currV = vGround[currDisp];  //�ҵ�ÿһ���Ӳ�ֵ��Ӧ��V
		//cout<<"curr disp:"<<currDisp<<endl;

		int low = currV + dH;  //ȷ������������Χ DHΪ����
		//if(low < 0) low = 10;

		int high = currV - dH / 2;
		if (high >= height) high = height - 1;

		float newV = currV;

		for (int v = currV; v >= low; v--)    //�������������ҵ�dh��Χ�����Ϸ����ߵ�
		{
			//cout<<"v = "<<v <<" currDisp = "<<currDisp<<endl;
			//if(*(vDispFilteredData + v * width + currDisp) == 255)
			if (vDispFiltered.at<uchar>(v, currDisp) == 255)
				newV = v;
		}

		if (newV == currV)   //�������������ҵ�dh��Χ���������ֱ�ߵĵ�
		{
			for (int v = currV; v <= high; v++)
			{
				if (vDispFiltered.at<uchar>(v, currDisp) == 255)
					newV = v;
			}
		}

		vGroundOffset[currDisp] = newV;
	}

}

vector<disp> sparseDisparityGrid(float* D, const int width, const int height, RoadEstimator::RansacROI roi, int32_t step_size);
vector<disp> sparseDisparityGrid(float* D, const int width, const int height, int32_t step_size, uint8_t* groundMap);


inline uint32_t getAddressOffsetImage(const int32_t& u, const int32_t& v, const int32_t& width) {
	return v*width + u;
}

float **allocateMatrix(int32_t nrow, int32_t ncol) {
	float **m;
	m = (float**)malloc(nrow*sizeof(float*));
	m[0] = (float*)calloc(nrow*ncol, sizeof(float));
	for (int32_t i = 1; i < nrow; i++) m[i] = m[i - 1] + ncol;
	return m;
}

void freeMatrix(float **m) {
	free(m[0]);
	free(m);
}

void zeroMatrix(float** m, int32_t nrow, int32_t ncol) {
	for (int32_t i = 0; i < nrow; i++)
	for (int32_t j = 0; j < ncol; j++)
		m[i][j] = 0;
}

void printMatrix(float** m, int32_t nrow, int32_t ncol) {
	for (int32_t i = 0; i < nrow; i++) {
		for (int32_t j = 0; j < ncol; j++)
			cout << m[i][j] << " ";
		cout << endl;
	}
}

void drawRandomPlaneSample(vector<disp> &d_list, float *plane);
vector<disp> sparseDisparityGrid(float* D, const int width, const int height, RoadEstimator::RansacROI roi, int32_t step_size);
bool gaussJordanElimination(float **a, const int n, float **b, int m, float eps = 1e-8);
void leastSquarePlane(vector<disp> &d_list, vector<int32_t> &ind, float *plane);

void RoadEstimator::getRoadPlane(float *plane)
{
	for (int i = 0; i < 3; i++)
		plane[i] = mPlane[i];
}

void RoadEstimator::computeRoadPlaneEstimate()
{
	// random seed
	srand(time(NULL));

	// get list with disparities
	float *D = (float*)mDisparity.data;
	vector<disp> d_list;
	if (mRoi.startX > 0 & mRoi.startY > 0 && mRoi.endX > 0 && mRoi.endY > 0)
		d_list = sparseDisparityGrid(D, mWidth, mHeight, mRoi, 5);
	else
		d_list = sparseDisparityGrid(D, mWidth, mHeight, 5, mGroundMap);
	// loop variables
	vector<int32_t> curr_inlier;
	vector<int32_t> best_inlier;

	for (int32_t i = 0; i < mNumSamples; i++) {

		// draw random samples and compute plane
		drawRandomPlaneSample(d_list, mPlane);

		// find inlier
		curr_inlier.clear();
		for (int32_t i = 0; i < d_list.size(); i++)
		if (fabs(mPlane[0] * d_list[i].u + mPlane[1] * d_list[i].v + mPlane[2] - d_list[i].d)<mDThreshold)
			curr_inlier.push_back(i);

		// is this a better solution? (=more inlier)
		if (curr_inlier.size()>best_inlier.size())
		{
			best_inlier = curr_inlier;
		}
	}

	// reoptimize plane with inliers only
	if (curr_inlier.size() > 3)
		leastSquarePlane(d_list, best_inlier, mPlane);
}

bool RoadEstimator::compute(float* disparity, const int width, const int height, uint8_t* groundMap, bool lineExist)
{
	cv::Mat disparityMat(height, width, CV_32FC1, disparity);
	disparityMat.copyTo(mDisparity);

	mWidth = mDisparity.cols;
	mHeight = mDisparity.rows;
	mGroundMap = groundMap;

	if (mWidth == 0 || mHeight == 0)
	{
		cout << "Invalid disparity data in RoadEstimator::compute..." << endl;
		return false;
	}

	if (!lineExist)
		mRoi = RansacROI(200, mWidth - 200, int(float(mHeight) / 2.0 + 0.5), mHeight);
	else
		mRoi = RansacROI(-1, -1, -1, -1);

	computeRoadPlaneEstimate();

	return true;
}

RoadEstimator::RoadEstimator() :
mRoi(0, 0, 0, 0)
{
	mWidth = 0;
	mHeight = 0;
	mDThreshold = 1;
	mNumSamples = 300;
}

vector<disp> sparseDisparityGrid(float* D, const int width, const int height, RoadEstimator::RansacROI roi, int32_t step_size)
{

	// init list
	vector<disp> d_list;
	int x0 = roi.startX;
	int x1 = roi.endX;
	int y0 = roi.startY;
	int y1 = roi.endY;

	// loop through disparity image
	for (int32_t u = max(x0, 0); u <= min(x1, width - 1); u += step_size) {
		for (int32_t v = max(y0, 0); v <= min(y1, height - 1); v += step_size) {
			float d = *(D + getAddressOffsetImage(u, v, width));
			if (d >= 1)
				d_list.push_back(disp(u, v, d));
		}
	}

	// return list
	return d_list;
}

vector<disp> sparseDisparityGrid(float* D, const int width, const int height, int32_t step_size, uint8_t* groundMap)
{

	// init list
	vector<disp> d_list;

	// loop through disparity image
	for (int32_t u = 0; u < width; u += step_size) {
		for (int32_t v = 0; v < height; v += step_size) {
			float d = *(D + getAddressOffsetImage(u, v, width));
			if (d >= 1 && *(groundMap + getAddressOffsetImage(u, v, width)) == 0)
				d_list.push_back(disp(u, v, d));
		}
	}

	// return list
	return d_list;
}

void drawRandomPlaneSample(vector<disp> &d_list, float *plane)
{

	int32_t num_data = d_list.size();
	vector<int32_t> ind;

	// draw 3 measurements
	int32_t k = 0;
	while (ind.size() < 3 && k < 1000)
	{

		// draw random measurement
		int32_t curr_ind = rand() % num_data;

		// first observation
		if (ind.size() == 0)
		{
			// simply add
			ind.push_back(curr_ind);

			// second observation
		}
		else if (ind.size() == 1)
		{
			// check distance to first point
			float diff_u = d_list[curr_ind].u - d_list[ind[0]].u;
			float diff_v = d_list[curr_ind].v - d_list[ind[0]].v;
			if (sqrt(diff_u*diff_u + diff_v*diff_v) > 50)
				ind.push_back(curr_ind);
			// third observation
		}
		else
		{

			// check distance to line between first and second point
			float vu = d_list[ind[1]].u - d_list[ind[0]].u;
			float vv = d_list[ind[1]].v - d_list[ind[0]].v;
			float norm = sqrt(vu*vu + vv*vv);
			float nu = +vv / norm;
			float nv = -vu / norm;
			float ru = d_list[curr_ind].u - d_list[ind[0]].u;
			float rv = d_list[curr_ind].v - d_list[ind[0]].v;
			if (fabs(nu*ru + nv*rv) > 50)
				ind.push_back(curr_ind);
		}

		k++;
	}

	// return zero plane on error
	if (ind.size() == 0) {
		plane[0] = 0;
		plane[1] = 0;
		plane[2] = 0;
		return;
	}

	// find least squares solution
	leastSquarePlane(d_list, ind, plane);
}

void leastSquarePlane(vector<disp> &d_list, vector<int32_t> &ind, float *plane) {

	int32_t n = 3; int32_t m = 1;
	float** A = allocateMatrix(n, n);
	float** b = allocateMatrix(n, m);

	// find parameters
	for (vector<int32_t>::iterator it = ind.begin(); it != ind.end(); it++)
	{
		float u = d_list[*it].u;
		float v = d_list[*it].v;
		float d = d_list[*it].d;
		A[0][0] += u*u;
		A[0][1] += u*v;
		A[0][2] += u;
		A[1][1] += v*v;
		A[1][2] += v;
		A[2][2] += 1;
		b[0][0] += u*d;
		b[1][0] += v*d;
		b[2][0] += d;
	}
	A[1][0] = A[0][1];
	A[2][0] = A[0][2];
	A[2][1] = A[1][2];

	if (gaussJordanElimination(A, 3, b, 1))
	{
		plane[0] = b[0][0];
		plane[1] = b[1][0];
		plane[2] = b[2][0];
	}

	else
	{
		plane[0] = 0;
		plane[1] = 0;
		plane[2] = 0;
	}

	freeMatrix(A);
	freeMatrix(b);
}

bool gaussJordanElimination(float **a, const int n, float **b, int m, float eps) {

	// index vectors for bookkeeping on the pivoting

	// int32_t indxc[n];
	// int32_t indxr[n];
	// int32_t ipiv[n];
	//xieyuechao
	int32_t* indxc = (int32_t*)malloc(sizeof(int32_t)*n);
	int32_t* indxr = (int32_t*)malloc(sizeof(int32_t)*n);
	int32_t* ipiv = (int32_t*)malloc(sizeof(int32_t)*n);

	// loop variables
	int32_t i, icol, irow, j, k, l, ll;
	float big, dum, pivinv, temp;

	// initialize pivots to zero
	for (j = 0; j < n; j++) ipiv[j] = 0;

	// main loop over the columns to be reduced
	for (i = 0; i < n; i++) {

		big = 0.0;

		// search for a pivot element
		for (j = 0; j < n; j++)
		if (ipiv[j] != 1)
		for (k = 0; k < n; k++)
		if (ipiv[k] == 0)
		if (fabs(a[j][k]) >= big) {
			big = fabs(a[j][k]);
			irow = j;
			icol = k;
		}
		++(ipiv[icol]);

		// We now have the pivot element, so we interchange rows, if needed, to put the pivot
		// element on the diagonal. The columns are not physically interchanged, only relabeled:
		// indxc[i], the column of the ith pivot element, is the ith column that is reduced, while
		// indxr[i] is the row in which that pivot element was originally located. If indxr[i] !=
		// indxc[i] there is an implied column interchange. With this form of bookkeeping, the
		// solution b��s will end up in the correct order, and the inverse matrix will be scrambled
		// by columns.
		if (irow != icol) {
			for (l = 0; l < n; l++) SWAP(a[irow][l], a[icol][l])
			for (l = 0; l < m; l++) SWAP(b[irow][l], b[icol][l])
		}

		indxr[i] = irow; // We are now ready to divide the pivot row by the
		indxc[i] = icol; // pivot element, located at irow and icol.

		// check for singularity
		if (fabs(a[icol][icol]) < eps) {
			return false;
		}

		pivinv = 1.0 / a[icol][icol];
		a[icol][icol] = 1.0;
		for (l = 0; l < n; l++) a[icol][l] *= pivinv;
		for (l = 0; l < m; l++) b[icol][l] *= pivinv;

		// Next, we reduce the rows except for the pivot one
		for (ll = 0; ll < n; ll++)
		if (ll != icol) {
			dum = a[ll][icol];
			a[ll][icol] = 0.0;
			for (l = 0; l < n; l++) a[ll][l] -= a[icol][l] * dum;
			for (l = 0; l < m; l++) b[ll][l] -= b[icol][l] * dum;
		}
	}

	// This is the end of the main loop over columns of the reduction. It only remains to unscramble
	// the solution in view of the column interchanges. We do this by interchanging pairs of
	// columns in the reverse order that the permutation was built up.
	for (l = n - 1; l >= 0; l--) {
		if (indxr[l] != indxc[l])
		for (k = 0; k < n; k++)
			SWAP(a[k][indxr[l]], a[k][indxc[l]])
	}

	// success
	//xieyuechao
	free(ipiv);
	free(indxc);
	free(indxr);

	return true;
}


void imageTo3d(Mat disp, Mat& x, Mat& y, Mat& z, Calib calib);
void ObsSegmentation(Mat x, Mat y, Mat z, Mat groundMap, Mat &obstacleMap, ObsParam obsparam);

ObsDetector::ObsDetector(Calib _calib, ObsParam _param)
{
	calib.base = _calib.base;
	calib.cu = _calib.cu;
	calib.cv = _calib.cv;
	calib.f = _calib.f;
	calib.vehicle_height = _calib.vehicle_height;

	obsparam.maxZ = _param.maxZ;
}

void ObsDetector::process(Mat disp, Mat groundMap)
{
	int height = disp.rows;
	int width = disp.cols;
	//cout << "height, width" << height << width << endl;
	Mat xTemp(height, width, CV_32FC1);
	Mat yTemp(height, width, CV_32FC1);
	Mat zTemp(height, width, CV_32FC1);

	obstacleMap.create(height, width, CV_8UC1);

	imageTo3d(disp, xTemp, yTemp, zTemp, calib);

	xTemp.copyTo(x);
	yTemp.copyTo(y);
	zTemp.copyTo(z);

	ObsSegmentation(x, y, z, groundMap, obstacleMap, obsparam);
}

void ObsSegmentation(Mat x, Mat y, Mat z, Mat groundMap, Mat &obstacleMap, ObsParam obsparam)
{
	int height = x.rows;
	int width = x.cols;

	for (int u = 0; u < width; u++)
	{
		for (int v = 0; v < height; v++)
		{
			if (z.at<float>(v, u) <obsparam.maxZ && groundMap.at<uchar>(v, u) == 255 && y.at<float>(v, u) >0.3 && y.at<float>(v, u) < 10) //todo:miny
			{
				obstacleMap.at<uchar>(v, u) = 255;
			}
			else
			{
				obstacleMap.at<uchar>(v, u) = 0;
			}
		}
	}
}

void imageTo3d(Mat disp, Mat& x, Mat& y, Mat& z, Calib calib)
{
	for (int i = 0; i < disp.rows; i++)
	{
		for (int j = 0; j < disp.cols; j++)
		{
			if (disp.at<uchar>(i, j)>0)
			{
				z.at<float>(i, j) = calib.base * calib.f / disp.at<uchar>(i, j);
				x.at<float>(i, j) = (j - calib.cu) * calib.base / disp.at<uchar>(i, j);
				y.at<float>(i, j) = (calib.cv - i) * calib.base / disp.at<uchar>(i, j) + calib.vehicle_height;
			}
			else
			{
				z.at<float>(i, j) = 10000;
				x.at<float>(i, j) = 0;
				y.at<float>(i, j) = 0;
			}
		}
	}
}

 SYSUGroundEstimator::SYSUGroundEstimator(Calib _calib, ObsParam _obsparam, GridParam _gridParam)
{
	calib = _calib;
	obsParam = _obsparam;
	gridParam = _gridParam;
}

SYSUGroundEstimator::~SYSUGroundEstimator()
{

}

bool SYSUGroundEstimator::compute(Mat left, Mat right, Mat& ground, Mat& disp)
{
	ColorEncoder encoder(ColorEncoder::ENCODING_RG, 0, 64, ColorEncoder::ENCODING_LINEAR);

	StereoParam param;
	param.mMaxDisp = 64;
	param.mP1 = 100;
	param.mP2 = 2700;
	param.mWinSize = 4;
	param.mPreFilterCap = 180;

	SYSUStereo sysuStereo(param);
	ObsDetector obsdetector(calib, obsParam);
	sysuStereo.process(left, right);

	int width = sysuStereo.disp.cols;
	int height = sysuStereo.disp.rows;

	GroundEstimatorParam groundParam;
	groundParam.uDispThreshold = 10;
	groundParam.sigmaGaussian = 0.7;
	groundParam.dH = -20;
	groundParam.multiLine = 0;
	groundParam.slopeAngle = SLOPEANGLE;
	GroundEstimator groundEstimator(groundParam);

	groundEstimator.process(sysuStereo.disp.data, width, height, param.mMaxDisp * 2);

	resize(left, left, Size(width, height));

	Mat groundMapMat(height, width, CV_8UC1, groundEstimator.groundMap);
	Mat groundMapMatRansac(height, width, CV_8UC1, groundEstimator.groundMapRansac);

	obsdetector.process(sysuStereo.disp, groundMapMat);

	Mat groundImg = left & groundMapMat;
	Mat obsImg = left & obsdetector.obstacleMap;
	Mat groundMapRansac(height, width, CV_8UC1, groundEstimator.groundMapRansac);
	Mat groundImgRansac = left & groundMapRansac;

	obsdetector.process(sysuStereo.disp, groundMapMatRansac);

	Mat obsImgRansac;
	cvtColor(left, obsImgRansac, CV_GRAY2RGB);

	for (int u = 0; u < left.cols; u++)
	{
		for (int v = 0; v < left.rows; v++)
		{
			if (groundMapMatRansac.at<uchar>(v, u) == 0)
			{
				uchar currgray = left.at<uchar>(v, u);
				uchar currr = (float)currgray*0.5;
				uchar currg = (float)currgray*0.5 + 128.0;
				uchar currb = (float)currgray*0.5;

				obsImgRansac.at<Vec3b>(v, u) = Vec3b(currr, currg, currb);
			}

			if (groundMapMatRansac.at<uchar>(v, u) > 0 && obsdetector.obstacleMap.at<uchar>(v, u) > 0)
			{
				uchar currgray = left.at<uchar>(v, u);
				uchar currr = (float)currgray*0.5 + 128;
				uchar currg = (float)currgray*0.5;
				uchar currb = (float)currgray*0.5;

				obsImgRansac.at<Vec3b>(v, u) = Vec3b(currb, currg, currr);
			}
		}
	}

	//Extract the contours so that
	vector<vector<Point> > contours0;
	vector<Vec4i> hierarchy;
	Mat contourImg(groundMapRansac.rows, groundMapRansac.cols, CV_8UC3, Scalar(0, 0, 0));
	int *groundPoints = (int*)malloc(sizeof(int)* contourImg.cols);
	Mat groundline;
	cvtColor(left, groundline, CV_GRAY2RGB);

	if (groundEstimator.bDetectGround)
	{
		groundMapMatRansac = 255 - groundMapRansac;
		findContours(groundMapRansac, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
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

		Scalar color(255, 255, 255);
		//Scalar color( rand()&255, rand()&255, rand()&255 );
		drawContours(contourImg, contours0, maxidx, color);

		for (int u = 0; u < contourImg.cols; u++)
		{
			groundPoints[u] = 1000000;
			for (int v = 0; v < contourImg.rows; v++)
			{
				if (contourImg.at<Vec3b>(v, u) == Vec3b(255, 255, 255) && v < groundPoints[u])
					groundPoints[u] = v;
			}
		}

		//07-20ע�ͣ����õ�����Ϣ����
		//for (int u = 0; u < contourImg.cols; u++)
		//{
		//	int v = groundPoints[u];
		//	if (v>0 && v < contourImg.rows)
		//	{
		//		for (int subv = v + 1; subv < contourImg.rows; subv++)
		//		{

		//			uchar currgray = left.at<uchar>(subv, u);
		//			uchar currr = (float)currgray*0.5 + 128;
		//			uchar currg = (float)currgray*0.5;
		//			uchar currb = (float)currgray*0.5;

		//			groundline.at<Vec3b>(subv, u) = Vec3b(currr, currg, currb);
		//		}
		//	}
		//}

		//imshow("groundline", groundline);
	}

	Mat obsAndGround(groundline.rows, groundline.cols, CV_8UC1,Scalar(0));

	for (int u = 0; u < left.cols; u++)
	{
		for (int v = 0; v < left.rows; v++)
		{
			//if(contourImg.at<uchar>(v,u) == 0 && obsdetector.obstacleMap.at<uchar>(v,u)>0)
			if (v<groundPoints[u] && obsdetector.obstacleMap.at<uchar>(v, u)>0)
			{
				uchar currgray = left.at<uchar>(v, u);

				uchar currd = sysuStereo.disp.at<uchar>(v, u);

				uchar r, g, b;
				//encoder.valueToColor(float(currd), r, g, b);
				//�û�kitti�е���ɫ��Ⱦ
				disp2Color(currd, r, g, b, sysuStereo.param.mMaxDisp);

				if (currd > 0)
				{
					uchar currr = (float)currgray*0.3 + r*0.7;
					uchar currg = (float)currgray*0.3 + g*0.7;
					uchar currb = (float)currgray*0.3 + b*0.7;

					//uchar currr = (float)currgray*0.3 + float(255)*0.7;
					//uchar currg = (float)currgray*0.3 + float(0x30)*0.7;
					//uchar currb = (float)currgray*0.3 + float(0x30)*0.7;

					groundline.at<Vec3b>(v, u) = Vec3b(currb, currg, currr);
					obsAndGround.at<uchar>(v, u) = 255;
				}
			}
		}
	}

	//grid map generation
	Mat x(left.rows, left.cols, CV_32FC1);
	Mat y(left.rows, left.cols, CV_32FC1);
	Mat z(left.rows, left.cols, CV_32FC1);

	int gridHeight = int(gridParam.maxZ / gridParam.gridSize + 0.5);
	int gridWidth = int(gridParam.maxX * 2 / gridParam.gridSize + 0.5);

	Mat gridMap(gridHeight, gridWidth, CV_16UC1, Scalar(0));
	Mat gridMap8U(gridHeight, gridWidth, CV_8UC1, Scalar(0));
	imageTo3d(sysuStereo.disp * 2, x, y, z, calib);

	for (int u = 0; u < left.cols; u++)
	{
		for (int v = 0; v < left.rows; v++)
		{
			if (obsAndGround.at<uchar>(v, u)>128 && y.at<float>(v, u)<3)
			{
				float currZ = z.at<float>(v, u);
				float currX = x.at<float>(v, u);

				//convert (x,z) to (gridu,gridv)
				if (currX<gridParam.maxX && currX>-gridParam.maxX && currZ<gridParam.maxZ)
				{
					int currGridU = (currX + gridParam.maxX) / gridParam.gridSize;
					int currGridZ = gridHeight - (currZ) / gridParam.gridSize;
					gridMap.at<ushort>(currGridZ, currGridU)++;
				}
			}
		}
	}

	for (int u = 0; u < gridMap.cols; u++)
	{
		for (int v = 0; v < gridMap.rows; v++)
		{
			if (gridMap.at<ushort>(v, u)>10)
				gridMap8U.at<uchar>(v, u) = 255;
		}
	}

	//resize(gridMap8U, gridMap8U, Size(gridMap8U.cols * 4, gridMap8U.rows * 4));
	//imshow("gridmap", gridMap8U);

	gridMap8U.copyTo(gridMap_);
	groundline.copyTo(ground);
	sysuStereo.disp.copyTo(disp);
	Mat obstacle;
	cvtColor(left, obstacle, CV_GRAY2BGR);
	//obstacleExtraction(groundPoints, obsdetector.obstacleMap, obsdetector.x, obsdetector.y, obsdetector.z, obstacle);
	//obstacle.copyTo(ground);
	free(groundPoints);

	return true;
}

void obstacleExtraction(int* groundPoints, Mat obstacleMap, Mat x, Mat y, Mat z, Mat &extractedObs)
{
	Mat BottomLine;
	extractedObs.copyTo(BottomLine);

	int width = obstacleMap.cols;
	int height = obstacleMap.rows;

	//find ground
	for (int u = 2; u < width; u += 5)
	{
		int currgroundPoint = groundPoints[u];
		if (currgroundPoint> 0 && currgroundPoint < height)
		{
			line(BottomLine, Point(u - 2, currgroundPoint), Point(u + 2, currgroundPoint), Scalar(255, 0, 0), 3);
		}
	}

	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours0;
	vector<Vec3i> uBottomUp;

	//contour detection on obstacle map
	findContours(obstacleMap, contours0, RETR_TREE, CHAIN_APPROX_SIMPLE);
	memset(obstacleMap.data, 0, sizeof(uchar)*width*height);
	Scalar contourColor(255, 255, 255);

	for (int i = 0; i < contours0.size(); i++)
	{

		int currerea = contourArea(contours0[i]);
		if (currerea>200)
		{
			drawContours(obstacleMap, contours0, i, contourColor, CV_FILLED);
		}
	}

	//find top obs upon the ground
	int lastDepthValue = -1;

	for (int u = 2; u < width; u += 5)
	{
		int currgroundPoint = groundPoints[u];
		int topObstaclev = -10;

		if (currgroundPoint > 10 && currgroundPoint < height)
		{
			for (int v = currgroundPoint - 10; v >= 0; v--)
			{
				if (obstacleMap.at<uchar>(v, u)>0)
					topObstaclev = v;
			}
		}

		if (topObstaclev >= 0)
		{
			uBottomUp.push_back(Vec3i(u, currgroundPoint, topObstaclev));
		}
	}

	vector<int> segments;
	for (int i = 1; i < uBottomUp.size() - 1; i++)
	{
		int curru = uBottomUp[i][0];    int prevu = uBottomUp[i - 1][0];   int nextu = uBottomUp[i + 1][0];
		int currBottom = uBottomUp[i][1];    int currUp = uBottomUp[i][2];
		int prevBottom = uBottomUp[i - 1][1];  int prevUp = uBottomUp[i - 1][2];
		int nextBottom = uBottomUp[i + 1][1];  int nextUp = uBottomUp[i + 1][2];
		if (curru - prevu>10)
		{
			segments.push_back(curru);

		}
		else
		{
			float prevHeight = abs(prevBottom - prevUp);
			float currHeight = abs(currBottom - currUp);

			float leftGradient = float(abs(currHeight - prevHeight)) / float(abs(curru - prevu));
			//float rightGradient = float(abs(nextUp - currUp))/float(abs(curru-nextu));
			if (leftGradient > 5/*||rightGradient>5*/)
			{
				segments.push_back(curru);
			}
		}
	}

	drawObstacle(uBottomUp, segments, extractedObs);

}

void drawObstacle(vector<Vec3i> uBottomUp, vector<int> segments, Mat &obstacleImage)
{
	if (uBottomUp.size() == 0)
		return;

	int width = obstacleImage.cols;
	int height = obstacleImage.rows;

	int numOfObstacles = 0;
	int prevSegment = 0;
	int uBottomUpInd = 0;
	Mat obstacleBin(height, width, CV_8UC3, Scalar(0, 0, 0));
	int curru, currBottom, currUp;
	int currSegment;
	for (int i = 0; i < segments.size(); i++)
	{
		currSegment = segments[i];

		curru = uBottomUp[uBottomUpInd][0];
		currBottom = uBottomUp[uBottomUpInd][1];
		currUp = uBottomUp[uBottomUpInd][2];

		while ((curru < currSegment) && (curru >= prevSegment))
		{

			cout << curru << " " << currBottom << " " << currUp << endl;

			rectangle(obstacleBin, Point(curru - 2, currBottom), Point(curru + 2, currUp), obsColor[numOfObstacles % 8], CV_FILLED);

			uBottomUpInd++;
			curru = uBottomUp[uBottomUpInd][0];
			currBottom = uBottomUp[uBottomUpInd][1];
			currUp = uBottomUp[uBottomUpInd][2];
		}

		if (currSegment - prevSegment > 5)
			numOfObstacles++;

		if (uBottomUpInd < uBottomUp.size() - 1 && uBottomUp[uBottomUpInd + 1][0] - curru>5)
			numOfObstacles++;

		prevSegment = currSegment;

	}

	if (uBottomUpInd < uBottomUp.size())
	{
		while ((curru < width) && (curru >= prevSegment))
		{
			rectangle(obstacleBin, Point(curru - 2, currBottom), Point(curru + 2, currUp), obsColor[numOfObstacles % 8], CV_FILLED);

			uBottomUpInd++;
			if (uBottomUpInd >= uBottomUp.size()) break;
			curru = uBottomUp[uBottomUpInd][0];
			currBottom = uBottomUp[uBottomUpInd][1];
			currUp = uBottomUp[uBottomUpInd][2];
		}
	}
	addWeighted(obstacleBin, 0.5, obstacleImage, 0.5, 0, obstacleImage);
}
