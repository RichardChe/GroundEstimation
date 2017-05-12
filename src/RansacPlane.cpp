#include "RANSACPlane.h"
#include "MatrixOp.h"
#include <time.h>

RANSACPlane::RANSACPlane(Parameter* param)
	:mParam(param)
{

}

RANSACPlane::RANSACPlane()
{

}

RANSACPlane::~RANSACPlane()
{

}

bool RANSACPlane::compute(Mat* disparity, Mat* groundMap) //如果直线不存在，groundMap就为空
{
	mParam->mWidth = disparity->cols;
	mParam->mHeight = disparity->rows;

	const int width = mParam->mWidth;
	const int height = mParam->mHeight;
	mDisparity = disparity;
	mGroundMap = groundMap;
	// random seed
	srand(time(NULL));

	mDisparity->convertTo(mDispFloat, CV_32FC1);
	// get list with disparities
	float *D = (float*)mDispFloat.data;
	
	if (mParam->mWidth == 0 || mParam->mHeight == 0)
	{
		std::cout << "Invalid disparity data in RoadEstimator::compute..." << std::endl;
		return false;
	}

	if (!groundMap)
		mRoi = RansacROI(200, width - 200, int(float(height) / 2.0 + 0.5), height);
	else
		mRoi = RansacROI(-1, -1, -1, -1);
	
	// random seed
	srand(time(NULL));

	// get list with disparities
	std::vector<RANSACPlane::Disp> d_list;

	d_list = sparseDisparityGrid(D, width, height, 5);
	// loop variables
	std::vector<int32_t> curr_inlier;
	std::vector<int32_t> best_inlier;

	for (int32_t i = 0; i < mParam->mNumSamples; i++) {

		// draw random samples and compute plane
		drawRandomPlaneSample(d_list, mPlane);

		// find inlier
		curr_inlier.clear();
		for (int32_t i = 0; i < d_list.size(); i++)
			if (fabs(mPlane[0] * d_list[i].u + mPlane[1] * d_list[i].v + mPlane[2] - d_list[i].d) < mParam->mDThreshold)
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

	return true;
}

void RANSACPlane::getRoadPlane(float *plane)
{
	for (int i = 0; i < 3; i++)
		plane[i] = mPlane[i];
}

std::vector<RANSACPlane::Disp> RANSACPlane::sparseDisparityGrid(float* D, const int width, const int height, int32_t step_size)
{

	// init list
	std::vector<RANSACPlane::Disp> d_list;
	uint8_t* groundMapData = mGroundMap->data;
	// loop through disparity image
	if (mGroundMap)
	{
		for (int32_t u = 0; u < width; u += step_size) {
			for (int32_t v = 0; v < height; v += step_size) {
				float d = *(D + getAddressOffsetImage(u, v, width));
				if (d >= 1 && *(groundMapData + getAddressOffsetImage(u, v, width)) == 0)
					d_list.push_back(RANSACPlane::Disp(u, v, d));
			}
		}
	}
	else
	{
		int x0 = mRoi.startX;
		int x1 = mRoi.endX;
		int y0 = mRoi.startY;
		int y1 = mRoi.endY;

		// loop through disparity image
		for (int32_t u = max(x0, 0); u <= min(x1, width - 1); u += step_size) {
			for (int32_t v = max(y0, 0); v <= min(y1, height - 1); v += step_size) {
				float d = *(D + getAddressOffsetImage(u, v, width));
				if (d >= 1)
					d_list.push_back(RANSACPlane::Disp(u, v, d));
			}
		}
	}

	// return list
	return d_list;
}

void RANSACPlane::drawRandomPlaneSample(std::vector<RANSACPlane::Disp> &d_list, float *plane)
{

	int32_t num_data = d_list.size();
	std::vector<int32_t> ind;

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

void RANSACPlane::leastSquarePlane(std::vector<RANSACPlane::Disp> &d_list, std::vector<int32_t> &ind, float *plane) {

	int32_t n = 3; int32_t m = 1;
	float** A = MatrixOP:: allocateMatrix(n, n);
	float** b = MatrixOP:: allocateMatrix(n, m);

	// find parameters
	for (std::vector<int32_t>::iterator it = ind.begin(); it != ind.end(); it++)
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

	MatrixOP:: freeMatrix(A);
	MatrixOP:: freeMatrix(b);
}

bool RANSACPlane::gaussJordanElimination(float **a, const int n, float **b, int m, float eps) {

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