#include <stdio.h>

/*
	a  (in) - a double array of size 'size'
	b  (in) - a double array of size 'size'
	c (out) - an allocated double array of size 'size'
*/
__device__ void diff_double(double *a, double *b, double *c, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		c[i] = a[i] - b[i];
	}
}

/*
	a  (in) - a double array of size 'size'
	b  (in) - a double array of size 'size'
	c (out) - an allocated double array of size 'size'
*/
__device__ void add_double(double *a, double *b, double *c, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

/*
	a     (in) - a double array of size 'size'
	scale (in) - a double
	c    (out) - an allocated double array of size 'size'
*/
__device__ void scale_double(double *v, double scale, double *c, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		c[i] = v[i]*scale;
	}
}

__device__ double dot(double *a, double *b, int size)
{
	double result = 0;
	for (int i = 0; i < size; i++)
	{
		result += a[i]*b[i];
	}
	return result;
}

__device__ double norm_l2(double *v, int size)
{
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += v[i]*v[i];
	}
	return sqrt(sum);
}

__device__ void matrix_product_3xn_nx1(double *m, double *v, double *out, int size)
{
	for (int j = 0; j < size; j++)
	{
		out[j] = m[j*3+0]*v[0] + m[j*3+1]*v[1] + m[j*3+2]*v[2];
	}
}


__device__ void distort_lin_to_nonlin(double point2D[2], calib_t *calib, double out[2])
{
	double xh[3];
	double point2Dh[3] = {point2D[0], point2D[1], 1};
	matrix_product_3xn_nx1(calib->KKinv, point2Dh, xh, 3);
	double x[2] = {xh[0]/xh[2], xh[1]/xh[2]};

	double cx = x[0];
	double cy = x[1];
	double r2 = cx*cx + cy*cy;
	double rad1 = calib->kc[0];
	double rad2 = calib->kc[1];
	double tan1 = calib->kc[2];
	double tan2 = calib->kc[3];
	double rad3 = calib->kc[4];
	double delta = rad1*r2 + rad2*r2*r2 + rad3*r2*r2*r2;

	// diff starts here
	double dx[2] = {
		2*tan1*cx*cy + tan2*(r2+2*cx*cx),
		tan1*(r2+2*cy*cy) + 2*tan2*cx*cy
	};
	double tmp[2];
	scale_double(x, delta+1, tmp, 2);
	add_double(tmp, dx, xh, 2);
	xh[2] = 1;
	// diff ends here

	matrix_product_3xn_nx1(calib->KK, xh, point2Dh, 3);
	out[0] = point2Dh[0]/point2Dh[2];
	out[1] = point2Dh[1]/point2Dh[2];
}

__device__ void distort_nonlin_to_lin(double point2D[2], calib_t *calib, double out[2])
{
	double xh[3];
	double point2Dh[3] = {point2D[0], point2D[1], 1};
	matrix_product_3xn_nx1(calib->KKinv, point2Dh, xh, 3);
	double x[2] = {xh[0]/xh[2], xh[1]/xh[2]};

	double cx = x[0];
	double cy = x[1];
	double r2 = cx*cx + cy*cy;
	double rad1 = calib->kc[0];
	double rad2 = calib->kc[1];
	double tan1 = calib->kc[2];
	double tan2 = calib->kc[3];
	double rad3 = calib->kc[4];
	double delta = rad1*r2 + rad2*r2*r2 + rad3*r2*r2*r2;

	// diff starts here
	double Q = 1 + (4*rad1*r2 + 6*rad2*r2*r2 + 8*tan1*cy + 8*tan2*cx);
	double dx[2] = {
		(cx*delta + 2*tan1*cx*cy + tan2*(r2+2*cx*cx))/Q,
		(cy*delta + tan1*(r2+2*cy*cy) + 2*tan2*cx*cy)/Q
	};

	diff_double(x, dx, xh, 2);
	xh[2] = 1;
	// diff ends here

	matrix_product_3xn_nx1(calib->KK, xh, point2Dh, 3);
	out[0] = point2Dh[0]/point2Dh[2];
	out[1] = point2Dh[1]/point2Dh[2];
}

__device__ void project_3D_to_2D(double point3D[3], calib_t *calib, double point2D[2])
{
	double point3Dh[4] = {point3D[0], point3D[1], point3D[2], 1};
	double point2Dh[3];
	matrix_product_3xn_nx1(calib->P, point3Dh, point2Dh, 4);
	point2D[0] = point2Dh[0]/point2Dh[2];
	point2D[1] = point2Dh[1]/point2Dh[2];
	if (calib->kc[0] != 0 || calib->kc[1] != 0 || calib->kc[2] != 0  || calib->kc[3] != 0 || calib->kc[4] != 0 )
	{
		double tmp[2] = {point2D[0], point2D[1]};
		distort_lin_to_nonlin(tmp, calib, point2D);
	}
}

__device__ void project_2D_to_3D(double point2D[2], double z, calib_t *calib, double point3D[3])
{
	double point2Dh[3] = {point2D[0], point2D[1], 1};
	if (calib->kc[0] != 0 || calib->kc[1] != 0 || calib->kc[2] != 0  || calib->kc[3] != 0 || calib->kc[4] != 0 )
	{
		distort_nonlin_to_lin(point2D, calib, point2Dh);
	}

	double point3Dh[4];
	matrix_product_3xn_nx1(calib->Pinv, point2Dh, point3Dh, 4);
	point3D[0] = point3Dh[0]/point3Dh[3];
	point3D[1] = point3Dh[1]/point3Dh[3];
	point3D[2] = point3Dh[2]/point3Dh[3];

	double p0[3] = {0,0,z};
	double pn[3] = {0,0,-1};
	double r1[3];
	diff_double(point3D, calib->poscam, r1, 3);
	double div = sqrt(dot(r1, r1, 3));
	r1[0] = r1[0]/div;
	r1[1] = r1[1]/div;
	r1[2] = r1[2]/div;

	double diff[3];
	diff_double(p0, calib->poscam, diff, 3);
	double d = dot(diff, pn, 3) / dot(r1, pn, 3);
	r1[0] = r1[0]*d;
	r1[1] = r1[1]*d;
	r1[2] = r1[2]*d;
	add_double(calib->poscam, r1, point3D, 3);
}

__device__ int line_plane_intersection(double plane_normal[3], double plane_point[3], double line_direction[3], double line_point[3], double intersection[3])
{
	double epsilon = 0.000001;
	double prod = dot(plane_normal, line_direction, 3);
	if (abs(prod) < epsilon)
	{
		return -1;
	}

	double w[3];
	diff_double(line_point, plane_point, w, 3);
	double factor = -dot(plane_normal, w, 3)/prod;
	double tmp1[3];
	add_double(w, plane_point, tmp1, 3);
	double tmp2[3];
	scale_double(line_direction, factor, tmp2, 3);
	add_double(tmp1, tmp2, intersection, 3);
	return 0;
}

/*
	point2D (in) - position in the heatmap (in pixels)
	point3D (in) - ball position
	calib   (in) - the calibration
	returns:
		the distance
*/
__device__ double compute_length3D(double point2D[2], double point3D[3], calib_t *calib)
{
	double plane_normal[3];
	diff_double(calib->poscam, point3D, plane_normal, 3);
	double ray2floor[3];
	project_2D_to_3D(point2D, 0.0, calib, ray2floor);
	double line_direction[3];
	diff_double(calib->poscam, ray2floor, line_direction, 3);
	double intersection[3];
	line_plane_intersection(plane_normal, point3D, line_direction, calib->poscam, intersection);
	double radius[3];
	diff_double(point3D, intersection, radius, 3);
	return norm_l2(radius, 3);
}

__global__ void BallDistance(double *map, calib_t *calib, double *ball3D)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (y < calib->img_height && x < calib->img_width)
	{
		int offset = x + calib->img_width * y;
		double point2D[2] = {x,y};
		map[offset] = compute_length3D(point2D, ball3D, calib);
	}
}