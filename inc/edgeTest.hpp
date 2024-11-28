#pragma once

#ifdef _WIN32
	#ifdef EDGETEST_EXPORTS
		#define EDGETEST_EXPORT __declspec(dllexport)
	#else
		#define EDGETEST_EXPORT __declspec(dllimport)
	#endif
#else
    #define EDGETEST_EXPORT
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <ctime>

#define CLOCKS_PER_MS (CLOCKS_PER_SEC / 1000)

EDGETEST_EXPORT void * xmalloc(size_t size);
EDGETEST_EXPORT int greater(double a, double b);
EDGETEST_EXPORT double dist(double x1, double y1, double x2, double y2);
EDGETEST_EXPORT void gaussian_kernel(double * kernel, int n, double sigma, double mean);
EDGETEST_EXPORT void gaussian_filter(uchar* image, uchar* out, int iHeight, int iWidth, double sigma);
EDGETEST_EXPORT double chain(int from, int to, double * Ex, double * Ey, double * Gx, double * Gy, int X, int Y);
EDGETEST_EXPORT void compute_gradient(double * Gx, double * Gy, double * modG, uchar * image, int X, int Y);
EDGETEST_EXPORT void compute_edge_points(double * Ex, double * Ey, double * modG,
	double * Gx, double * Gy, int X, int Y);
EDGETEST_EXPORT void chain_edge_points(int * next, int * prev, double * Ex,
	double * Ey, double * Gx, double * Gy, int X, int Y);
EDGETEST_EXPORT void thresholds_with_hysteresis(int * next, int * prev,
	double * modG, int X, int Y, double th_h, double th_l);
EDGETEST_EXPORT void list_chained_edge_points(double ** x, double ** y, int * N,int ** curve_limits,
	int * M,int * next, int * prev,	double * Ex, double * Ey, int X, int Y);
EDGETEST_EXPORT void devernay(double ** x, double ** y, int * N, int ** curve_limits, int * M,
	uchar * image, uchar * gauss, int X, int Y, double sigma, double th_h, double th_l);

