/*
*  util.h
*  OpenCL Tutorial
*
*  Created by David Black-Schaffer on 07/07/2011.
*  Copyright 2011 Uppsala University. All rights reserved.
*
*/
#ifndef _util_h_
#define _util_h_

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <windows.h>

#define DEFAULT_DEVICE_TYPE CL_DEVICE_TYPE_GPU

// Range limit, arbitrary. (E.g., data is created to fit this range.)
#define LIMIT 100
// Increase to start with a larger initial range
#define INITIAL_RANGE_MULTIPLE 4
#define BIG_RANGE LIMIT*INITIAL_RANGE_MULTIPLE*100
// Data size. Use about 1024*7 for 196MB x 2, 1024*5 for 100MBx2
#define SIZE 1024*4
#define SIZE_BYTES sizeof(float)*SIZE*SIZE

// Define to 1 to not include first and last measurements in performance statistics
#define PERF_IGNORE_FIRST_LAST	1
#define MAX_PERF_SAMPLES 100

// Timer structure
#ifndef perf_defined
#define perf_defined 1
typedef struct perf_struct {
	int current_sample;
	double samples[MAX_PERF_SAMPLES];
	struct timeval tv;
} perf;
#endif

// Compute
float find_range(float *data, int size);

// Timer functions
void init_perf(perf *t);
void start_perf_measurement(perf *t);
void stop_perf_measurement(perf *t);
int get_number_of_samples(perf *t);
void print_perf_measurement(perf *t);
double get_total_perf_time(perf *t);
double get_average_perf_time(perf *t);


// Utility functions
void swap(float **in, float **out);			// Swap two float array pointers
void create_data(float **in, float **out);	// Create the data matricies
void print_data(float *data);				// Print out the data for debugging

int gettimeofday(struct timeval * tp, struct timezone * tzp);

#endif