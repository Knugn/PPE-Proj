#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <string>
#include <tiff.h>
#include <tiffio.h>
#include "custom_types.h"
#include "xml_aux.h"
#include "dct8x8_block.h"
#include <vector>
//#include "gettimeofday.h"
#include "config.h"
#include <chrono>
#include "immintrin.h"
#include <omp.h>
#include "opencl_utils.h"
#include "util.h"
//#include <windows.h>

using namespace std;

#define NUM_THREADS 4

#ifdef SIMD
	#define yIncr 4
#else
	#define yIncr 1
#endif

#ifdef OpenCL
#define NUM_LCL_SAD_VALUES_X	(WINDOW_SIZE*2)
#define NUM_LCL_SAD_VALUES_Y	(WINDOW_SIZE*2)
#define NUM_LCL_SAD_VALUES		(NUM_LCL_SAD_VALUES_X*NUM_LCL_SAD_VALUES_Y)

int opencl_cores = 1024;
cl_device_id opencl_device;
cl_context opencl_context;
cl_command_queue opencl_queue;
cl_program opencl_program;
cl_kernel convert_kernel, blur_kernel, mvs_kernel;
cl_mem in_R, in_G, in_B;
cl_mem Y_buffer, Cb_buffer, Cr_buffer, Cb_blurred_buffer, Cr_blurred_buffer;
cl_mem Y_buffer_prev, Cb_blurred_buffer_prev, Cr_blurred_buffer_prev;
cl_mem SAD_buffer;

int num_match_blocks_x, num_match_blocks_y, nSADsTotal, nSADBytes;

perf program_perf, create_perf, write_perf, read_perf, finish_perf, cleanup_perf;
perf total_perf, convert_perf;
void init_all_perfs();
void print_perfs();
#endif

void loadImage(int number, string path, Image** photo) {
    string filename;
    TIFFRGBAImage img;
    char emsg[1024];
    
	filename = path + to_string(number) + ".tiff";
    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    if(tif==NULL) fprintf(stderr,"Failed opening image: %s\n", filename);
    if (!(TIFFRGBAImageBegin(&img, tif, 0, emsg))) TIFFError(filename.c_str(), emsg);
     
    uint32 w, h;
    size_t npixels;
    uint32* raster;
     
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    npixels = w * h;
    raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
     
    TIFFReadRGBAImage(tif, w, h, raster, 0);
     
    if(*photo==NULL) 
		*photo = new Image((int)w, (int)h, FULLSIZE);

	//Matlab and LibTIFF store the image diferently.
	//Necessary to mirror the image horizonatly to be consistent

	for (int i = 0; i<(int)h; i++) {
		for (int j = 0; j<(int)w; j++) {
            // The inversion is ON PURPOSE
            (*photo)->rc->data[(h-1-i)*w+j] = (float)TIFFGetR(raster[i*w+j]);
            (*photo)->gc->data[(h-1-i)*w+j] = (float)TIFFGetG(raster[i*w+j]);
            (*photo)->bc->data[(h-1-i)*w+j] = (float)TIFFGetB(raster[i*w+j]);
        }
    }
     
    _TIFFfree(raster);
    TIFFRGBAImageEnd(&img);
    TIFFClose(tif);
     
}



void convertRGBtoYCbCr(Image* in, Image* out){
    int width = in->width;
    int height = in->height;
	
#ifdef OMP
#pragma omp parallel for 
#endif
	for (int x = 0; x<height; x++) {
		for (int y = 0; y<width; y+=yIncr) {
#ifndef SIMD
			float R = in->rc->data[x*width + y];
			float G = in->gc->data[x*width + y];
			float B = in->bc->data[x*width + y];
			float Y = 0 + ((float)0.299*R) + ((float)0.587*G) + ((float)0.113*B);
			float Cb = 128 - ((float)0.168736*R) - ((float)0.331264*G) + ((float)0.5*B);
			float Cr = 128 + ((float)0.5*R) - ((float)0.418688*G) - ((float)0.081312*B);
			out->rc->data[x*width + y] = Y;
			out->gc->data[x*width + y] = Cb;
			out->bc->data[x*width + y] = Cr;
#else				
			__m128 rVec, gVec, bVec;

			rVec = _mm_load_ps(&(in->rc->data[x*width + y]));
			gVec = _mm_load_ps(&(in->gc->data[x*width + y]));
			bVec = _mm_load_ps(&(in->bc->data[x*width + y]));
			
			__m128 cons[] = {
				_mm_set_ps1((float)0.299),
				_mm_set_ps1((float)0.587),
				_mm_set_ps1((float)0.113) };
			
			__m128 yVec0 = _mm_mul_ps(cons[0],rVec);
			__m128 yVec1 = _mm_add_ps(yVec0, _mm_mul_ps(cons[1], gVec));
			__m128 yVec = _mm_add_ps(yVec1, _mm_mul_ps(cons[2], bVec));

			cons[0] = _mm_set_ps1((float)0.168736);
			cons[1] = _mm_set_ps1((float)0.331264);
			cons[2] = _mm_set_ps1((float)0.5);

			__m128 cbVec = _mm_set_ps1(128);
			__m128 cbVec1 = _mm_sub_ps(cbVec, _mm_mul_ps(cons[0], rVec));
			__m128 cbVec2 = _mm_sub_ps(cbVec1, _mm_mul_ps(cons[1], gVec));
			cbVec = _mm_add_ps(cbVec2, _mm_mul_ps(cons[2], bVec));
			
			cons[0] = _mm_set_ps1((float)0.5);
			cons[1] = _mm_set_ps1((float)0.418688);
			cons[2] = _mm_set_ps1((float)0.081312);

			__m128 crVec = _mm_set_ps1(128);
			__m128 crVec1 = _mm_add_ps(crVec, _mm_mul_ps(cons[0], rVec));
			__m128 crVec2 = _mm_sub_ps(crVec1, _mm_mul_ps(cons[1], gVec));
			crVec = _mm_sub_ps(crVec2, _mm_mul_ps(cons[2], bVec));

			_mm_store_ps(&(out->rc->data[x*width + y]), yVec);
			_mm_store_ps(&(out->gc->data[x*width + y]), cbVec); 
			_mm_store_ps(&(out->bc->data[x*width + y]), crVec);
#endif
		
        }
    }
}

#ifdef OpenCL
void setup_cl(int width, int height) {
	cl_int error;
	std::string sourcePath = "..//kernel.cl";
	std::ifstream sourceFile(sourcePath);
	if (!sourceFile.is_open())
		std::cerr << "Failed to open kernel source file \"" << sourcePath << "\"" << std::endl;
	std::string program_text((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());
	char * writable = new char[program_text.size() + 1];
	std::copy(program_text.begin(), program_text.end(), writable);
	writable[program_text.size()] = '\0';

	start_perf_measurement(&program_perf);
	// Create the program
	opencl_program = clCreateProgramWithSource(opencl_context, 1,(const char **)&writable, NULL, &error);
	checkError(error, "clCreateProgramWithSource");

	// Compile the program and check for errors
	error = clBuildProgram(opencl_program, 1, &opencl_device, NULL, NULL, NULL);
	// Get the build errors if there were any
	if (error != CL_SUCCESS) {
		printf("clCreateProgramWithSource failed (%d). Getting program build log.\n", error);
		cl_int error2;
		char build_log[10000];
		error2 = clGetProgramBuildInfo(opencl_program, opencl_device, CL_PROGRAM_BUILD_LOG, 10000, build_log, NULL);
		checkError(error2, "clGetProgramBuildInfo");
		printf("Build Failed. Log:\n%s\n", build_log);
	}
	checkError(error, "clBuildProgram");

	// Create the computation kernel
	convert_kernel = clCreateKernel(opencl_program, "convert_RGB_to_YCbCr", &error);
	checkError(error, "clCreateKernel");

	blur_kernel = clCreateKernel(opencl_program, "blur", &error);
	checkError(error, "clCreateKernel");

	mvs_kernel = clCreateKernel(opencl_program, "motion_vector_search", &error);
	checkError(error, "clCreateKernel");

	cl_ulong kernel_size;
	error = clGetKernelWorkGroupInfo(mvs_kernel, opencl_device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &kernel_size, NULL);
	checkError(error, "clGetKernelWorkGroupInfo");
	printf("MVS kernel local mem size: \"%d\".\n", kernel_size);

	num_match_blocks_x = (width - (2 * WINDOW_SIZE)) / BLOCK_SIZE;
	num_match_blocks_y = (height - (2 * WINDOW_SIZE)) / BLOCK_SIZE;
	nSADsTotal = num_match_blocks_x*num_match_blocks_y*NUM_LCL_SAD_VALUES;
	nSADBytes = nSADsTotal*sizeof(float);
	
	stop_perf_measurement(&program_perf);

	int image_byte_size = width*height*sizeof(float);
	start_perf_measurement(&create_perf);
	// Create the data objects
	in_R = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, image_byte_size, NULL, &error);
	checkError(error, "clCreateBuffer");
	in_G = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, image_byte_size, NULL, &error);
	checkError(error, "clCreateBuffer");
	in_B = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, image_byte_size, NULL, &error);
	checkError(error, "clCreateBuffer");

	Y_buffer = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, image_byte_size, NULL, &error);
	checkError(error, "clCreateBuffer");
	Cb_buffer = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, image_byte_size, NULL, &error);
	checkError(error, "clCreateBuffer");
	Cr_buffer = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, image_byte_size, NULL, &error);
	checkError(error, "clCreateBuffer");

	Cb_blurred_buffer = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, image_byte_size, NULL, &error);
	checkError(error, "clCreateBuffer");
	Cr_blurred_buffer = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, image_byte_size, NULL, &error);
	checkError(error, "clCreateBuffer");
	
	Y_buffer_prev = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, image_byte_size, NULL, &error);
	checkError(error, "clCreateBuffer");
	Cb_blurred_buffer_prev = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, image_byte_size, NULL, &error);
	checkError(error, "clCreateBuffer");
	Cr_blurred_buffer_prev = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, image_byte_size, NULL, &error);
	checkError(error, "clCreateBuffer");

	SAD_buffer = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, nSADBytes, NULL, &error);
	checkError(error, "clCreateBuffer");

	stop_perf_measurement(&create_perf);

	delete writable;
	//free(program_text);

}

void copy_data_to_device(cl_mem buffer_to_write_to, float *input_buffer, int nbytes) {
	cl_int error;
	error = clEnqueueWriteBuffer(opencl_queue, buffer_to_write_to, CL_FALSE, 0, nbytes, input_buffer, 0, NULL, NULL);
	checkError(error, "clEnqueueWriteBuffer");
}

void read_back_data(cl_mem buffer_to_read_from, float *result_buffer, int nbytes) {
	cl_int error;
	// Enqueue a read to get the data back
	error = clEnqueueReadBuffer(opencl_queue, buffer_to_read_from, CL_FALSE, 0, nbytes, result_buffer, 0, NULL, NULL);
	checkError(error, "clEnqueueReadBuffer");
}


void convertRGBtoYCbCr_cl(Image* in, Image* out) 
{
	cl_int error;

	start_perf_measurement(&write_perf);
	//copy_data_to_device(in, out);
	int n_input_bytes = in->width * in->height * sizeof(float);
	copy_data_to_device(in_R, in->rc->data, n_input_bytes);
	copy_data_to_device(in_G, in->gc->data, n_input_bytes);
	copy_data_to_device(in_B, in->bc->data, n_input_bytes);
	error = clFinish(opencl_queue);
	checkError(error, "clFinish");
	stop_perf_measurement(&write_perf);

	start_perf_measurement(&convert_perf);
	// Set the kernel arguments
	error = clSetKernelArg(convert_kernel, 0, sizeof(in_R), &in_R);
	checkError(error, "clSetKernelArg in");
	error = clSetKernelArg(convert_kernel, 1, sizeof(in_G), &in_G);
	checkError(error, "clSetKernelArg in");
	error = clSetKernelArg(convert_kernel, 2, sizeof(in_B), &in_B);
	checkError(error, "clSetKernelArg in");
	error = clSetKernelArg(convert_kernel, 3, sizeof(Y_buffer), &Y_buffer);
	checkError(error, "clSetKernelArg in");
	error = clSetKernelArg(convert_kernel, 4, sizeof(Cb_buffer), &Cb_buffer);
	checkError(error, "clSetKernelArg in");
	error = clSetKernelArg(convert_kernel, 5, sizeof(Cr_buffer), &Cr_buffer);
	checkError(error, "clSetKernelArg in");

	// Enqueue the kernel
	size_t global_dimensions[1] = { in->width * in->height };
	size_t local_dimensions[1] = { opencl_cores };
	error = clEnqueueNDRangeKernel(opencl_queue, convert_kernel, 1, NULL, global_dimensions, local_dimensions, 0, NULL, NULL);
	checkError(error, "clEnqueueNDRangeKernel");
	error = clFinish(opencl_queue);
	checkError(error, "running kernel");
	stop_perf_measurement(&convert_perf);

	start_perf_measurement(&read_perf);
	int n_output_bytes = out->width * out->height * sizeof(float);
	read_back_data(Y_buffer, out->rc->data, n_output_bytes);
	read_back_data(Cb_buffer, out->gc->data, n_output_bytes);
	read_back_data(Cr_buffer, out->bc->data, n_output_bytes);
	error = clFinish(opencl_queue);
	checkError(error, "clFinish");
	stop_perf_measurement(&read_perf);

	
}
#endif

Channel* lowPass(Channel* in, Channel* out){
    // Applies a simple 3-tap low-pass filter in the X- and Y- dimensions.
    // E.g., blur
    // weights for neighboring pixels
    const float a=0.25;
    const float b=0.5;
    const float c=0.25;

#ifdef SIMD
	__m128 aVec = _mm_set_ps1(a);
	__m128 bVec = _mm_set_ps1(b);
	__m128 cVec = _mm_set_ps1(c);
#endif


	int width = in->width; 
	int height = in->height;
     
    //out = in; TODO Is this necessary?
	for(int i=0; i<width*height; i++) out->data[i] =in->data[i];

	// In X
#ifdef OMP
#pragma omp parallel for 
#endif
	for (int x = 1; x<(height - 1); x++) {
		for (int y = 1; y<(width - 1); y+=yIncr) {
#ifndef SIMD
            out->data[x*width+y] = a*in->data[(x-1)*width+y]+b*in->data[x*width+y]+c*in->data[(x+1)*width+y];
#else
			__m128 row1 = _mm_load_ps(&(in->data[(x - 1)*width + y]));
			__m128 row2 = _mm_load_ps(&(in->data[(x)*width + y]));
			__m128 row3 = _mm_load_ps(&(in->data[(x + 1)*width + y]));

			__m128 resVec0 = _mm_mul_ps(aVec, row1);
			__m128 resVec1 = _mm_add_ps(resVec0, _mm_mul_ps(bVec, row2));
			__m128 resVec = _mm_add_ps(resVec1, _mm_mul_ps(cVec, row3));

			_mm_store_ps(&(out->data[x*width + y]), resVec);
#endif
        }
    }

	// In Y
#ifdef OMP
#pragma omp parallel for 
#endif
	for (int x = 1; x<(height - 1); x++) {
		for (int y = 1; y<(width - 1); y+=yIncr) {
#ifndef SIMD
            out->data[x*width+y] = a*out->data[x*width+(y-1)]+b*out->data[x*width+y]+c*out->data[x*width+(y+1)];
#else
			__m128 left, right;
			__m128 centre = _mm_load_ps(&(in->data[x*width + y]));
			if (y == 0)
				left = _mm_set_ps(0, out->data[x*width + y], out->data[x*width + y + 1], out->data[x*width + y + 2]);
			else
				left = _mm_set_ps(out->data[x*width + y - 1], out->data[x*width + y], out->data[x*width + y + 1], out->data[x*width + y + 2]);
			if (y + 4 == width)
				right = _mm_set_ps(out->data[x*width + y + 1], out->data[x*width + y + 2], out->data[x*width + y + 3], 0);
			else
				right = _mm_set_ps(out->data[x*width + y + 1], out->data[x*width + y + 2], out->data[x*width + y + 3], out->data[x*width + y + 4]);
				
			__m128 resVec = _mm_add_ps(
				_mm_mul_ps(bVec, centre),
				_mm_add_ps(_mm_mul_ps(aVec, left), _mm_mul_ps(cVec, right)));

			_mm_store_ps(&(out->data[x*width + y]), resVec);
#endif
        }
    }
     
    return out;
}

#ifdef OpenCL
Channel* lowPass_cl(cl_mem in_buffer, cl_mem out_buffer, Channel* out)
{
	cl_int error;
	error = clSetKernelArg(blur_kernel, 0, sizeof(in_buffer), &in_buffer);
	checkError(error, "clSetKernelArg in");
	error = clSetKernelArg(blur_kernel, 1, sizeof(out_buffer), &out_buffer);
	checkError(error, "clSetKernelArg in");

	// Enqueue the kernel
	size_t global_dimensions[2] = { out->width, out->height };
	size_t local_dimensions[2] = { 32, 32 };
	error = clEnqueueNDRangeKernel(opencl_queue, blur_kernel, 2, NULL, global_dimensions, local_dimensions, 0, NULL, NULL);
	checkError(error, "clEnqueueNDRangeKernel");
	error = clFinish(opencl_queue);
	checkError(error, "running kernel");
	//stop_perf_measurement(&convert_perf);

	//start_perf_measurement(&read_perf);
	int n_output_bytes = out->width * out->height * sizeof(float);
	read_back_data(out_buffer, out->data, n_output_bytes);
	error = clFinish(opencl_queue);
	checkError(error, "clFinish");
	//stop_perf_measurement(&read_perf);
}
#endif

std::vector<mVector>* motionVectorSearch(Frame* source, Frame* match, int width, int height) {
    std::vector<mVector> *motion_vectors = new std::vector<mVector>(); // empty list of ints

    float Y_weight = 0.5;
    float Cr_weight = 0.25;
    float Cb_weight = 0.25;
     
    //Window size is how much on each side of the block we search
    int window_size = 16;
    int block_size = 16;
     
    //How far from the edge we can go since we don't special case the edges
    int inset = (int) max((float)window_size, (float)block_size);
    int iter=0;

	for (int mx = inset; mx<width - (inset + window_size) + 1; mx += block_size) {
		for (int my = inset; my<height - (inset + window_size) + 1; my += block_size) {

            float best_match_sad = 1e10;
            int best_match_location[2] = {0, 0};

			for (int sx = mx - window_size; sx<mx + window_size; sx++) {
				for (int sy = my - window_size; sy<my + window_size; sy++) {
                    float current_match_sad = 0;
                    // Do the SAD
					for (int x = 0; x<block_size; x++) {
						for (int y = 0; y<block_size; y++) {
                            int match_x = mx+x;
                            int match_y = my+y;
                            int search_x = sx+x;
                            int search_y = sy+y;

                            float diff_Y = abs(match->Y->data[match_x*width+match_y] - source->Y->data[search_x*width+search_y]);
                            float diff_Cb = abs(match->Cb->data[match_x*width+match_y] - source->Cb->data[search_x*width+search_y]);
                            float diff_Cr = abs(match->Cr->data[match_x*width+match_y] - source->Cr->data[search_x*width+search_y]);
                             
                            float diff_total = Y_weight*diff_Y + Cb_weight*diff_Cb + Cr_weight*diff_Cr;
                            current_match_sad = current_match_sad + diff_total;

                        }
                    } //end SAD
                     
                    if (current_match_sad <= best_match_sad){
                        best_match_sad = current_match_sad;
                        best_match_location[0] = sx-mx;
                        best_match_location[1] = sy-my;
                    }        
                }
            }
             
            mVector v;
            v.a=best_match_location[0];
            v.b=best_match_location[1];
            motion_vectors->push_back(v);
 
        }
    }

    return motion_vectors;
}

#ifdef OpenCL
std::vector<mVector>* motionVectorSearch_cl(Frame* source, Frame* match, int width, int height) 
{
	cl_int error;
	error = clSetKernelArg(mvs_kernel, 0, sizeof(int), &width);
	checkError(error, "clSetKernelArg in");
	cl_mem args[7] = {
		Y_buffer_prev, Cb_blurred_buffer_prev, Cr_blurred_buffer_prev,
		Y_buffer, Cb_blurred_buffer, Cr_blurred_buffer,
		SAD_buffer
	};
	for (int i = 0; i < 7; i++) {
		error = clSetKernelArg(mvs_kernel, i+1, sizeof(args[i]), &(args[i]));
		checkError(error, "clSetKernelArg in");
	}
	
	
	// Enqueue the kernel
	size_t work_group_size = 512;
	size_t local_dimensions[3] = { 1, 1, work_group_size };
	size_t global_dimensions[3] = { num_match_blocks_x, num_match_blocks_y, work_group_size };
	
	error = clEnqueueNDRangeKernel(opencl_queue, mvs_kernel, 3, NULL, global_dimensions, local_dimensions, 0, NULL, NULL);
	checkError(error, "clEnqueueNDRangeKernel");
	error = clFinish(opencl_queue);
	checkError(error, "running kernel");

	
	float* SADs = new float[nSADsTotal];
	read_back_data(SAD_buffer, SADs, nSADBytes);
	std::vector<mVector> *motion_vectors = new std::vector<mVector>(); // empty list of ints
	for (int my = 0; my < num_match_blocks_y; my++) {
		for (int mx = 0; mx < num_match_blocks_x; mx++) {
			float best_match_sad = 1e10;
			int best_match_location[2] = { 0, 0 };
			for (int sy = 0; sy < NUM_LCL_SAD_VALUES_Y; sy++) {
				for (int sx = 0; sx < NUM_LCL_SAD_VALUES_X; sx++) {
					int idx = ((my * num_match_blocks_x) + mx) * NUM_LCL_SAD_VALUES + sy * NUM_LCL_SAD_VALUES_X + sx;
					float cur_sad = SADs[idx];
					if (cur_sad < best_match_sad)
					{
						best_match_sad = cur_sad;
						best_match_location[0] = sx - WINDOW_SIZE;
						best_match_location[1] = sy - WINDOW_SIZE;
					}
				}
			}
			mVector v;
			v.a = best_match_location[0];
			v.b = best_match_location[1];
			motion_vectors->push_back(v);
		}
	}
	delete SADs;
	return motion_vectors;
}
#endif

Frame* computeDelta(Frame* i_frame_ycbcr, Frame* p_frame_ycbcr, std::vector<mVector>* motion_vectors){
    Frame *delta = new Frame(p_frame_ycbcr);
 
    int width = i_frame_ycbcr->width;
    int height = i_frame_ycbcr->height;
    int window_size = 16;
    int block_size = 16;
    // How far from the edge we can go since we don't special case the edges
    int inset = (int) max((float) window_size, (float)block_size);
     
    int current_block = 0;
	for (int mx = inset; mx<height - (inset + window_size) + 1; mx += block_size) {
		for(int my=inset; my<width-(inset+window_size)+1; my+=block_size) {
        
            int vector[2];
            vector[0]=(int)motion_vectors->at(current_block).a;
            vector[1]=(int)motion_vectors->at(current_block).b;
             
            // copy the block
                for(int y=0; y<block_size; y++) {
                    for(int x=0; x<block_size; x++) {
 
                    int src_x = mx+vector[0]+x;
                    int src_y = my+vector[1]+y;
                    int dst_x = mx+x;
                    int dst_y = my+y;
                    delta->Y->data[dst_x*width+dst_y] = delta->Y->data[dst_x*width+dst_y] - i_frame_ycbcr->Y->data[src_x*width+src_y];
                    delta->Cb->data[dst_x*width+dst_y] = delta->Cb->data[dst_x*width+dst_y] - i_frame_ycbcr->Cb->data[src_x*width+src_y];
                    delta->Cr->data[dst_x*width+dst_y] = delta->Cr->data[dst_x*width+dst_y] - i_frame_ycbcr->Cr->data[src_x*width+src_y];
                }
            }
 
            current_block = current_block + 1;
        }
    }
    return delta;
}
 

Channel* downSample(Channel* in){
	int width = in->width;
	int height = in->height;
	int w2=width/2;
	int h2=height/2;

	Channel* out = new Channel((width/2),(height/2));

		for(int y2=0,y=0; y2<w2; y2++) {
			for (int x2 = 0, x = 0; x2<h2; x2++) {

            out->data[x2*w2+y2]= in->data[x*width+y];
            x+=2;
        }
        y+=2;
    }
 
    return out;
}


void dct8x8(Channel* in, Channel* out){
	int width = in->width; 
	int height = in -> height;

    // 8x8 block dct on each block
    for(int i=0; i<width*height; i++) {
        in->data[i] -= 128;
        out->data[i] = 0; //zeros
    }
 
    for(int y=0; y<width; y+=8) {
        for(int x=0; x<height; x+=8) {
            dct8x8_block(&(in->data[x*width+y]),&(out->data[x*width+y]), width);
        }
    }
}


void round_block(float* in, float* out, int stride){
    float quantMatrix[8][8] ={
        {16, 11, 10, 16,  24,  40,  51,  61},
        {12, 12, 14, 19,  26,  58,  60,  55},
        {14, 13, 16, 24,  40,  57,  69,  56},
        {14, 17, 22, 29,  51,  87,  80,  62},
        {18, 22, 37, 56,  68, 109, 103,  77},
        {24, 35, 55, 64,  81, 104, 113,  92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99},
    };
 
    for(int y=0; y<8; y++) {
        for(int x=0; x<8; x++) { 
            quantMatrix[x][y] = ceil(quantMatrix[x][y]/QUALITY);
            out[x*stride+y] = (float)round(in[x*stride+y]/quantMatrix[x][y]);
        }
    }
}


void quant8x8(Channel* in, Channel* out) {
	int width = in->width;
	int height = in->height;

    for(int i=0; i<width*height; i++) {
        out->data[i]=0; //zeros
    }
    
    for (int y=0; y<width; y+=8) {
        for (int x=0; x<height; x+=8) {       
            round_block(&(in->data[x*width+y]), &(out->data[x*width+y]), width);
        }
    }
}


void dcDiff(Channel* in, Channel* out) {
	int width = in->width;
	int height = in->height;

    int number_of_dc = width*height/64;
	double* dc_values_transposed = new double[number_of_dc];
    double* dc_values = new double[number_of_dc];
 
    int iter = 0;
    for(int j=0; j<width; j+=8){
        for(int i=0; i<height; i+=8) {
            dc_values_transposed[iter] = in->data[i*width+j];
            dc_values[iter] = in->data[i*width+j];
            iter++;
        }
    }
 
    int new_w = (int) max((float)(width/8), 1);
    int new_h = (int) max((float)(height/8), 1);
     
    out->data[0] = (float)dc_values[0];
  
    double prev = 0.;
    iter = 0;
    for (int j=0; j<new_w; j++) {
        for (int i=0; i<new_h; i++) {
            out->data[iter]= (float)(dc_values[i*new_w+j] - prev);
            prev = dc_values[i*new_w+j];
            iter++;
        }
    }
	delete dc_values_transposed;
	delete dc_values;

}

void cpyBlock(float* in, float* out, int blocksize, int stride) {
    for (int j=0; j<blocksize; j++) {
        for (int i=0; i<blocksize; i++) {
            out[i*blocksize+j] = in[i*stride+j];
        }
    }
}


void zigZagOrder(Channel* in, Channel* ordered) {
	int width = in->width;
	int height = in->height;
    int zigZagIndex[64]={0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,
        48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,
        44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63};
     
    int blockNumber=0;
    float _block[MPEG_CONSTANT];
 
    for(int x=0; x<height; x+=8) {
        for(int y=0; y<width; y+=8) {
             cpyBlock(&(in->data[x*width+y]), _block, 8, width); //block = in(x:x+7,y:y+7);
            //Put the coefficients in zig-zag order
            float zigZagOrdered[MPEG_CONSTANT] = { 0 };
            for (int index=0; index < MPEG_CONSTANT; index++){
                zigZagOrdered[index] = _block[zigZagIndex[index]];
            }
            for (int i=0; i<MPEG_CONSTANT; i++) 
                ordered->data[blockNumber*MPEG_CONSTANT+i] = zigZagOrdered[i];
            blockNumber++;
        }
    }
}


void encode8x8(Channel* ordered, SMatrix* encoded){
	int width = encoded->height;
	int height = encoded->width;
    int num_blocks = height;
    
	for(int i=0; i<num_blocks; i++) {
		std::string block_encode[MPEG_CONSTANT];
		for (int j=0; j<MPEG_CONSTANT; j++) {
            block_encode[j]="\0"; //necessary to initialize every string position to empty string
        }

		double* block = new double[width];
        for(int y=0; y<width; y++) block[y] = ordered->data[i*width+y];
        int num_coeff = MPEG_CONSTANT; //width
        int encoded_index = 0;
        int in_zero_run = 0;
        int zero_count = 0;
 
        // Skip DC coefficient
        for(int c=1; c<num_coeff; c++){
            double coeff = block[c];
            if (coeff == 0){
                if (in_zero_run == 0){
                    zero_count = 0;
                    in_zero_run = 1;
                }
                zero_count = zero_count + 1;
            }
            else {
                if (in_zero_run == 1){
                    in_zero_run = 0;
					block_encode[encoded_index] = "Z" + std::to_string(zero_count);
                    encoded_index = encoded_index+1;
                }
				block_encode[encoded_index] = std::to_string((int)coeff);
                encoded_index = encoded_index+1;
            }
        }
 
        // If we were in a zero run at the end attach it as well.    
        if (in_zero_run == 1) {
            if (zero_count > 1) {
				block_encode[encoded_index] = "Z" + std::to_string(zero_count);
            } else {
				block_encode[encoded_index] = "0";
            }
        }
 
 
        for(int it=0; it < MPEG_CONSTANT; it++) {
			if (block_encode[it].length() > 0) 
				encoded->data[i*width+it] = new std::string(block_encode[it]);
            else
                it = MPEG_CONSTANT;
        }
		delete block;
    }
}

void swap_buffers(cl_mem* a, cl_mem* b) {
	cl_mem temp = *a;
	*a = *b;
	*b = temp;
}

int encode() {
    int end_frame = int(N_FRAMES);
    int i_frame_frequency = int(I_FRAME_FREQ);
	struct timeval starttime, endtime;
	double runtime[10] = {0};
    
    // Hardcoded paths
    string image_path =  "..\\..\\inputs\\" + string(image_name) + "\\" + image_name + ".";
	string stream_path = "..\\..\\outputs\\stream_c_" + string(image_name) + ".xml";

    xmlDocPtr stream = NULL;
 
    Image* frame_rgb = NULL;
    Image* previous_frame_rgb = NULL;
    Frame* previous_frame_lowpassed = NULL;
 
    loadImage(0, image_path, &frame_rgb);
 
    int width = frame_rgb->width;
    int height = frame_rgb->height;
    int npixels = width*height;

#ifdef OpenCL
	setup_cl(width, height);
#endif
 
	delete frame_rgb;

	createStatsFile();
    stream = create_xml_stream(width, height, QUALITY, WINDOW_SIZE, BLOCK_SIZE);
    vector<mVector>* motion_vectors = NULL;

    for (int frame_number = 0 ; frame_number < end_frame ; frame_number++) {
		frame_rgb = NULL;
        loadImage(frame_number, image_path, &frame_rgb);

        //  Convert to YCbCr
		print("Covert to YCbCr...");
        
		Image* frame_ycbcr = new Image(width, height, FULLSIZE);
		gettimeofday(&starttime, NULL);
#ifdef OpenCL
		start_perf_measurement(&total_perf);
		convertRGBtoYCbCr_cl(frame_rgb, frame_ycbcr);
		stop_perf_measurement(&total_perf);
#else
		convertRGBtoYCbCr(frame_rgb, frame_ycbcr);
#endif
		gettimeofday(&endtime, NULL);
		runtime[0] = double(endtime.tv_sec)*1000.0f + double(endtime.tv_usec)/1000.0f - double(starttime.tv_sec)*1000.0f - double(starttime.tv_usec)/1000.0f; //in ms
        
		dump_image(frame_ycbcr, "frame_ycbcr", frame_number);
		delete frame_rgb;
 
        // We low pass filter Cb and Cr channesl
        print("Low pass filter..."); 

		gettimeofday(&starttime, NULL);
		Channel* frame_blur_cb = new Channel(width, height);
        Channel* frame_blur_cr = new Channel(width, height);
		Frame *frame_lowpassed = new Frame(width, height, FULLSIZE);
		
#ifndef OpenCL
		lowPass(frame_ycbcr->gc, frame_blur_cb);
		lowPass(frame_ycbcr->bc, frame_blur_cr);
#else
		lowPass_cl(Cb_buffer, Cb_blurred_buffer, frame_blur_cb);
		lowPass_cl(Cr_buffer, Cr_blurred_buffer, frame_blur_cr);
#endif

		frame_lowpassed->Y->copy(frame_ycbcr->rc);
		frame_lowpassed->Cb->copy(frame_blur_cb);
        frame_lowpassed->Cr->copy(frame_blur_cr);
		gettimeofday(&endtime, NULL);
		runtime[1] = double(endtime.tv_sec)*1000.0f + double(endtime.tv_usec)/1000.0f - double(starttime.tv_sec)*1000.0f - double(starttime.tv_usec)/1000.0f; //in ms   
        
		dump_frame(frame_lowpassed, "frame_ycbcr_lowpass", frame_number);
		delete frame_ycbcr; 
		delete frame_blur_cb; 
		delete frame_blur_cr;

        Frame *frame_lowpassed_final = NULL;
 
        if (frame_number % i_frame_frequency != 0) { 
            // We have a P frame 
            // Note that in the first iteration we don't enter this branch!
            
			//Compute the motion vectors
			print("Motion Vector Search...");

			gettimeofday(&starttime, NULL);
#ifndef OpenCL
            motion_vectors = motionVectorSearch(previous_frame_lowpassed, frame_lowpassed, frame_lowpassed->width, frame_lowpassed->height);
#else
			motion_vectors = motionVectorSearch_cl(previous_frame_lowpassed, frame_lowpassed, frame_lowpassed->width, frame_lowpassed->height);
#endif
			gettimeofday(&endtime, NULL);
			runtime[2] = double(endtime.tv_sec)*1000.0f + double(endtime.tv_usec)/1000.0f - double(starttime.tv_sec)*1000.0f - double(starttime.tv_usec)/1000.0f; //in ms 

			print("Compute Delta...");
			gettimeofday(&starttime, NULL);
            frame_lowpassed_final = computeDelta(previous_frame_lowpassed, frame_lowpassed, motion_vectors);
			gettimeofday(&endtime, NULL);
			runtime[3] = double(endtime.tv_sec)*1000.0f + double(endtime.tv_usec)/1000.0f - double(starttime.tv_sec)*1000.0f - double(starttime.tv_usec)/1000.0f; //in ms 

        } else {
            // We have a I frame 
            motion_vectors = NULL;
            frame_lowpassed_final = new Frame(frame_lowpassed);
        }
		delete frame_lowpassed; frame_lowpassed=NULL;

		if (frame_number > 0) delete previous_frame_lowpassed;
        previous_frame_lowpassed = new Frame(frame_lowpassed_final); 

		swap_buffers(&Y_buffer_prev, &Y_buffer);
		swap_buffers(&Cb_blurred_buffer_prev, &Cb_blurred_buffer);
		swap_buffers(&Cr_blurred_buffer_prev, &Cr_blurred_buffer);
 
        // Downsample the difference
		print("Downsample...");
		
		gettimeofday(&starttime, NULL);
        Frame* frame_downsampled = new Frame(width, height, DOWNSAMPLE);
 		
        // We don't touch the Y frame
		frame_downsampled->Y->copy(frame_lowpassed_final->Y);
        Channel* frame_downsampled_cb = downSample(frame_lowpassed_final->Cb);
		frame_downsampled->Cb->copy(frame_downsampled_cb);       
        Channel* frame_downsampled_cr = downSample(frame_lowpassed_final->Cr);
		frame_downsampled->Cr->copy(frame_downsampled_cr);
        gettimeofday(&endtime, NULL);
		runtime[4] = double(endtime.tv_sec)*1000.0f + double(endtime.tv_usec)/1000.0f - double(starttime.tv_sec)*1000.0f - double(starttime.tv_usec)/1000.0f; //in ms 

        dump_frame(frame_downsampled, "frame_downsampled", frame_number);
		delete frame_lowpassed_final; 
		delete frame_downsampled_cb; 
		delete frame_downsampled_cr; 

        // Convert to frequency domain
		print("Convert to frequency domain...");
        
		gettimeofday(&starttime, NULL);
		Frame* frame_dct = new Frame(width, height, DOWNSAMPLE);
		
        dct8x8(frame_downsampled->Y, frame_dct->Y);
        dct8x8(frame_downsampled->Cb, frame_dct->Cb);
        dct8x8(frame_downsampled->Cr, frame_dct->Cr);
        gettimeofday(&endtime, NULL);
		runtime[5] = double(endtime.tv_sec)*1000.0f + double(endtime.tv_usec)/1000.0f - double(starttime.tv_sec)*1000.0f - double(starttime.tv_usec)/1000.0f; //in ms 

        dump_frame(frame_dct, "frame_dct", frame_number);
		delete frame_downsampled;
		
        //Quantize the data
		print("Quantize...");
		
		gettimeofday(&starttime, NULL);
        Frame* frame_quant = new Frame(width, height, DOWNSAMPLE);

        quant8x8(frame_dct->Y, frame_quant->Y);
		quant8x8(frame_dct->Cb, frame_quant->Cb);
		quant8x8(frame_dct->Cr, frame_quant->Cr);
        gettimeofday(&endtime, NULL);
		runtime[6] = double(endtime.tv_sec)*1000.0f + double(endtime.tv_usec)/1000.0f - double(starttime.tv_sec)*1000.0f - double(starttime.tv_usec)/1000.0f; //in ms      
        
		dump_frame(frame_quant, "frame_quant", frame_number);
		delete frame_dct;

        //Extract the DC components and compute the differences
		print("Compute DC differences...");
        
		gettimeofday(&starttime, NULL);  
		Frame* frame_dc_diff = new Frame(1, (width/8)*(height/8), DCDIFF); //dealocate later
		   
        dcDiff(frame_quant->Y, frame_dc_diff->Y);
        dcDiff(frame_quant->Cb, frame_dc_diff->Cb);
        dcDiff(frame_quant->Cr, frame_dc_diff->Cr);
		gettimeofday(&endtime, NULL);
		runtime[7] = double(endtime.tv_sec)*1000.0f + double(endtime.tv_usec)/1000.0f - double(starttime.tv_sec)*1000.0f - double(starttime.tv_usec)/1000.0f; //in ms      
         
        dump_dc_diff(frame_dc_diff, "frame_dc_diff", frame_number);

		// Zig-zag order for zero-counting
		print("Zig-zag order...");
        gettimeofday(&starttime, NULL);
		
		Frame* frame_zigzag = new Frame(MPEG_CONSTANT, width*height/MPEG_CONSTANT, ZIGZAG);
		
        zigZagOrder(frame_quant->Y, frame_zigzag->Y);
        zigZagOrder(frame_quant->Cb, frame_zigzag->Cb);
        zigZagOrder(frame_quant->Cr, frame_zigzag->Cr);
		gettimeofday(&endtime, NULL);
		runtime[8] = double(endtime.tv_sec)*1000.0f + double(endtime.tv_usec)/1000.0f - double(starttime.tv_sec)*1000.0f - double(starttime.tv_usec)/1000.0f; //in ms  
        
		dump_zigzag(frame_zigzag, "frame_zigzag", frame_number);
		delete frame_quant;
 
        // Encode coefficients
		print("Encode coefficients...");
        
		gettimeofday(&starttime, NULL);
		FrameEncode* frame_encode = new FrameEncode(width, height, MPEG_CONSTANT);
 
        encode8x8(frame_zigzag->Y, frame_encode->Y);
        encode8x8(frame_zigzag->Cb, frame_encode->Cb);
        encode8x8(frame_zigzag->Cr, frame_encode->Cr);
		gettimeofday(&endtime, NULL);
		runtime[9] = double(endtime.tv_sec)*1000.0f + double(endtime.tv_usec)/1000.0f - double(starttime.tv_sec)*1000.0f - double(starttime.tv_usec)/1000.0f; //in ms  
       
		delete frame_zigzag;
	       
        stream_frame(stream, frame_number, motion_vectors, frame_number-1, frame_dc_diff, frame_encode);
        write_stream(stream_path, stream);

		delete frame_dc_diff;
		delete frame_encode;
 
        if (motion_vectors != NULL) {
            free(motion_vectors);
			motion_vectors = NULL;
        }

		writestats(frame_number, frame_number % i_frame_frequency, runtime);

    }
	
	closeStats();
	
	
    
	return 0;
}
 

int main(int argc, const char * argv[]) {
	auto begin = chrono::high_resolution_clock::now();
#ifdef OpenCL
	if (argc == 2) {
		opencl_cores = atoi(argv[1]);
	}
	setup_cl(argc, argv, &opencl_device, &opencl_context, &opencl_queue);
#endif
	/*
	char pBuf[1000];
	int bytes = GetModuleFileName(NULL, pBuf, 1000);
	if(bytes == 0)
		return -1;
	else
		return bytes;
		*/
	
#ifdef OMP
	omp_set_num_threads(NUM_THREADS);
	omp_set_dynamic(0);
#endif
    encode();
#ifdef OpenCL
	print_perfs();
#endif
	auto end = chrono::high_resolution_clock::now();
	auto dur = end - begin;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	cout << "ms: " << ms << endl;
	//system("pause");
    return 0;
}

#ifdef OpenCL
void init_all_perfs() {
	init_perf(&total_perf);
	init_perf(&convert_perf);
	init_perf(&program_perf);
	init_perf(&create_perf);
	init_perf(&write_perf);
	init_perf(&read_perf);
	init_perf(&finish_perf);
	init_perf(&cleanup_perf);

}

void print_perfs() {
	printf("Total:          ");
	print_perf_measurement(&total_perf);
	printf("Convert Kernel:  ");
	print_perf_measurement(&convert_perf);

	printf("Read Data:      ");
	print_perf_measurement(&read_perf);

	printf("Compile Program:");
	print_perf_measurement(&program_perf);
	printf("Create Buffers: ");
	print_perf_measurement(&create_perf);
	printf("Write Data:     ");
	print_perf_measurement(&write_perf);
	printf("Finish:         ");
	print_perf_measurement(&finish_perf);
	printf("Cleanup:        ");
	print_perf_measurement(&cleanup_perf);

}
#endif