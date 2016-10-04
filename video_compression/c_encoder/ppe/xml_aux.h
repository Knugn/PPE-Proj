#ifndef xml_aux_h
#define xml_aux_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "custom_types.h"
#include <libxml/parser.h>
#include <libxml/tree.h>

void print(std::string s);

xmlDocPtr create_xml_stream(int width, int height, int quality, int window_size, int block_size);

void mat2str(char* buf, float* mat, int width, int height);

int array2str(char* buf, char** arr, int size);

void stream_image(xmlDocPtr XML, xmlNodePtr parentNode, Channel* dc_diff, SMatrix* image_zero, int width, int height, int image_zero_size);

void stream_frame(xmlDocPtr XML, int frame_number, std::vector<mVector>* motion_vectors, int ref_frame_number, Frame* dc_diff, FrameEncode* image_zero);

void write_stream(std::string stream_path, xmlDocPtr stream);


void dump_image(Image* image, const char *name, int frame_number);

void dump_frame(Frame* frame, const char *name, int frame_number);

void dump_dc_diff(Frame* frame, const char *name, int frame_number);

void dump_zigzag(Frame* frame, const char *name, int frame_number);
/*
void dump_motion_vectors(int* motion, int frame_number);
*/

void createStatsFile(void);

void writestats(int framenum, int is_pframe, double* runtime);

void closeStats(void);

#endif
