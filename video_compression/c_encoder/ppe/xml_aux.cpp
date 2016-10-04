#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include "xml_aux.h"
#include "assert.h"
#include "config.h"
#include <map>
#include <iomanip>
#include <windows.h>

using namespace std;

void print(std::string s){
	cout << s << endl;
}

xmlDocPtr create_xml_stream(int width, int height, int quality, int window_size, int block_size){
	xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr root_node = xmlNewNode(NULL, BAD_CAST "STREAM");
    xmlDocSetRootElement(doc, root_node);
    
    char buf[16];
    sprintf_s(buf, "%d", width);
	xmlNewProp(root_node, BAD_CAST "width", BAD_CAST buf);

	sprintf_s(buf, "%d", height);
    xmlNewProp(root_node, BAD_CAST "height", BAD_CAST buf);

	sprintf_s(buf, "%d", quality);
    xmlNewProp(root_node, BAD_CAST "quality", BAD_CAST buf);

	sprintf_s(buf, "%d", window_size);
    xmlNewProp(root_node, BAD_CAST "window_size", BAD_CAST buf);

	sprintf_s(buf, "%d", block_size);
    xmlNewProp(root_node, BAD_CAST "block_size", BAD_CAST buf);

    //xmlSaveFormatFileEnc("-", doc, "UTF-8", 1);
    
    return doc;
}



std::string vector2str_(float* mat, int size){
	std::string buf = "[ ";
	//char aux[20];
	for(int i=0; i<size; i++){
		if(i!=0) buf += " ";
		buf+=std::to_string((int)mat[i]); //sprintf_s(aux, "%.0f", mat[i]);
		//strcat_s(buf, bufsize, aux);
	}
	buf+="]";//strcat_s(buf,bufsize,"]");
	return buf;
}


std::string array2str_(int* block_count, std::string** arr, int size){
    int i=0;
	std::string buf;
    while(arr[i] != NULL && i<size) {
		if(i>0) buf+=" ";
		buf+=(*(arr[i]));
        i++;
    }
	*block_count = i;
	return buf;
}

void stream_image(xmlDocPtr XML, xmlNodePtr parentNode, Channel* dc_diff, SMatrix* image_zero/*, int width, int height, int image_zero_size*/){
	xmlNodePtr dcNode = xmlNewNode(NULL, BAD_CAST "DC");
	std::string buf;
	std::string res =  vector2str_(dc_diff->data, dc_diff->height/*image_zero_size*/); //TO DO transpose matrix?
	xmlNodePtr content = xmlNewText(BAD_CAST res.c_str());
	xmlAddChild(dcNode, content);
	xmlAddChild(parentNode, dcNode);
	
	xmlNodePtr blocksNode = xmlNewNode(NULL, BAD_CAST "BLOCKS");
	int block_count = 0;
	int aux = 0;
	
	for(int i=0; i<image_zero->width/*image_zero_size*/; i++){
        xmlNodePtr bNode = xmlNewNode(NULL, BAD_CAST "B");
        buf ="\0";
		buf += std::to_string(i+1);
		xmlNewProp(bNode, BAD_CAST "id", BAD_CAST buf.c_str());
		std::string* s = image_zero->data[i*64];
		buf.clear();
		std::string res2 = array2str_(&aux, &(image_zero->data[i*64]), 64);
		block_count +=aux;
		xmlNodePtr text_content = xmlNewText(BAD_CAST res2.c_str());
        xmlAddChild(bNode, text_content);
        xmlAddChild(blocksNode,bNode);
	}
	buf.clear();
	buf += std::to_string(block_count);
	xmlNewProp(blocksNode, BAD_CAST "coeffs", BAD_CAST buf.c_str());
    xmlAddChild(parentNode,blocksNode);
	
}

void stream_frame(xmlDocPtr XML, int frame_number, std::vector<mVector>* motion_vectors, int ref_frame_number, Frame* dc_diff, FrameEncode* image_zero){
	//char buf[16];
	std::string buf;
	xmlNodePtr rootNode  = xmlDocGetRootElement(XML);
	xmlNodePtr frameNode = xmlNewNode(NULL, BAD_CAST "FRAME");
	xmlAddChild(rootNode, frameNode);
		
	buf = std::to_string(frame_number);//sprintf_s(buf, "%d", frame_number);
	xmlNewProp(frameNode, BAD_CAST "number", BAD_CAST buf.c_str());
	if (motion_vectors==NULL)
		xmlNewProp(frameNode, BAD_CAST "type", BAD_CAST "I");
	else {
		xmlNewProp(frameNode, BAD_CAST "type", BAD_CAST "P");
		buf = std::to_string(ref_frame_number); //sprintf(buf, "%d", ref_frame_number);
    	xmlNewProp(frameNode, BAD_CAST "base", BAD_CAST buf.c_str());
    	xmlNodePtr motionVectorsNode = xmlNewNode(NULL, BAD_CAST "MOTION_VECTORS");
		for (int i=0; i<(int)motion_vectors->size(); i++) {
            xmlNodePtr mvNode = xmlNewNode(NULL, BAD_CAST "MV");
			buf.clear();//buf[0]='\0';
            //sprintf(buf,"[%d %d]", motion_vectors->at(i).a, motion_vectors->at(i).b);
			buf = "[" + std::to_string( motion_vectors->at(i).a) + " " + std::to_string(motion_vectors->at(i).b) + "]";
			xmlNodePtr content = xmlNewText(BAD_CAST buf.c_str());
    		xmlAddChild(mvNode, content);
    		xmlAddChild(motionVectorsNode,mvNode);
        }
    	xmlAddChild(frameNode,motionVectorsNode);
	}
	
	xmlNodePtr YNode = xmlNewNode(NULL, BAD_CAST "Y");
	xmlNodePtr CrNode = xmlNewNode(NULL, BAD_CAST "Cr");
	xmlNodePtr CbNode = xmlNewNode(NULL, BAD_CAST "Cb");
	xmlAddChild(frameNode,YNode);
	xmlAddChild(frameNode,CrNode);
	xmlAddChild(frameNode,CbNode);
	
	
	stream_image(XML, YNode, dc_diff->Y, image_zero->Y/*, 128, 128, 256*/);
	stream_image(XML, CrNode, dc_diff->Cr, image_zero->Cr/*, 64, 64, 64*/);
	stream_image(XML, CbNode, dc_diff->Cb, image_zero->Cb/*, 64, 64, 64*/);
	
}

void write_stream(std::string stream_path, xmlDocPtr stream){
	xmlSaveFormatFileEnc(stream_path.c_str(), stream, "UTF-8", 1);
}

void dump_matrix(float *matrix, int width, int height, const char *filename, int DOWNSAMPLED) {
	if (!DUMP_TO_DEBUG) return; 

	assert(matrix != NULL);
	assert(filename != NULL);
	assert(width>=0 && height>=0);
	
 	FILE *fp;
	fopen_s(&fp, filename, "w");
	assert(fp != NULL);

	int w = width;
	int h = height;

//	if (DOWNSAMPLED) {
//		w = max(w/2, 1);
//		h = max(1, h/2);
//	}

	for (int i=0; i<h ; i++) {
		for (int j=0; j<w; j++) {
			fprintf(fp, "%g\t", matrix[i*w + j]);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
}

void dump_image(Image* image, const char *name, int frame_number) {
	if (!DUMP_TO_DEBUG) return; 

	assert(image != NULL);

	std::wstring dumpDirPath = L"..\\..\\dump";
	CreateDirectory(dumpDirPath.c_str(), NULL);

	std::wstring dirPath = L"..\\..\\dump\\c";
	CreateDirectory(dirPath.c_str(), NULL);

	string filename = "..//..//dump//c//";
	filename.append(to_string((_Longlong)frame_number));
	filename.append("-");
	filename.append(name);

	string rc_filename = filename.c_str();
	string gc_filename = filename.c_str();
	string bc_filename = filename.c_str();

	rc_filename.append(".Y");
	rc_filename.append(".mat");
	printf("Dumping %s\n", rc_filename.c_str());

	dump_matrix(image->rc->data, image->rc->width, image->rc->height, rc_filename.c_str(), false);

	bc_filename.append(".Cb");
	bc_filename.append(".mat");
	printf("Dumping %s\n", bc_filename.c_str());
	
	dump_matrix(image->gc->data, image->gc->width, image->gc->height, bc_filename.c_str(), image->type);

	gc_filename.append(".Cr");
	gc_filename.append(".mat");
	printf("Dumping %s\n", gc_filename.c_str());

	dump_matrix(image->bc->data, image->bc->width, image->bc->height, gc_filename.c_str(), image->type);
	
	return;
}



void dump_zigzag(Frame* frame, const char *name, int frame_number) {
	if (!DUMP_TO_DEBUG) return; 

	assert(frame != NULL);

	string filename = "..//..//dump//c//";
	filename.append(to_string((_Longlong)frame_number));
	filename.append("-");

	filename.append(name);

	string Y_filename = filename.c_str();
	string Cr_filename = filename.c_str();
	string Cb_filename = filename.c_str();

	Y_filename.append(".Y");
	Y_filename.append(".mat");
	printf("Dumping %s\n", Y_filename.c_str());

	dump_matrix(frame->Y->data, frame->Y->width, frame->Y->height, Y_filename.c_str(), false);

	Cb_filename.append(".Cb");
	Cb_filename.append(".mat");
	printf("Dumping %s\n", Cb_filename.c_str());
	
	dump_matrix(frame->Cb->data, frame->Cb->width, frame->Cb->height, Cb_filename.c_str(), false);

	Cr_filename.append(".Cr");
	Cr_filename.append(".mat");
	printf("Dumping %s\n", Cr_filename.c_str());

	dump_matrix(frame->Cr->data, frame->Cr->width, frame->Cr->height, Cr_filename.c_str(), false);
}

void dump_dc_diff(Frame* frame, const char *name, int frame_number) {
	if (!DUMP_TO_DEBUG) return; 

	assert(frame != NULL);

	string filename = "..//..//dump//c//";
	filename.append(to_string((_Longlong)frame_number));
	filename.append("-");

	filename.append(name);

	string Y_filename = filename.c_str();
	string Cr_filename = filename.c_str();
	string Cb_filename = filename.c_str();

	Y_filename.append(".Y");
	Y_filename.append(".mat");
	printf("Dumping %s\n", Y_filename.c_str());

	dump_matrix(frame->Y->data, frame->Y->width, frame->Y->height, Y_filename.c_str(), false);

	Cb_filename.append(".Cb");
	Cb_filename.append(".mat");
	printf("Dumping %s\n", Cb_filename.c_str());
	
	//dump_matrix(frame->Cb, 1, frame->height/4, Cb_filename.c_str(), false);
	dump_matrix(frame->Cb->data, frame->Cb->width, frame->Cb->height, Cb_filename.c_str(), false);

	Cr_filename.append(".Cr");
	Cr_filename.append(".mat");
	printf("Dumping %s\n", Cr_filename.c_str());

	//dump_matrix(frame->Cr, 1, frame->height/4, Cr_filename.c_str(), false);
	dump_matrix(frame->Cr->data, frame->Cr->width, frame->Cr->height, Cr_filename.c_str(), false);
}


void dump_frame(Frame* frame, const char *name, int frame_number) {
	if (!DUMP_TO_DEBUG) return; 

	assert(frame != NULL);

	string filename = "..//..//dump//c//";
	filename.append(to_string((_Longlong)frame_number));
	filename.append("-");

	filename.append(name);

	string Y_filename = filename.c_str();
	string Cr_filename = filename.c_str();
	string Cb_filename = filename.c_str();

	Y_filename.append(".Y");
	Y_filename.append(".mat");
	printf("Dumping %s\n", Y_filename.c_str());

	dump_matrix(frame->Y->data, frame->Y->width, frame->Y->height, Y_filename.c_str(), false);

	Cb_filename.append(".Cb");
	Cb_filename.append(".mat");
	printf("Dumping %s\n", Cb_filename.c_str());
	
	dump_matrix(frame->Cb->data, frame->Cb->width, frame->Cb->height, Cb_filename.c_str(), frame->type);

	Cr_filename.append(".Cr");
	Cr_filename.append(".mat");
	printf("Dumping %s\n", Cr_filename.c_str());

	dump_matrix(frame->Cr->data, frame->Cr->width, frame->Cr->height, Cr_filename.c_str(), frame->type);
}
/*
void dump_motion_vectors(int* motion, int frame_number) {
	if (!DUMP_TO_DEBUG) return; 
	printf("Dumping Motion Vectors\n");
}

*/

std::string function_name[10] = {"Convert to YCbCr:", "Low pass filter:", "Motion Vector Search:", "Compute Delta:", "Downsample:", "Convert to frequency domain:", "Quantize:", "Compute DC differences:", "Zig-zag order:", "Encode coefficients:"};
double runtime_accum[10] = {0};

void createStatsFile(void){
	std::wstring dirPath = L"..\\..\\outputs";
	CreateDirectory(dirPath.c_str(), NULL);
	std::string path = "..\\..\\outputs\\execution_stats.txt";
	if (std::remove(path.c_str()) !=0) return;
}

void writestats(int framenum, int is_pframe, double* runtime){
	std::ofstream file;
	file.open("..\\..\\outputs\\execution_stats.txt", ios::app);
	string ftype = is_pframe?"P":"I";
	file << "********* " << "Frame: " << std::to_string(framenum) << " (" << ftype << ")" << " *********" << std::endl;
	for(int i=0; i<10; i++){
		if( !( (i==2 || i==3) && !is_pframe) ){
			runtime_accum[i] += runtime[i]; 
			file << setw(30) << left << function_name[i] << std::setprecision(0) << runtime[i] << "ms" << std::endl;
		}
	}
	file << std::endl;
	file.close();
}

void closeStats(void){
	double total = 0;
	std::ofstream file;
	file.open("..\\..\\outputs\\execution_stats.txt", ios::app);
	file << "********* " << "    Total   " << " *********" << std::endl;
	for(int i=0; i<10; i++){
		total += runtime_accum[i];
		file << setw(30) << left << function_name[i] << std::setprecision(0) << runtime_accum[i] << "ms" << std::endl;
	}
	file << std::endl;
	file << setw(30) << left << "Total runtime: " << total << "ms" << std::endl;
	file.close();
}
