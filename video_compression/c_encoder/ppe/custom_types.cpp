#include "custom_types.h"

#include <stdlib.h>
#include <vector>
#include <math.h>
#include "config.h"


float max(float a, float b) { 
    if(a > b) return a;
    return b;
}

/*
float round(float number){
    return number < 0.0 ? (float)ceil(number - 0.5) : (float)floor(number + 0.5);
}
*/

Channel::Channel(int _width, int _height){
	width = _width;
	height = _height;
	data = new float[_width*_height];
}

Channel::Channel(Channel* in){
	width = in->width;
	height = in->height;
	
	int npixels = in->width*in->height;
	data = new float[npixels];
	
	for(int i=0; i<npixels; i++)
		data[i] = in->data[i];
}

Channel::~Channel(){
	delete data;
}
/*
void Channel::operator=(Channel* ch){
	int npixels = ch->width * ch->height;
	
	this->width = ch->width;
	this->height = ch->height;
	
	for(int i=0; i<npixels; i++)
		this->data[i] = ch->data[i];
}
*/

void Channel::copy(Channel* ch){
	int npixels = ch->width * ch->height;
	
	this->width = ch->width;
	this->height = ch->height;
	
	for(int i=0; i<npixels; i++)
		this->data[i] = ch->data[i];
}

Image::Image(int _w, int _h, int _type){
	width = _w;
	height = _h;
	type = _type;

	rc = new Channel(_w, _h);
	if (_type == DOWNSAMPLE) {
		_w = _w/2;
		_h = _h/2;
	}
	gc = new Channel(_w, _h);
	bc = new Channel(_w, _h);
}

Image::~Image(){
	delete this->rc;
	delete this->gc;
	delete this->bc;
	this->rc=NULL;
	this->gc=NULL;
	this->bc=NULL;
}

Frame::Frame(int _w, int _h, int _type){
	width = _w;
	height = _h;
	type = _type;

	Y = new Channel(_w, _h);
	if (_type == DOWNSAMPLE) {
		_w = _w/2;
		_h = _h/2;
	}
	if (_type == DCDIFF){
		_w = (int)max(float(_w/4), 1.);
		_h = (int)max(float(_h/4), 1.);
	}
	if (_type == ZIGZAG){
		_h = _h/4;
	}

	Cb = new Channel(_w, _h);
	Cr = new Channel(_w, _h);
}

Frame::Frame(Frame* in){
	width = in->width;
	height = in->height;
	type = in->type;

	Y = new Channel(in->Y);
	Cb = new Channel(in->Cb);
	Cr = new Channel(in->Cr);

}

Frame::~Frame(){
	delete Y;
	delete Cb;
	delete Cr;
	Y=NULL;
	Cb=NULL;
	Cr=NULL;
}



SMatrix::SMatrix(int _width, int _height){
	width = _width;
	height = _height;
	//data = (std::string**)malloc(_width*_height*sizeof(std::string*));
	data = new std::string*[_width*_height];
	for(int i=0; i<_width*_height; i++) data[i] = NULL;
}

SMatrix::~SMatrix(){
	int n = width * height;
	for(int i=0; i<n; i++) delete data[i];
	delete data;
}


FrameEncode::FrameEncode(int _w, int _h, int _mpg){
	width = _w*_h/MPEG_CONSTANT;
	height = MPEG_CONSTANT;

	Y = new SMatrix(_w*_h/MPEG_CONSTANT, MPEG_CONSTANT);
	Cb = new SMatrix((_w/2)*(_h/2)/MPEG_CONSTANT, MPEG_CONSTANT);
	Cr = new SMatrix((_w/2)*(_h/2)/MPEG_CONSTANT, MPEG_CONSTANT);
}


FrameEncode::~FrameEncode(){
	delete Y;
	delete Cb;
	delete Cr;
	Y=NULL;
	Cb=NULL;
	Cr=NULL;
}