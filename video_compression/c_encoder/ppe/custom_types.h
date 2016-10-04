#ifndef custom_types_h
#define custom_types_h

#include <vector>
#include <string>


float max(float a, float b);
 
//float round(float number);


class Channel{
public:
	float* data;
	//std::vector<float> *data;
	int width;
    int height;

	Channel(int _width, int _height);

	Channel(Channel* in);

	~Channel();
	//void operator=(Channel* c);
	void copy(Channel* c);
}; 


class Image{
public:
	Channel *rc;
    Channel *gc;
    Channel *bc;
    int width;
    int height;
    // Flag to check if the image is downsampled or not
    int type;

	Image(int _w, int _h, int _type);

	~Image();
};


class Frame{
	public:
	Channel *Y;
    Channel *Cb;
    Channel *Cr;
    int width;
    int height;
    // Flag to check if the image is downsampled or not
    int type;

	Frame(int _w, int _h, int _type);
	Frame(Frame* in);

	~Frame();

};


class SMatrix{
public:
	std::string** data;
	int width;
    int height;

	SMatrix(int _width, int _height);

	~SMatrix();
	//void operator=(SMatrix* c);
}; 

class FrameEncode{
	public:
	SMatrix *Y;
    SMatrix *Cb;
    SMatrix *Cr;
    int width;
    int height;
    // Flag to check if the image is downsampled or not
    int type;

	FrameEncode(int _w, int _h, int _mpg);

	~FrameEncode();

};

typedef struct smVector{
    int a;
    int b;
} mVector;


#endif