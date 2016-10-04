#include "dct8x8_block.h"

void dct8x8_block(float* in_8x8, float* out, int stride){
	double c1=0.980785;
	double c2=0.923880;
	double c3=0.831470;
	double c4=0.707107;
	double c5=0.555570;
	double c6=0.382683;
	double c7=0.195090;
	
	double One_D_DCT_Row_8x8[8][8];
	
	for(int row_number=0; row_number< 8; row_number++){
        
    	//sample image value initialization from input matrix
		double f0=in_8x8[row_number*stride+0];
		double f1=in_8x8[row_number*stride+1];
		double f2=in_8x8[row_number*stride+2];
		double f3=in_8x8[row_number*stride+3];
		double f4=in_8x8[row_number*stride+4];
		double f5=in_8x8[row_number*stride+5];
		double f6=in_8x8[row_number*stride+6];
		double f7=in_8x8[row_number*stride+7];
        
   		//first stage of FLOWGRAPH (Chen,Fralick and Smith)
		double i0=f0+f7;
		double i1=f1+f6;
		double i2=f2+f5;
		double i3=f3+f4;
		double i4=f3-f4;
		double i5=f2-f5;
		double i6=f1-f6;
		double i7=f0-f7;
        
		//second stage of FLOWGRAPH (Chen,Fralick and Smith)
		double j0=i0+i3;
		double j1=i1+i2;
		double j2=i1-i2;
		double j3=i0-i3;
		double j4=i4;
		double j5=(i6-i5)*c4;
		double j6=(i6+i5)*c4;
		double j7=i7;
        
		//third stage of FLOWGRAPH (Chen,Fralick and Smith)
		double k0=(j0+j1)*c4;
		double k1=(j0-j1)*c4;
		double k2=(j2*c6)+(j3*c2);
		double k3=(j3*c6)-(j2*c2);
		double k4=j4+j5;
		double k5=j4-j5;
		double k6=j7-j6;
		double k7=j7+j6;
        
		//fourth stage of FLOWGRAPH; 1-dimensional DCT coefficients
		double F0=k0/2;
		double F1=(k4*c7+k7*c1)/2;
		double F2=k2/2;
		double F3=(k6*c3-k5*c5)/2;
		double F4=k1/2;
		double F5=(k5*c3+k6*c5)/2;
		double F6=k3/2;
		double F7=(k7*c7-k4*c1)/2;
        
		//DCT coefficient assignment
		One_D_DCT_Row_8x8[row_number][0]=F0;
		One_D_DCT_Row_8x8[row_number][1]=F1;
		One_D_DCT_Row_8x8[row_number][2]=F2;
		One_D_DCT_Row_8x8[row_number][3]=F3;
		One_D_DCT_Row_8x8[row_number][4]=F4;
		One_D_DCT_Row_8x8[row_number][5]=F5;
		One_D_DCT_Row_8x8[row_number][6]=F6;
		One_D_DCT_Row_8x8[row_number][7]=F7;
	}
	
	
	for (int column_number=0; column_number<8; column_number++){
		
		//sample image value initialization
		double f0=One_D_DCT_Row_8x8[0][column_number];
		double f1=One_D_DCT_Row_8x8[1][column_number];
		double f2=One_D_DCT_Row_8x8[2][column_number];
		double f3=One_D_DCT_Row_8x8[3][column_number];
		double f4=One_D_DCT_Row_8x8[4][column_number];
		double f5=One_D_DCT_Row_8x8[5][column_number];
		double f6=One_D_DCT_Row_8x8[6][column_number];
		double f7=One_D_DCT_Row_8x8[7][column_number];
        
		//first stage of FLOWGRAPH (Chen,Fralick and Smith)
		double i0=f0+f7;
		double i1=f1+f6;
		double i2=f2+f5;
		double i3=f3+f4;
		double i4=f3-f4;
		double i5=f2-f5;
		double i6=f1-f6;
		double i7=f0-f7;
        
		//second stage of FLOWGRAPH (Chen,Fralick and Smith)
		double j0=i0+i3;
		double j1=i1+i2;
		double j2=i1-i2;
		double j3=i0-i3;
		double j4=i4;
		double j5=(i6-i5)*c4;
		double j6=(i6+i5)*c4;
		double j7=i7;
        
		//third stage of FLOWGRAPH (Chen,Fralick and Smith)
		double k0=(j0+j1)*c4;
		double k1=(j0-j1)*c4;
		double k2=(j2*c6)+(j3*c2);
		double k3=(j3*c6)-(j2*c2);
		double k4=j4+j5;
		double k5=j4-j5;
		double k6=j7-j6;
		double k7=j7+j6;
        
		//fourth stage of FLOWGRAPH; Desired DCT coefficients
		double F0=k0/2;
		double F1=(k4*c7+k7*c1)/2;
		double F2=k2/2;
		double F3=(k6*c3-k5*c5)/2;
		double F4=k1/2;
		double F5=(k5*c3+k6*c5)/2;
		double F6=k3/2;
		double F7=(k7*c7-k4*c1)/2;
        
		//DCT coefficient assignment
		out[0*stride+column_number]=(float)F0;
		out[1*stride+column_number]=(float)F1;
		out[2*stride+column_number]=(float)F2;
		out[3*stride+column_number]=(float)F3;
		out[4*stride+column_number]=(float)F4;
		out[5*stride+column_number]=(float)F5;
		out[6*stride+column_number]=(float)F6;
		out[7*stride+column_number]=(float)F7;
        
        
	} 
	
}
