function [DCT_8x8] = dct8x8_block(in_8x8)
% ------------------------------------------------------------------------
% FLOWGRAPH forward dct (Chen,Fralick and Smith)
% ------------------------------------------------------------------------%
% From http://www.mathworks.se/matlabcentral/fileexchange/15335-jpeg-encoder-decoder/content/JPEG%20Encoder%20Decoder/JPEG%20Encoder%20Decoder/jpeg.m

% constant cosine values will be used for both forward & inverse flowgraph DCT
c1=0.980785;
c2=0.923880;
c3=0.831470;
c4=0.707107;
c5=0.555570;
c6=0.382683;
c7=0.195090;


%---------------------------row calculation FDCT--------------------------
for row_number=1:8
    
    %sample image value initialization from input matrix
    f0=in_8x8(row_number,1);
    f1=in_8x8(row_number,2);
    f2=in_8x8(row_number,3);
    f3=in_8x8(row_number,4);
    f4=in_8x8(row_number,5);
    f5=in_8x8(row_number,6);
    f6=in_8x8(row_number,7);
    f7=in_8x8(row_number,8);

   %first stage of FLOWGRAPH (Chen,Fralick and Smith)
    i0=f0+f7;
    i1=f1+f6;
    i2=f2+f5;
    i3=f3+f4;
    i4=f3-f4;
    i5=f2-f5;
    i6=f1-f6;
    i7=f0-f7;
    
    %second stage of FLOWGRAPH (Chen,Fralick and Smith)
    j0=i0+i3;
    j1=i1+i2;
    j2=i1-i2;
    j3=i0-i3;
    j4=i4;
    j5=(i6-i5)*c4;
    j6=(i6+i5)*c4;
    j7=i7;
    
    %third stage of FLOWGRAPH (Chen,Fralick and Smith)
    k0=(j0+j1)*c4;
    k1=(j0-j1)*c4;
    k2=(j2*c6)+(j3*c2);
    k3=(j3*c6)-(j2*c2);
    k4=j4+j5;
    k5=j4-j5;
    k6=j7-j6;
    k7=j7+j6;
    
    %fourth stage of FLOWGRAPH; 1-dimensional DCT coefficients
    F0=k0/2;
    F1=(k4*c7+k7*c1)/2;
    F2=k2/2;
    F3=(k6*c3-k5*c5)/2;
    F4=k1/2;
    F5=(k5*c3+k6*c5)/2;
    F6=k3/2;
    F7=(k7*c7-k4*c1)/2;
    
    %DCT coefficient assignment
   One_D_DCT_Row_8x8(row_number,1)=F0;
   One_D_DCT_Row_8x8(row_number,2)=F1;
   One_D_DCT_Row_8x8(row_number,3)=F2;
   One_D_DCT_Row_8x8(row_number,4)=F3;
   One_D_DCT_Row_8x8(row_number,5)=F4;
   One_D_DCT_Row_8x8(row_number,6)=F5;
   One_D_DCT_Row_8x8(row_number,7)=F6;
   One_D_DCT_Row_8x8(row_number,8)=F7;

end    %end of row calculations
%---------------------------end: row calculation FDCT---------------------


%--------------------------- column calculation FDCT----------------------
for column_number=1:8   %start of column calculation
    
    %sample image value initialization
    f0=One_D_DCT_Row_8x8(1,column_number);
    f1=One_D_DCT_Row_8x8(2,column_number);
    f2=One_D_DCT_Row_8x8(3,column_number);
    f3=One_D_DCT_Row_8x8(4,column_number);
    f4=One_D_DCT_Row_8x8(5,column_number);
    f5=One_D_DCT_Row_8x8(6,column_number);
    f6=One_D_DCT_Row_8x8(7,column_number);
    f7=One_D_DCT_Row_8x8(8,column_number);
 
   %first stage of FLOWGRAPH (Chen,Fralick and Smith)
    i0=f0+f7;
    i1=f1+f6;
    i2=f2+f5;
    i3=f3+f4;
    i4=f3-f4;
    i5=f2-f5;
    i6=f1-f6;
    i7=f0-f7;
    
    %second stage of FLOWGRAPH (Chen,Fralick and Smith)
    j0=i0+i3;
    j1=i1+i2;
    j2=i1-i2;
    j3=i0-i3;
    j4=i4;
    j5=(i6-i5)*c4;
    j6=(i6+i5)*c4;
    j7=i7;
    
    %third stage of FLOWGRAPH (Chen,Fralick and Smith)
    k0=(j0+j1)*c4;
    k1=(j0-j1)*c4;
    k2=(j2*c6)+(j3*c2);
    k3=(j3*c6)-(j2*c2);
    k4=j4+j5;
    k5=j4-j5;
    k6=j7-j6;
    k7=j7+j6;
    
    %fourth stage of FLOWGRAPH; Desired DCT coefficients
    F0=k0/2;
    F1=(k4*c7+k7*c1)/2;
    F2=k2/2;
    F3=(k6*c3-k5*c5)/2;
    F4=k1/2;
    F5=(k5*c3+k6*c5)/2;
    F6=k3/2;
    F7=(k7*c7-k4*c1)/2;
    
    %DCT coefficient assignment
    DCT_8x8(1,column_number)=F0;
    DCT_8x8(2,column_number)=F1;
    DCT_8x8(3,column_number)=F2;
    DCT_8x8(4,column_number)=F3;
    DCT_8x8(5,column_number)=F4;
    DCT_8x8(6,column_number)=F5;
    DCT_8x8(7,column_number)=F6;
    DCT_8x8(8,column_number)=F7;
 
end    %end of column calculations
%---------------------------end: column calculation FDCT------------------


end    % end of function flowgraph_forward_dct