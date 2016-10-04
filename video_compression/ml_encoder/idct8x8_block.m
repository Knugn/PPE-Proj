% ------------------------------------------------------------------------
% FLOWGRAPH Inverse dct (Chen,Fralick and Smith)
% ------------------------------------------------------------------------
function [out_8x8] = idct8x8_block(DCT_8x8)

% constant cosine values will be used for both forward & inverse flowgraph DCT
c1=0.980785;
c2=0.923880;
c3=0.831470;
c4=0.707107;
c5=0.555570;
c6=0.382683;
c7=0.195090;


%---------------------------row calculation Inverse DCT-------------------
for row_number=1:8
    
    %DCT coefficient initialization
    F0=DCT_8x8(row_number,1);
    F1=DCT_8x8(row_number,2);
    F2=DCT_8x8(row_number,3);
    F3=DCT_8x8(row_number,4);
    F4=DCT_8x8(row_number,5);
    F5=DCT_8x8(row_number,6);
    F6=DCT_8x8(row_number,7);
    F7=DCT_8x8(row_number,8);

    % first stage of FLOWGRAPH (Chen,Fralick and Smith)
    k0=F0/2;
    k1=F4/2;
    k2=F2/2;
    k3=F6/2;
    k4=(F1/2*c7-F7/2*c1);
    k5=(F5/2*c3-F3/2*c5);
    k6=F5/2*c5+F3/2*c3;
    k7=F1/2*c1+F7/2*c7;
    
    % second stage of FLOWGRAPH (Chen,Fralick and Smith)
    j0=(k0+k1)*c4;
    j1=(k0-k1)*c4;
    j2=(k2*c6-k3*c2);
    j3=k2*c2+k3*c6;
    j4=k4+k5;
    j5=(k4-k5);
    j6=(k7-k6);
    j7=k7+k6;

    % third stage of FLOWGRAPH (Chen,Fralick and Smith)
    i0=j0+j3;
    i1=j1+j2;
    i2=(j1-j2);
    i3=(j0-j3);
    i4=j4;
    i5=(j6-j5)*c4;
    i6=(j5+j6)*c4;
    i7=j7;
    
    % fourth stage of FLOWGRAPH (Chen,Fralick and Smith)
    f0=i0+i7;
    f1=i1+i6;
    f2=i2+i5;
    f3=i3+i4;
    f4=(i3-i4);
    f5=(i2-i5);
    f6=(i1-i6);
    f7=(i0-i7);

    %1 dimensional sample image vale assignment only after row calculations
    One_D_IDCT_Row_8x8(row_number,1)=f0;
    One_D_IDCT_Row_8x8(row_number,2)=f1;
    One_D_IDCT_Row_8x8(row_number,3)=f2;
    One_D_IDCT_Row_8x8(row_number,4)=f3;
    One_D_IDCT_Row_8x8(row_number,5)=f4;
    One_D_IDCT_Row_8x8(row_number,6)=f5;
    One_D_IDCT_Row_8x8(row_number,7)=f6;
    One_D_IDCT_Row_8x8(row_number,8)=f7;

end
%---------------------------end: row calculation Inverse DCT--------------


%---------------------------column calculation Inverse DCT----------------
for column_number=1:8
    
    %DCT coefficient initialization
    F0=One_D_IDCT_Row_8x8(1,column_number);
    F1=One_D_IDCT_Row_8x8(2,column_number);
    F2=One_D_IDCT_Row_8x8(3,column_number);
    F3=One_D_IDCT_Row_8x8(4,column_number);
    F4=One_D_IDCT_Row_8x8(5,column_number);
    F5=One_D_IDCT_Row_8x8(6,column_number);
    F6=One_D_IDCT_Row_8x8(7,column_number);
    F7=One_D_IDCT_Row_8x8(8,column_number);

    % first stage of FLOWGRAPH (Chen,Fralick and Smith)
    k0=F0/2;
    k1=F4/2;
    k2=F2/2;
    k3=F6/2;
    k4=(F1/2*c7-F7/2*c1);
    k5=(F5/2*c3-F3/2*c5);
    k6=F5/2*c5+F3/2*c3;
    k7=F1/2*c1+F7/2*c7;
    
    % second stage of FLOWGRAPH (Chen,Fralick and Smith)
    j0=(k0+k1)*c4;
    j1=(k0-k1)*c4;
    j2=(k2*c6-k3*c2);
    j3=k2*c2+k3*c6;
    j4=k4+k5;
    j5=(k4-k5);
    j6=(k7-k6);
    j7=k7+k6;

    % third stage of FLOWGRAPH (Chen,Fralick and Smith)
    i0=j0+j3;
    i1=j1+j2;
    i2=(j1-j2);
    i3=(j0-j3);
    i4=j4;
    i5=(j6-j5)*c4;
    i6=(j5+j6)*c4;
    i7=j7;
    
    % fourth stage of FLOWGRAPH (Chen,Fralick and Smith)
    f0=i0+i7;
    f1=i1+i6;
    f2=i2+i5;
    f3=i3+i4;
    f4=(i3-i4);
    f5=(i2-i5);
    f6=(i1-i6);
    f7=(i0-i7);

    % Desired sample image values assignment only after 2 dimensional inverse transformation
    out_8x8(1,column_number)=f0;
    out_8x8(2,column_number)=f1;
    out_8x8(3,column_number)=f2;
    out_8x8(4,column_number)=f3;
    out_8x8(5,column_number)=f4;
    out_8x8(6,column_number)=f5;
    out_8x8(7,column_number)=f6;
    out_8x8(8,column_number)=f7;

end
%---------------------------end: column calculation Inverse DCT-----------


end    % end of function flowgraph_inverse_dct