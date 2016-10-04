function out = convertYCbCrtoRGB(in)
% Takes in a 0-255 YCbCr image and converts it to RGB 0-255
% From: http://en.wikipedia.org/wiki/YCbCr
% http://www.equasys.de/colorconversion.html
    width = size(in,1);
    height = size(in,2); 
    out = zeros(width, height, 3);
    for x=1:width
        for y=1:height
            Y = in(x,y,1);
            Cb = in(x,y,2);
            Cr = in(x,y,3);
			R = Y + 1.402*(Cr-128);
			G = Y - 0.34414*(Cb-128) - 0.71313*(Cr-128);
			B = Y + 1.772*(Cb-128);
            out(x,y,1) = R;
            out(x,y,2) = G;
            out(x,y,3) = B;
        end
    end
end