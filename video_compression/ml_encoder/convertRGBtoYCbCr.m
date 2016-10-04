function out = convertRGBtoYCbCr(in)
% Takes in a 0-255 RGB image and converts it to YCbCr 0-255
% From: http://en.wikipedia.org/wiki/YCbCr
% http://www.equasys.de/colorconversion.html
    height = size(in, 1);
    width = size(in, 2); 
    out = zeros(height, width, 3);
    for x=1:height
        for y=1:width
            R = in(x,y,1);
            G = in(x,y,2);
            B = in(x,y,3);
            Y = 0+(0.299*R)+(0.587*G)+(0.113*B);
            Cb = 128-(0.168736*R)-(0.331264*G)+(0.5*B);
            Cr = 128+(0.5*R)-(0.418688*G)-(0.081312*B);
            out(x,y,1) = Y;
            out(x,y,2) = Cb;
            out(x,y,3) = Cr;
        end
    end 
    save('out');
end