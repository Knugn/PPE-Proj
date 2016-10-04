function out = clamp(in, min, max)
	out = in;
	out(find(out<min))=min;
	out(find(out>max))=max;
end
