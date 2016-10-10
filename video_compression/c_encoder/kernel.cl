
kernel void convert_RGB_to_YCbCr(
	global float* R, global float* G, global float* B,
	global float* Y, global float* Cb, global float* Cr) 
{
	int size = get_global_size(0);
	int idx = get_global_id(0);
	if (idx >= size) return;

	float r = R[idx];
	float g = G[idx];
	float b = B[idx];

	float y = 0 + ((float)0.299*r) + ((float)0.587*g) + ((float)0.113*b);
	float cb = 128 - ((float)0.168736*r) - ((float)0.331264*g) + ((float)0.5*b);
	float cr = 128 + ((float)0.5*r) - ((float)0.418688*g) - ((float)0.081312*b);

	Y[idx] = y;
	Cb[idx] = cb;
	Cr[idx] = cr;
}

kernel void blur(global float* in, global float* out)
{
	float a = 0.25f;
	float b = 0.5f;
	float c = 0.25f;

	local float tile_in[1024];
	local float tile_mid[1024];

	int width = get_global_size(0);
	int height = get_global_size(1);

	int local_width = get_local_size(0);
	int local_height = get_local_size(1);

	int x = get_global_id(0);
	int y = get_global_id(1);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);

	tile_in[local_y*local_width + local_x] = in[y*width + x];

	/*
	if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
		out[y*width + x] = in[y*width + x];
		return;
	}*/
	if (x >= width || y >= height)
		return;
	if (local_y == 0 || local_y >= local_height - 1) {
		out[y*width + x] = tile_in[local_y*local_width + local_x];
		return;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	tile_mid[local_y*local_width + local_x] =
		a*tile_in[(local_y - 1)*local_width + local_x] +
		b*tile_in[(local_y + 0)*local_width + local_x] +
		c*tile_in[(local_y + 1)*local_width + local_x];
	
	if (local_x == 0 || local_x >= local_width - 1) {
		out[y*width + x] = tile_mid[local_y*local_width + local_x];
		return;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	out[y*width + x] =
		a*tile_mid[(local_y)*local_width + local_x - 1] +
		b*tile_mid[(local_y)*local_width + local_x + 0] +
		c*tile_mid[(local_y)*local_width + local_x + 1];
	/*
	float result =
		a * (a * in[(y - 1)*width + x - 1] + b* in[(y + 0)*width + x - 1] + c * in[(y + 1)*width + x - 1]) +
		b * (a * in[(y - 1)*width + x + 0] + b* in[(y + 0)*width + x + 0] + c * in[(y + 1)*width + x + 0]) +
		c * (a * in[(y - 1)*width + x + 1] + b* in[(y + 0)*width + x + 1] + c * in[(y + 1)*width + x + 1]);
	*/
	/*
	float result =
		a*(a * in[(y - 1)*width + x - 1] + b * in[(y - 1)*width + x] + c * in[(y - 1)*width + x + 1]) +
		b*(a * in[(y + 0)*width + x - 1] + b * in[(y + 0)*width + x] + c * in[(y + 0)*width + x + 1]) +
		c*(a * in[(y + 1)*width + x - 1] + b * in[(y + 1)*width + x] + c * in[(y + 1)*width + x + 1]);
		*/
	//out[y*width + x] = result;
}

/*
kernel void update(global float *in, global float *out) {
	int WIDTH = get_global_size(0);
	int HEIGHT = get_global_size(1);
	// Don't do anything if we are on the edge.
	if (get_global_id(0) == 0 || get_global_id(1) == 0)
		return;
	if (get_global_id(0) == (WIDTH-1) || get_global_id(1) == (HEIGHT-1))
		return;
	int y = get_global_id(1);	
	int x = get_global_id(0); 
	// Load the data
	float a = in[WIDTH*(y-1)+(x)];
	float b = in[WIDTH*(y)+(x-1)];
	float c = in[WIDTH*(y+1)+(x)];
	float d = in[WIDTH*(y)+(x+1)];
	float e = in[WIDTH*y+x];
	// Do the computation and write back the results
	out[WIDTH*y+x] = (0.1*a+0.2*b+0.2*c+0.1*d+0.4*e);
}


kernel void simple_range(global float *data, int total_size, global float *range) {
	// Only one thread does this to avoid synchronization
	if (get_global_id(0) == 0) {
		float min, max;
		min = max = 0.0f;

		for (int i=0; i<total_size; i++) {
			if (data[i] < min)
				min = data[i];
			else if (data[i] > max)
				max = data[i];
		}
		
		range[0] = min;
		range[1] = max;
	}
}


kernel void range(global float *data, int total_size, global float *range) {
	float max, min;

	// Find out which items this work-item processes
	int size_per_workitem = total_size/get_global_size(0);
	int start = size_per_workitem*get_global_id(0);
	int stop = start+size_per_workitem;
	
	// Finds the min/max for our chunk of the data
	min = max = 0.0f;
	for (int i=start; i<stop; i++) {
		if (data[i] < min)
			min = data[i];
		else if (data[i] > max)
			max = data[i];
	}
	
	// Write the min and max back to the range we will return to the host
	range[get_global_id(0)*2] = min;
	range[get_global_id(0)*2+1] = max;
}

kernel void range_coalesced(global float *data, int total_size, global float *range) {
	float max, min;

	// Work-items in a work-group process neighboring elements on each iteration
	int size_per_workgroup = total_size/get_num_groups(0);
	int start = size_per_workgroup*get_group_id(0)+get_local_id(0);
	int stop = start+size_per_workgroup;
	
	// Each work-item finds the min/max for its chunk of the data
	min = max = 0.0f;
	for (int i=start; i<stop; i+=get_local_size(0)) {
		if (data[i] < min)
			min = data[i];
		else if (data[i] > max)
			max = data[i];
	}
	
	// Write the min and max back to the range we will return to the host
	range[get_global_id(0)*2] = min;
	range[get_global_id(0)*2+1] = max;
}*/
