
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

}

#define BLOCK_SIZE				16
#define WINDOWS_SIZE			16
#define BLOCKS_PER_WG_X			2
#define BLOCKS_PER_WG_Y			1
#define NUM_LCL_MAT_PIXELS_X	(BLOCKS_PER_WG_X*BLOCK_SIZE)
#define NUM_LCL_MAT_PIXELS_Y	(BLOCKS_PER_WG_Y*BLOCK_SIZE)
#define NUM_LCL_MAT_PIXELS		(NUM_LCL_MAT_PIXELS_X*NUM_LCL_MAT_PIXELS_Y)
#define NUM_LCL_SRC_PIXELS_X	((2 + BLOCKS_PER_WG_X)*BLOCK_SIZE)
#define NUM_LCL_SRC_PIXELS_Y	((2 + BLOCKS_PER_WG_Y)*BLOCK_SIZE)
#define NUM_LCL_SRC_PIXELS		(NUM_LCL_SRC_PIXELS_X*NUM_LCL_SRC_PIXELS_Y)
//#define WG_SIZE 256
#define Y_WEIGHT	0.5
#define CB_WEIGHT	0.25
#define CR_WEIGHT	0.25

kernel void motion_vector_search_smart(
	int width,
	global float* srcY, global float* srcCb, global float* srcCr,
	global float* matY, global float* matCb, global float* matCr,
	global float* colSADs)
{
	int gSizeX = get_global_size(0);
	int gSizeY = get_global_size(1);

	int gidx = get_global_id(0);
	int gidy = get_global_id(1);

	int wgs = get_local_size(2);
	int lid = get_local_id(2);

	local float lSrcY[NUM_LCL_SRC_PIXELS];
	local float lSrcCb[NUM_LCL_SRC_PIXELS];
	local float lSrcCr[NUM_LCL_SRC_PIXELS];

	local float lMatY[NUM_LCL_MAT_PIXELS];
	local float lMatCb[NUM_LCL_MAT_PIXELS];
	local float lMatCr[NUM_LCL_MAT_PIXELS];

	int baseMatPixY = gidy * BLOCK_SIZE + WINDOWS_SIZE;
	int baseMatPixX = gidx * BLOCK_SIZE + WINDOWS_SIZE;

	for (int lMatPixIdx = lid; lMatPixIdx < NUM_LCL_MAT_PIXELS; lMatPixIdx += wgs)
	{
		int lMatPixY = lMatPixIdx / NUM_LCL_MAT_PIXELS_X;
		int lMatPixX = lMatPixIdx % NUM_LCL_MAT_PIXELS_X;
		int gMatPixY = baseMatPixY + lMatPixY;
		int gMatPixX = baseMatPixX + lMatPixX;
		int gMatPixIdx = gMatPixY*width + gMatPixX;
		lMatY[lMatPixIdx] = matY[gMatPixIdx];
		lMatCb[lMatPixIdx] = matCb[gMatPixIdx];
		lMatCr[lMatPixIdx] = matCr[gMatPixIdx];
	}

	int baseSrcPixY = baseMatPixY - WINDOWS_SIZE;
	int baseSrcPixX = baseMatPixX - WINDOWS_SIZE;

	for (int lSrcPixIdx = lid; lSrcPixIdx < NUM_LCL_MAT_PIXELS; lSrcPixIdx += wgs)
	{
		int lSrcPixY = lSrcPixIdx / NUM_LCL_SRC_PIXELS_X;
		int lSrcPixX = lSrcPixIdx % NUM_LCL_SRC_PIXELS_X;
		int gSrcPixY = baseSrcPixY + lSrcPixY;
		int gSrcPixX = baseSrcPixX + lSrcPixX;
		int gSrcPixIdx = gSrcPixY*width + gSrcPixX;
		lSrcY[lSrcPixIdx] = srcY[gSrcPixIdx];
		lSrcCb[lSrcPixIdx] = srcCb[gSrcPixIdx];
		lSrcCr[lSrcPixIdx] = srcCr[gSrcPixIdx];
	}
	
	// We will only work on one source row at a time
	// Discard superflous work items (unlikely)
	if (lid >= NUM_LCL_MAT_PIXELS_X * WINDOWS_SIZE * 2)
		return;

	int lSrcPixXBase = lid / NUM_LCL_MAT_PIXELS_X;
	int lSrcPixXOffset = lid / NUM_LCL_MAT_PIXELS_X;
	int lMatPixX = lSrcPixXBase + lSrcPixXOffset;

	// TODO: need another loop layer in case we have too few work items
	for (int lSrcPixYBase = 0; lSrcPixYBase < WINDOWS_SIZE * 2; lSrcPixYBase++) {
		float colSAD = 0;
		for (int lMatPixY = 0; lMatPixY < BLOCK_SIZE; lMatPixY++) {
			int lSrcPixY = lSrcPixYBase + lMatPixY;
			int lSrcPixIdx = lSrcPixY * NUM_LCL_SRC_PIXELS_X + lMatPixX;
			int lMatPixIdx = lMatPixY * NUM_LCL_MAT_PIXELS_X + lMatPixX;
			float diffY = fabs(lMatY[lMatPixIdx] - lSrcY[lSrcPixIdx]);
			float diffCb = fabs(lMatCb[lMatPixIdx] - lSrcCb[lSrcPixIdx]);
			float diffCr = fabs(lMatCr[lMatPixIdx] - lSrcCr[lSrcPixIdx]);
			float diffTotal = Y_WEIGHT*diffY + CB_WEIGHT*diffCb + CR_WEIGHT*diffCr;
			colSAD += diffTotal;
		}
		// TODO: write to colSADs[]
	}
		
}

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

kernel void motion_vector_search_stupid(
	int width,
	int numTarTilesX,
	global float* srcY, global float* srcCb, global float* srcCr,
	global float* tarY, global float* tarCb, global float* tarCr,
	global float* rowSADs)
{
	int numTarTiles = get_global_size(0);
	int numSrcTiles = get_global_size(1);
	int tileSize = TILE_WIDTH * TILE_HEIGHT;
		
	int tarTileIdx = get_global_id(0);
	int srcTileIdx = get_global_id(1);
	int tileOffset = get_global_id(2);

	int tileY = tileOffset / TILE_WIDTH;
	int tileX = tileOffset % TILE_WIDTH;
	
	int tarTileX = tarTileIdx % numTarTilesX;
	int tarTileY = tarTileIdx / numTarTilesX;
	int tarPixY = (tarTileY + 1) * TILE_HEIGHT + tileY;
	int tarPixX = (tarTileX + 1) * TILE_WIDTH + tileX;
	int tarIdx = tarPixY * width + tarPixX;
	
	int srcOffsetY = -16 + srcTileIdx / 32;
	int srcOffsetX = -16 + srcTileIdx % 32;
	int srcPixY = tarPixY + srcOffsetY;
	int srcPixX = tarPixX + srcOffsetX;
	int srcIdx = srcPixY * width + srcPixX;
	
	float Y_weight = 0.5;
	float Cr_weight = 0.25;
	float Cb_weight = 0.25;
	
	float diffY = fabs(tarY[tarIdx] - srcY[srcIdx]);
	float diffCb = fabs(tarCb[tarIdx] - srcCb[srcIdx]);
	float diffCr = fabs(tarCr[tarIdx] - srcCr[srcIdx]);
	float diffTotal = Y_weight*diffY + Cb_weight*diffCb + Cr_weight*diffCr;
	
	local float wgSADs[TILE_WIDTH];
	wgSADs[tileX] = diffTotal;
	if (tileX >= 8)
		return;
	wgSADs[tileX] = wgSADs[tileX + 8];
	if (tileX >= 4)
		return;
	wgSADs[tileX] = wgSADs[tileX + 4];
	if (tileX >= 2)
		return;
	wgSADs[tileX] = wgSADs[tileX + 2];
	if (tileX >= 1)
		return;
	wgSADs[tileX] = wgSADs[tileX + 1];

	int rowSADidx = tileY*numTarTiles*numSrcTiles + tarTileIdx*numSrcTiles + srcTileIdx;
	rowSADs[rowSADidx] = wgSADs[tileX];
	
}

kernel void rowSADsSum(global float* rowSADs, global float* SADs) 
{
	int size = get_global_size(0);
	int SADidx = get_global_id(0);
	float sum = 0;
	for (int row = 0; row < 16; row++) {
		sum += rowSADs[row*size + SADidx];
	}
	SADs[SADidx] = sum;
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
