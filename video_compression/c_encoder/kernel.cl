
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

	barrier(CLK_LOCAL_MEM_FENCE); // for testing

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
	
	barrier(CLK_LOCAL_MEM_FENCE); // for testing

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
#define WINDOW_SIZE				16
#define BLOCKS_PER_WG_X			1
#define BLOCKS_PER_WG_Y			1
#define NUM_LCL_MAT_PIXELS_X	(BLOCKS_PER_WG_X*BLOCK_SIZE)
#define NUM_LCL_MAT_PIXELS_Y	(BLOCKS_PER_WG_Y*BLOCK_SIZE)
#define NUM_LCL_MAT_PIXELS		(NUM_LCL_MAT_PIXELS_X*NUM_LCL_MAT_PIXELS_Y)
#define NUM_LCL_SRC_PIXELS_X	((2 + BLOCKS_PER_WG_X)*BLOCK_SIZE)
#define NUM_LCL_SRC_PIXELS_Y	((2 + BLOCKS_PER_WG_Y)*BLOCK_SIZE)
#define NUM_LCL_SRC_PIXELS		(NUM_LCL_SRC_PIXELS_X*NUM_LCL_SRC_PIXELS_Y)
#define NUM_LCL_SAD_VALUES_X	(WINDOW_SIZE*2)
#define NUM_LCL_SAD_VALUES_Y	(WINDOW_SIZE*2)
#define NUM_LCL_SAD_VALUES		(NUM_LCL_SAD_VALUES_X*NUM_LCL_SAD_VALUES_Y)
//#define WG_SIZE 256
#define Y_WEIGHT	0.5
#define CB_WEIGHT	0.25
#define CR_WEIGHT	0.25

kernel void motion_vector_search(
	const int width,
	global float* srcY, global float* srcCb, global float* srcCr,
	global float* matY, global float* matCb, global float* matCr,
	global float* SADs)
{
	int wgs = get_local_size(0);
	int lid = get_local_id(0);

	//int nMatBlocks = get_global_size(1);
	int gSizeX = get_global_size(1);
	int gSizeY = get_global_size(2);

	int gidx = get_global_id(1);
	int gidy = get_global_id(2);

	local float lSrcY[NUM_LCL_SRC_PIXELS];
	local float lSrcCb[NUM_LCL_SRC_PIXELS];
	local float lSrcCr[NUM_LCL_SRC_PIXELS];

	local float lMatY[NUM_LCL_MAT_PIXELS];
	local float lMatCb[NUM_LCL_MAT_PIXELS];
	local float lMatCr[NUM_LCL_MAT_PIXELS];
	
	int baseMatPixY = gidy * BLOCK_SIZE + WINDOW_SIZE;
	int baseMatPixX = gidx * BLOCK_SIZE + WINDOW_SIZE;

	int baseSrcPixY = baseMatPixY - WINDOW_SIZE;
	int baseSrcPixX = baseMatPixX - WINDOW_SIZE;

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

	for (int lSrcPixIdx = lid; lSrcPixIdx < NUM_LCL_SRC_PIXELS; lSrcPixIdx += wgs)
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
	if (lid >= NUM_LCL_MAT_PIXELS_X * WINDOW_SIZE * 2)
		return;

	local float lSADs[NUM_LCL_SAD_VALUES];
	local float foo[NUM_LCL_SAD_VALUES_X][BLOCK_SIZE];

	int lSrcPixXOffset = lid % NUM_LCL_MAT_PIXELS_X;
	int lSrcPixXBaseIncr = wgs / NUM_LCL_MAT_PIXELS_X;


	for (int lSrcPixYBase = 0; lSrcPixYBase < WINDOW_SIZE * 2; lSrcPixYBase++) {
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int lSrcPixXBase = lid / NUM_LCL_MAT_PIXELS_X; lSrcPixXBase < NUM_LCL_SAD_VALUES_X; 
			lSrcPixXBase += lSrcPixXBaseIncr) 
		{
			int lSrcPixX = lSrcPixXBase + lSrcPixXOffset;
			int lMatPixX = lSrcPixXOffset;
			float colSAD = 0;
			for (int lMatPixY = 0; lMatPixY < BLOCK_SIZE; lMatPixY++) {
				int lSrcPixY = lSrcPixYBase + lMatPixY;
				int lSrcPixIdx = lSrcPixY * NUM_LCL_SRC_PIXELS_X + lSrcPixX;
				int lMatPixIdx = lMatPixY * NUM_LCL_MAT_PIXELS_X + lMatPixX;
				float diffY = fabs(lMatY[lMatPixIdx] - lSrcY[lSrcPixIdx]);
				float diffCb = fabs(lMatCb[lMatPixIdx] - lSrcCb[lSrcPixIdx]);
				float diffCr = fabs(lMatCr[lMatPixIdx] - lSrcCr[lSrcPixIdx]);

				float diffTotal = Y_WEIGHT*diffY + CB_WEIGHT*diffCb + CR_WEIGHT*diffCr;
				colSAD += diffTotal;
			}
			
			foo[lSrcPixXBase][lSrcPixXOffset] = colSAD;
			
			for (int nSummers = BLOCK_SIZE / 2; 1 <= nSummers; nSummers /= 2) {
				barrier(CLK_LOCAL_MEM_FENCE);
				if (lSrcPixXOffset < nSummers)
					foo[lSrcPixXBase][lSrcPixXOffset] += foo[lSrcPixXBase][lSrcPixXOffset + nSummers];
			}
			if (lSrcPixXOffset == 0) {
				lSADs[lSrcPixXBase*NUM_LCL_SAD_VALUES_X + lSrcPixYBase] = foo[lSrcPixXBase][lSrcPixXOffset];
			}
			
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Assuming one work group operating on 1 match tile
	int gSADBaseIdx = (gidy*gSizeX + gidx)*NUM_LCL_SAD_VALUES;

	for (int lSADIdx = lid; lSADIdx < NUM_LCL_SAD_VALUES; lSADIdx += wgs)
	{
		int gSADIdx = gSADBaseIdx + lSADIdx;
		SADs[gSADIdx] = lSADs[lSADIdx];
	}
	
}
