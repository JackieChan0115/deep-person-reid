#include "floodfill.h"
extern "C"
{
	FloodFill floodFill;
	void process(uchar *img,int width, int height, int len, uchar *batchimgs, int *rands){
		floodFill.process(img,width, height, len, batchimgs, rands);
	}
}