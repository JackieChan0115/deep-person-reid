#include "sincurlimage.h"
extern "C"
{
	SincurlImage sincurlImage;

	void setparams(float _sin_a_factor, float _sin_w_factor){
	    sincurlImage.setparams(_sin_a_factor, _sin_w_factor);
	}

	void process(uchar* imgs,int h, int w){
		sincurlImage.process(imgs,h, w);
	}
}