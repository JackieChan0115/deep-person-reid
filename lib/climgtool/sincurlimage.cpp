#include "sincurlimage.h"
#include <cmath>
#include <ctime>


SincurlImage::SincurlImage():sin_a_factor(0.04),sin_w_factor(0.05){
	srand((unsigned)time(NULL));
	mean[0] = 0;
	mean[1] = 0;
	mean[2] = 0;
}

void SincurlImage::setparams(float _sin_a_factor, float _sin_w_factor){
    this->sin_a_factor = _sin_a_factor;
    this->sin_w_factor = _sin_w_factor;
}
int SincurlImage::sin_curl(int w, int x){
	return int(w * sin_a_factor *  sin(sin_w_factor * x));
}

//[www,.h..www]
void SincurlImage::process(uchar* imgs,int h, int w){
	int up = int(PI / sin_w_factor);
	int start = rand()%(up+1);
	int i = 0,topbase=0,mv=0,_end=0;
	for(;i<h;i++){
		mv = sin_curl(w, start+i);
		//mv 大于0向右移动， 小于0向左移动
		topbase = i*w*3;
		if(mv>0){
			for(int t = w-1;t>=mv;t--){
				imgs[topbase+t*3] = imgs[topbase+3*(t-mv)];
				imgs[topbase+t*3+1] = imgs[topbase+3*(t-mv)+1];
				imgs[topbase+t*3+2] = imgs[topbase+3*(t-mv)+2];
			}
			for(int t = 0;t<mv;t++){
				imgs[topbase+3*t] = mean[0];
				imgs[topbase+3*t+1] = mean[1];
				imgs[topbase+3*t+2] = mean[2];
			}
		}
		else{
			mv = -mv;
			_end = w-mv;
			for(int t = 0;t<_end;t++){
				imgs[topbase+3*t] = imgs[topbase+3*(mv+t)];
				imgs[topbase+3*t+1] = imgs[topbase+3*(mv+t)+1];
				imgs[topbase+3*t+2] = imgs[topbase+3*(mv+t)+2];
			}
			for(int t = _end;t<w;t++){
				imgs[topbase+3*t] = mean[0];
				imgs[topbase+3*t+1] = mean[1];
				imgs[topbase+3*t+2] = mean[2];
			}
		}
	}
}