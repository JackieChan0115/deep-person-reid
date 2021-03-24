#ifndef PY_C_SINCURL_IMAGE_EXCUTE_H
#define PY_C_SINCURL_IMAGE_EXCUTE_H

#include <iostream>
typedef unsigned char uchar;
const double PI = 3.1415926;

class SincurlImage{
private:
	//float _probability = 1;
	float sin_a_factor;
	float sin_w_factor;
	int mean[3];
public:
	SincurlImage();
	void setparams(float _sin_a_factor, float _sin_w_factor);
	int sin_curl(int w, int x);
	void process(uchar* imgs,int h, int w);
};

#endif