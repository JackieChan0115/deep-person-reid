#ifndef PY_C_FLOODFILL_IMAGE_EXCUTE_H
#define PY_C_FLOODFILL_IMAGE_EXCUTE_H

#include <iostream>
typedef unsigned char uchar;

struct QNode{
    int x,y;
    uchar new_r,new_g,new_b;
    uchar old_r,old_g,old_b;
    QNode *next;
    QNode(){
        next = NULL;
        x = 0;
        y = 0;
        new_r = 0;
        new_g = 0;
        new_b = 0;
        old_r = 0;
        old_g = 0;
        old_b = 0;
    }
    void setPoint(int _x, int _y){
        x = _x;
        y = _y;
    }
    void setOld(uchar r, uchar g, uchar b){
        old_r = r;
        old_g = g;
        old_b = b;
    }
    void setNew(uchar r, uchar g, uchar b){
        new_r = r;
        new_g = g;
        new_b = b;
    }
    void printPoint(){
        printf("(x,y)=(%d,%d)\\n",x,y);
    }
    void printOld(){
        printf("old(r,g,b)=(%d,%d,%d)\\n",old_r,old_g,old_b);
    }
    void printNew(){
        printf("new(r,g,b)=(%d,%d,%d)\\n",new_r,new_g,new_b);
    }
 };
class FloodFill
{
private:
	int diff[3];
	void execute_fill(uchar *img,uchar *mask, QNode *node, QNode &head,int width,int height);
public:
	FloodFill();
	void process(uchar *img,int width, int height, int len, uchar *batchimgs, int *rands);

	~FloodFill();
};

#endif