
#include "floodfill.h"
#include <cstring>
#include <ctime>

FloodFill::FloodFill(){
	diff[0] = 5;
	diff[1] = 5;
	diff[2] = 5;
	srand((unsigned)time(NULL));
}
FloodFill::~FloodFill(){

}

void FloodFill::execute_fill(uchar *img,uchar *mask, QNode *node, QNode &head,int width,int height){
    int base = (node->y * width + node->x)*3;
    uchar _r = img[base], _g = img[base+1], _b = img[base+2];
    int dis_r = int(node->old_r) - _r, dis_g = int(node->old_g) - _g, dis_b = int(node->old_b) - _b;
    int a_dis_r = dis_r>0?dis_r:-dis_r, a_dis_g = dis_g>0?dis_g:-dis_g, a_dis_b = dis_b>0?dis_b:-dis_b;
    if(a_dis_r<diff[0]&&a_dis_g<diff[1]&&a_dis_b<diff[2]){
        uchar this_r = 0, this_g=0, this_b=0;

        if(node->new_r<dis_r) this_r = 0;
        else if(-dis_r > 255 - node->new_r) this_r = 255;
        else this_r = node->new_r - dis_r;

        if(node->new_g<dis_g) this_g = 0;
        else if(-dis_g > 255 - node->new_g) this_g = 255;
        else this_g = node->new_g - dis_g;

        if(node->new_b<dis_b) this_b = 0;
        else if(-dis_b>255 - node->new_b) this_b = 255;
        else this_b = node->new_b - dis_b;
        
        img[base] = this_r;
        img[base+1] = this_g;
        img[base+2] = this_b;
        mask[node->y*width+node->x]=1;
        if(node->y+1<height && mask[(node->y+1)*width+node->x]==0){
            QNode *temp = (QNode*)malloc(sizeof(QNode));
            temp->setPoint(node->x,node->y+1);
            temp->setNew(this_r, this_g,this_b);
            temp->setOld(_r,_g,_b);
            temp->next = head.next;
            head.next = temp;
        }
        if(node->y-1>=0 && mask[(node->y-1)*width+node->x]==0){
            QNode *temp = (QNode*)malloc(sizeof(QNode));
            temp->setPoint(node->x,node->y-1);
            temp->setNew(this_r, this_g,this_b);
            temp->setOld(_r,_g,_b);
            temp->next = head.next;
            head.next = temp;
        }
        if(node->x+1<width && mask[node->y*width+node->x+1]==0){
            QNode *temp = (QNode*)malloc(sizeof(QNode));
            temp->setPoint(node->x+1,node->y);
            temp->setNew(this_r, this_g,this_b);
            temp->setOld(_r,_g,_b);
            temp->next = head.next;
            head.next = temp;
        }
        if(node->x-1>=0 && mask[node->y*width+node->x-1]==0){
            QNode *temp = (QNode*)malloc(sizeof(QNode));
            temp->setPoint(node->x-1,node->y);
            temp->setNew(this_r, this_g,this_b);
            temp->setOld(_r,_g,_b);
            temp->next = head.next;
            head.next = temp;
        }
    }
}

void FloodFill::process(uchar *img,int width, int height, int len, uchar *batchimgs, int *rands){
    int offset = width*height*3;
    QNode head;
    uchar* mask = (uchar*)malloc(sizeof(uchar)*width*height);
    memset(mask,0,sizeof(uchar)*width*height);
    int steps = height/len;
    for(int i = 0;i<len;i++){
        int x = rand()%width;
        int y = rand()%steps+i*steps;
        QNode *temp = (QNode*)malloc(sizeof(QNode));
        int base = (y*width+x)*3;
        int start = offset*rands[i]+base;
        temp->setPoint(x,y);
        temp->setOld(img[base],img[base+1],img[base+2]);
        temp->setNew(batchimgs[start],batchimgs[start+1],batchimgs[start+2]);
        head.next = temp;
        temp->next = NULL;
        while(head.next!=NULL){
            temp = head.next;
            head.next = head.next->next;
            if(mask[temp->y*width+temp->x]==0){
                execute_fill(img, mask, temp, head,width,height);
            }
            free(temp);
            temp = NULL;
        }
    }
    free(mask);
}

