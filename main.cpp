#include <iostream>
#include <string>
#include <cv.hpp>
#include <jni.h>
#include <android/log.h>
#include "CMDetect2.h"
#define TAG "CVLOG"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using namespace std;
using namespace cv;

CMDetect2 g_CMDetect2;
Mat yuvImg;
Mat g_SrcImg;

extern "C"
{
    int cmDetect(uchar* yuv,int width, int height, float* pos_x,float* pos_y) {
        static long sProcessId = 0;
        static long tickBeg,tickEnd,freq=getTickFrequency();
        tickBeg=getTickCount();
        LOGE("cmDetect %ld begin width-height:%d %d",sProcessId, width,height);
        if(!sProcessId) {
            yuvImg.create(height*3/2,width,CV_8UC1);
            g_SrcImg.create(Size(width,height),CV_8UC3);
        }
        ++sProcessId;
        *pos_x=*pos_y=-1;
        memcpy(yuvImg.data, yuv, width*height*3/2);
        cvtColor(yuvImg,g_SrcImg,CV_YUV2BGR_NV12);
        tickEnd = getTickCount();
        LOGE("cmDetect init time spend: %f",(double)(tickEnd-tickBeg)/freq);
        tickBeg=tickEnd;
        vector<Point> centers;
        vector<int> radius;
        g_CMDetect2.detect(g_SrcImg,centers,radius);
        tickEnd = getTickCount();
        LOGE("cmDetect detect time spend: %f",(double)(tickEnd-tickBeg)/freq);
        tickBeg=tickEnd;
        if(centers.size()>0) {
            *pos_x=centers[0].x;
            *pos_y=480-centers[0].y;
        }
        tickEnd = getTickCount();
        LOGE("cmDetect end time spend: %f pos_x=%f, pos_y=%f", (double)(tickEnd-tickBeg)/freq,*pos_x, *pos_y);
        return 0;
    }
    
}

