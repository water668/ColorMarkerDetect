#include "CMDetect2.h"
#include <fstream>
#include <jni.h>
#include <android/log.h>
#define TAG "CVLOG"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define CASCADE_LENGTH 15
#define PROBE_HALF_SPACING 8
#define SCAN_STEP 1
int IMG_WIDTH;
int IMG_HEIGHT;

#define PT_DISTANCE_THRESHOLD2 8
#define PT_CLUSTER_THRESHOLD 4
//#define CLASSIFIER_FILE "Classifier.txt"
#define CLASSIFIER_FILE "/sdcard/cascade.txt"

using namespace std;
using namespace cv;

CMDetect2::CMDetect2()
{
    loadClassifier();
}

CMDetect2::~CMDetect2()
{
}

void CMDetect2::detect(Mat& img, vector<Point>& centers,vector<int>& radius) {
    LOGE("cmDetect4 detect width=%d height=%d",img.size().width, img.size().height);
    static long tickBeg, tickEnd, freq=getTickFrequency();
    tickBeg=getTickCount();
    IMG_WIDTH=img.size().width;
    IMG_HEIGHT=img.size().height;
    centers.clear();
    radius.clear();
    mCandSize=0;
    //1, scan img
    uchar* dataPtr[4];
    for(int row = PROBE_HALF_SPACING; row<IMG_HEIGHT-PROBE_HALF_SPACING; row+=SCAN_STEP) {
        uchar* rowUp=img.ptr<uchar>(row-PROBE_HALF_SPACING);
        uchar* rowDown=img.ptr<uchar>(row+PROBE_HALF_SPACING);
        for(int col = PROBE_HALF_SPACING; col <IMG_WIDTH-PROBE_HALF_SPACING; col+=SCAN_STEP) {
            int leftShift=3*(col-PROBE_HALF_SPACING);
            int rightShift=3*(col+PROBE_HALF_SPACING);
            setDataPtr(dataPtr,rowUp,rowDown,leftShift,rightShift);
            if(isProbeFit(dataPtr)&&mCandSize<MAX_PIXELS){
                mCandPixel[mCandSize]=Point(col,row);
                ++mCandSize;
            }
        }
    }
    tickEnd = getTickCount();
    LOGE("cmDetect scan time spend: %f",(double)(tickEnd-tickBeg)/freq);
    tickBeg=tickEnd;
    // cluster
    int maxScoreId = -1;
    int maxScore=0;
    for(int i=0; i<mCandSize; ++i) {
        mCandScore[i]=0;
        int clu[3],cld[3],cru[3],crd[3];
        for(int j=i+1; j<mCandSize; ++j) {
            if(abs(mCandPixel[j].x-mCandPixel[i].x)<PT_DISTANCE_THRESHOLD2 && abs(mCandPixel[j].y-mCandPixel[i].y)<PT_DISTANCE_THRESHOLD2) {
                ++mCandScore[i];
            }
        }
        if(mCandScore[i]>maxScore) {
            maxScore=mCandScore[i];
            maxScoreId= i;
        }
    }
    tickEnd = getTickCount();
    LOGE("cmDetect cluster time spend: %f, mCandSize=%d, maxScore=%d, x=%d,y=%d",(double)(tickEnd-tickBeg)/freq, mCandSize, maxScore,mCandPixel[maxScoreId].x,mCandPixel[maxScoreId].y);
    tickBeg=tickEnd;
    if(maxScore>PT_CLUSTER_THRESHOLD) {
        //find center
        int cx=0,cy=0,count=0;
        int shift=20;
        int upBound=mCandPixel[maxScoreId].y-shift,downBound=mCandPixel[maxScoreId].y+shift;
        int leftBound=mCandPixel[maxScoreId].x-shift,rightBound=mCandPixel[maxScoreId].x+shift;
        for(int idx=0;idx<mCandSize; ++idx) {
            if(mCandScore[idx]>PT_CLUSTER_THRESHOLD) {
                if (mCandPixel[idx].y>upBound&&mCandPixel[idx].y<downBound&&mCandPixel[idx].x>leftBound&&mCandPixel[idx].x<rightBound) {
                    cx+=mCandPixel[idx].x; cy+=mCandPixel[idx].y;++count;
                }
            }
        }
        cx=cx/count; cy=cy/count;
        tickEnd = getTickCount();
        LOGE("cmDetect find center time spend: %f cx=%d  cy=%d",(double)(tickEnd-tickBeg)/freq,cx,cy);
        tickBeg=tickEnd;
        // calc radius
        int halfSpa=PROBE_HALF_SPACING;
        int okHalfSpa=halfSpa;
        while(halfSpa<okHalfSpa+3) {
            halfSpa+=1;
            if (cy-halfSpa<0||cy+halfSpa>=IMG_HEIGHT||cx-halfSpa<0||cx+halfSpa>=IMG_WIDTH)
                break;
            uchar* rowUp=img.ptr<uchar>(cy-halfSpa);
            uchar* rowDown=img.ptr<uchar>(cy+halfSpa);
            int leftShift=3*(cx-halfSpa);
            int rightShift=3*(cx+halfSpa);
            setDataPtr(dataPtr,rowUp,rowDown,leftShift,rightShift);
            if(isProbeFit(dataPtr))
                okHalfSpa=halfSpa;
        }
        if(okHalfSpa>PROBE_HALF_SPACING*1) {
            centers.push_back(Point(cx,cy));
            radius.push_back(1.414*okHalfSpa);
        }
        tickEnd = getTickCount();
        LOGE("cmDetect calc radius time spend: %f okHalfSpa=%d",(double)(tickEnd-tickBeg)/freq, okHalfSpa);
    }
}

int CMDetect2::loadClassifier() {
    LOGE("cmDetect loadClassifier begin file=%s", CLASSIFIER_FILE);
    fstream fs;
    fs.open(CLASSIFIER_FILE,ios_base::in);
    
    LOGE("cmDetect loadClassifier is_open()=%d",fs.is_open());
    int count=0;
    fs>>count>>mClassShift;
    T_Classifier cls;
    for(int i=0; i<count; ++i) {
        
        fs>>cls.id>>cls.m>>cls.b1>>cls.b2;
        getCh(cls.id,cls.ch1,cls.ch2);
        mClass.push_back(cls);
        LOGE("cmDetect i=%d id=%d", i, cls.id);
    }
    fs.close();
    LOGE("cmDetect loadClassifier count=%d mClassShift=%d", count, mClassShift);
}

void CMDetect2::getCh(int clsId,int& ch1,int& ch2) {
switch(clsId) {
    case 0:ch1=0;ch2=3;break;
    case 1:ch1=0;ch2=6;break;
    case 2:ch1=0;ch2=9;break;
    case 3:ch1=1;ch2=4;break;
    case 4:ch1=1;ch2=7;break;
    case 5:ch1=1;ch2=10;break;
    case 6:ch1=2;ch2=5;break;
    case 7:ch1=2;ch2=8;break;
    case 8:ch1=2; ch2=11;break;
    case 9:ch1=3;ch2=6;break;
    case 10:ch1=3;ch2=9;break;
    case 11:ch1=4;ch2=7;break;
    case 12:ch1=4;ch2=10;break;
    case 13:ch1=5;ch2=8;break;
    case 14:ch1=5;ch2=11;break;
    case 15:ch1=6;ch2=9;break;
    case 16:ch1=7;ch2=10;break;
    case 17:ch1=8;ch2=11;break;
}
}

bool CMDetect2::isProbeFit(uchar** data) {// assume channel seq is bgr
    if(data[0][1]<50||data[1][0]<60||data[1][1]<60||data[1][2]<60||data[2][2]<60)
        return false;
    for(int cIdx=0;cIdx<CASCADE_LENGTH;++cIdx) {
        T_Classifier& cls=mClass[cIdx];
        int value1=data[cls.ch1/3][2-cls.ch1%3];
        int value2=data[cls.ch2/3][2-cls.ch1%3];
        int dd=(value2<<mClassShift)-value1*cls.m;
        if (dd<cls.b1&&dd>cls.b2) {
            if (cIdx==CASCADE_LENGTH-1)
                return true;
        }else break;
    }
    return false;
}
