#ifndef CMDETECT2_H
#define CMDETECT2_H

#include <iostream>
#include <string>
#include <cv.hpp>
#include <highgui.h>

#define MAX_PIXELS  1000

using namespace std;
using namespace cv;

typedef struct {
    int id;
    int ch1;
    int ch2;
    int m;
    int b1;
    int b2;
    int t_min1;
    int t_min2;
    int t_max1;
    int t_max2;
} T_Classifier;

class CMDetect2
{
    public:
        CMDetect2();
        virtual ~CMDetect2();

        void detect(Mat& img, vector<Point>& centers,vector<int>& radius);
    private:
        int loadClassifier();
        void getCh(int clsId,int& ch1,int& ch2);
        bool isProbeFit(uchar** data);
        inline void setDataPtr(uchar** dataPtr, uchar* rowUp,uchar*rowDown,int leftShift, int rightShift) {
            dataPtr[0]=rowUp+rightShift;
            dataPtr[1]=rowDown+leftShift;
            dataPtr[2]=rowUp+leftShift;
            dataPtr[3]=rowDown+rightShift;
            //dataPtr[0]=rowDown+rightShift;
            //dataPtr[1]=rowUp+leftShift;
            //dataPtr[2]=rowDown+leftShift;
            //dataPtr[3]=rowUp+rightShift;
        }
    private:
        vector<T_Classifier> mClass;
        Point mCandPixel[MAX_PIXELS];
        int mCandScore[MAX_PIXELS];
        int mCandSize;
        int mClassShift;
};

#endif // CMDETECT2_H
