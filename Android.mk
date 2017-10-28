LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
include ../opencv/build/OpenCV.mk
LOCAL_MODULE    := cmdetector

LOCAL_SRC_FILES := \
    CMDetect2.cpp  \
    main.cpp

LOCAL_C_INCLUDES := \
    ../opencv/opencv-3.1.0/include \
    ../opencv/opencv-3.1.0/include/opencv \
    ../opencv/opencv-3.1.0/include/opencv2 \
    ../opencv/opencv-3.1.0/modules/core/include \
    ../opencv/opencv-3.1.0/modules/imgproc/include \
    ../opencv/opencv-3.1.0/modules/photo/include \
    ../opencv/opencv-3.1.0/modules/video/include \
    ../opencv/opencv-3.1.0/modules/objdetect/include \
    ../opencv/opencv-3.1.0/modules/highgui/include \
    ../opencv/opencv-3.1.0/modules/imgcodecs/include \
    ../opencv/opencv-3.1.0/modules/videoio/include \
    ../opencv/opencv-3.1.0/modules/features2d/include \
    ../opencv/opencv-3.1.0/modules/flann/include \
    ../opencv/opencv-3.1.0/modules/calib3d/include \
    
    
LOCAL_LDLIBS +=  -llog -ldl

include $(BUILD_SHARED_LIBRARY)

