// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_DETECTOR_OPENCV_HPP
#define FACE_DETECTOR_OPENCV_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
using namespace cv::dnn;
//using namespace cv;

class FaceDetectorOpencv
{
public:
 FaceDetectorOpencv(cv::CommandLineParser parser) {load(parser);}; 
  
  FaceDetectorOpencv() {};
  ~FaceDetectorOpencv() {};
  
  void detectFaceOpenCVDNN(const cv::Mat& frameOpenCVDNN,  std::vector<cv::Rect> &faces,
    Net net);

  void
    load(cv::CommandLineParser parser);
 
  void
  process
    (
    cv::Mat frameOpenCVDNN,
    std::vector<cv::Rect> &faces
    );
  
  const size_t inWidth = 300;
  const size_t inHeight = 300;
  const double inScaleFactor = 1.0;
  const float confidenceThreshold = 0.7;
  cv::dnn::Net net;

private:
  std::string _faceDetectorConfigFile;
  std::string _faceDetectorWeightFile;
  

};

#endif /* FACE_DETECTOR_OPENCV_HPP */
