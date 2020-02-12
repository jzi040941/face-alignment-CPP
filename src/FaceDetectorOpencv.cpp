#include <FaceDetectorOpencv.hpp>
//#include <trace.hpp>
//#include <utils.hpp>
//#include <ModernPosit.h>
//#include <boost/program_options.hpp>
using namespace cv::dnn;
using namespace cv;
  const float BBOX_SCALE = 0.3f;
  const cv::Size FACE_SIZE = cv::Size(256,256);
  const cv::Scalar meanVal(104.0, 177.0, 123.0);


void FaceDetectorOpencv::detectFaceOpenCVDNN(const cv::Mat& frameOpenCVDNN,  std::vector<Rect> &faces,
    Net net)
{    
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
#ifdef CAFFE
        cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
#else
        cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
#endif

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    faces.clear();
    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            
            //faces.push_back(upm::FaceAnnotation());
            //faces[i].bbox.pos = cv::Rect2f(x1,y1,x2-x1,y2-y1);
            std::cout<<"xyxy"<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<std::endl;
            faces.push_back(Rect(Point2i(x1,y1),Point2i(x2,y2)));
            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }



}

  void FaceDetectorOpencv::load(cv::CommandLineParser parser)
{
  _faceDetectorConfigFile = samples::findFile( parser.get<String>("face_ssd_config_file") );
  _faceDetectorWeightFile = samples::findFile( parser.get<String>("face_ssd_weight_file") );



  std::cout<<"configfile:"<<_faceDetectorConfigFile<<std::endl;
  /* 
    if (not face_cascade.load(_cascade_name)) {
      std::cerr << "Cannot load cascade classifier: " << _cascade_name << std::endl;
    }
  */
  
  net = cv::dnn::readNetFromTensorflow(_faceDetectorWeightFile, _faceDetectorConfigFile);
  
}

void
FaceDetectorOpencv::process
  (
  cv::Mat frameOpenCVDNN,
  std::vector<Rect> &faces
  )
{
  detectFaceOpenCVDNN(frameOpenCVDNN,  faces, net);
  //faceDetector(frameOpenCVDNN,  faces, face_cascade);
  
}

