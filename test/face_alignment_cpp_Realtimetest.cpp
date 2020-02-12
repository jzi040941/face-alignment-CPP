#include <FaceDetectorOpencv.hpp>
#include <opencv2/core.hpp>

int
main
  (
  int argc,
  char **argv
  )
{
  cv::CommandLineParser parser(
            argc, argv,
            "{ help h usage ?    |      | give the following arguments in following format }"
            "{ model_filename f  |      | (required) path to binary file storing the trained model which is to be loaded [example - /data/file.dat]}"
 "{face_ssd_config_file|../data/opencv_face_detector.pbtxt|Path to face cascade(ssd model pbtxt).}"
"{face_ssd_weight_file|../data/opencv_face_detector_uint8.pb|Path to face cascade(ssd model pbfile).}"
            "{camera|0|Camera device number.}");
   
  FaceDetectorOpencv detector(parser);


  return 0;
}
