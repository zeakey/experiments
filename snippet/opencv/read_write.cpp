#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
#include <iostream>

using namespace cv;
using namespace std;

cv::Mat prepare_image(cv::Mat im) {
    im.convertTo(im, CV_32FC3);
    im -= cv::Scalar(1, 1, 1);
    im *= 0.1;
    return im;
}

float mat_mean(cv::Mat mat) {
    cv::Scalar mean = cv::mean(mat);
    return (mean.val[0] + mean.val[1] + mean.val[2]) / 3;
}

int main( int argc, char** argv ) {
    cout << "OpenCV version: " << CV_VERSION << endl;
    if( argc != 2)
    {
     cout <<" Missing argument: image file path" << endl;
     return -1;
    }

    Mat image, image_resize;

    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file
    if(! image.data ) {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    resize(image, image_resize, Size(50, 50));
    cout << "Raw mean=" << mat_mean(image) << endl;
    cout << "Resized mean=" << mat_mean(image_resize) << endl;
    return 0;
}
