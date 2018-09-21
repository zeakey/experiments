#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv ) {
    if( argc != 2)
    {
     cout <<" Missing argument: image file path" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data ) {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    int N = image.rows * image.cols * 3;
    float m = 0.0;
    for (int i = 0; i < N; ++i) m += image.data[i];

    m /= N;
    cout << "mean: " << m << ", mat.mean.val[0]:" << mean(image).val[1] << endl;

    return 0;
}