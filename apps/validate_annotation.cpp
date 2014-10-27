#include "swod/swod.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;


int main(int argc, char ** argv)
{
    const DataTypeTime SOURCE_IMAGE("image", 0);

    CV_Assert(argc == 3);
    int argn = 0;
    string annotationPath = argv[++argn];
    string annotationName = argv[++argn];

    vector<ImageAnnotation> ann;
    loadDatasetAnnotation(annotationPath, annotationName, ann);

    size_t errorsNum = 0;
    for (size_t i = 0; i < ann.size(); ++i)
    {
        const ImageAnnotation & a = ann[i];
        if (a.sources.count(SOURCE_IMAGE) == 0)
        {
            cout << "no source image" << endl;
            ++errorsNum;
            continue;
        }
        string imagePath = a.sources.at(SOURCE_IMAGE);

        Mat image = imread(imagePath);
        if (image.empty())
        {
            cout << "wrong image path \"" << imagePath << "\"" << endl;
            ++errorsNum;
            continue;
        }
        for (size_t j = 0; j < a.bboxes.size(); ++j)
        {
            const Rect & r = a.bboxes[j];
            if (r.x < 0 ||
                r.y < 0 ||
                r.width <= 0 ||
                r.height <= 0 ||
                image.cols < r.x + r.width ||
                image.rows < r.y + r.height)
            {
                cout << "invalid bounding box for \"" << imagePath << "\":\n"
                     << "bbox: (" << r.x << ", " << r.y << ", " << r.width << ", " << r.height << ") "
                     << "image size: " << image.cols << " x " << image.rows << endl;
                ++errorsNum;
            }
        }
    }

    cout << "Total number of errors: " << errorsNum << endl;
    return 0;
}
