#include "swod/swod.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace cv;


int main(int argc, char ** argv)
{
    int argn = 0;
    CV_Assert(argc == 4);
    string annotationPath = argv[++argn];
    string annotationName = argv[++argn];
    string resultDir = argv[++argn];

    vector<ImageAnnotation> ann;
    loadDatasetAnnotation(annotationPath, annotationName, ann);

    for (size_t i = 0; i < ann.size(); ++i)
    {
        const ImageAnnotation & a = ann[i];
        string imagePath = a.sources.at(DataTypeTime("image", 0));

        size_t n1 = imagePath.find("set");
        size_t n2 = imagePath.find("/I");
        size_t n3 = imagePath.rfind('.');
        string subdir = imagePath.substr(n1, n2 - n1);
        string fileName = imagePath.substr(n2 + 1, n3 - n2 - 1);

        if (resultDir[resultDir.length() - 1] == '/')
        {
            subdir = resultDir + subdir + "/";
        }
        else
        {
            subdir = resultDir + "/" + subdir + "/";
        }

        cout << subdir + fileName + ".txt" << endl;
        fstream f(subdir + fileName + ".txt", fstream::out);
        for (size_t j = 0; j < a.bboxes.size(); ++j)
        {
            const Rect & r = a.bboxes[j];
            f << r.x << "," << r.y << "," << r.width << "," << r.height
              << "," << a.scores[j] << "\n";
        }
        f.close();
    }

    return 0;
}
