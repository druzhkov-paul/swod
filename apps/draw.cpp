#include "swod/annotation.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <map>
#include <sstream>
#include <iostream>

using std::string;
using std::stringstream;
using std::map;
using std::cout;
using std::endl;
using namespace cv;


namespace
{
    string getFileName(const string & filePath)
    {
        size_t slashPos = filePath.rfind("/");
        string fileName = filePath;
        if (slashPos != string::npos)
        {
            fileName = filePath.substr(slashPos + 1);
        }
        return fileName;
    }


    void generateColors(map<int, Scalar> & colors)
    {
        colors[0] = CV_RGB(0, 0, 255);
        colors[1] = CV_RGB(0, 255, 0);
        colors[2] = CV_RGB(255, 0, 0);
        colors[3] = CV_RGB(0, 255, 255);
        colors[4] = CV_RGB(255, 0, 255);
    }


    void addRandomColor(map<int, Scalar> & colors, int n)
    {
        RNG & rng = theRNG();
        colors[n] = CV_RGB(uchar(rng), uchar(rng), uchar(rng));
    }
}


int main(int argc, char ** argv)
{
    if (argc != 3)
    {
        cout << "usage\n\t./draw annotation outputDir" << endl;
        return 1;
    }
    string annPath = argv[1];
    string outputDirPath = argv[2];
    vector<ImageAnnotation> ann;
    loadDatasetAnnotation(annPath, "annotation", ann);
    map<int, Scalar> colors;
    generateColors(colors);
    for (size_t i = 0; i < ann.size(); ++i)
    {
        string imagePath = ann[i].sources[DataTypeTime("image", 0)];
        Mat image = imread(imagePath);
        for (size_t j = 0; j < ann[i].bboxes.size(); ++j)
        {
            int label = ann[i].labels[j];
            if (colors.count(label) == 0)
            {
                addRandomColor(colors, label);
            }
            rectangle(image, ann[i].bboxes[j], colors.at(label), 2);
        }
        stringstream s;
        s << outputDirPath << getFileName(imagePath);
        imwrite(s.str(), image);
    }
    return 0;
}
