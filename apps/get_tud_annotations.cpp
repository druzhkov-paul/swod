#include "swod/swod.hpp"

#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace cv;


int main()
{
    string resultPosAnnFile = "./tud_positive.yml";
    string resultNegAnnFile = "./tud_negative.yml";
    vector<ImageAnnotation> positiveAnn;

    // positive TUD-MotionPairs
    string basePositiveAnnFile = "/home/paul/programs/pedestrian detection/data/TUD/TUD-MotionPairs/positive/train-pos.idl";
    fstream f(basePositiveAnnFile.c_str(), ios_base::in);
    for (size_t i = 0; !f.eof(); ++i)
    {
        string s;
        getline(f, s, '\n');
        ImageAnnotation ann;
        stringstream ss;
        ss << "img-" << setw(4) << setfill('0') << i << "-0.png";
        CV_Assert(static_cast<size_t>(1) == s.find(ss.str()));
        ann.sources[DataTypeTime("image", 0)] = "TUD-MotionPairs/positive/" + ss.str();
        ss.str("");
        ss << "img-" << setw(4) << setfill('0') << i << "-1.png";
        ann.sources[DataTypeTime("image", 1)] = "TUD-MotionPairs/positive/" + ss.str();

        size_t n = s.find(':');
        while ((n = s.find('(', n)) != string::npos)
        {
            Rect r;
            int x1, y1, x2, y2;
            n += sscanf(s.substr(n, string::npos).c_str(),
                        "(%d, %d, %d, %d)",
                        &x1, &y1, &x2, &y2) - 1;

            r.x = std::min(x1, x2);
            r.y = std::min(y1, y2);
            r.width = std::max(x1, x2);
            r.height = std::max(y1, y2);
            r.width -= r.x;
            r.height -= r.y;
            ann.addBBox(r, 1, ImageAnnotation::GROUND_TRUTH);
        }
        positiveAnn.push_back(ann);
    }
    f.close();



    // positive TUD-MotionPairs
    string baseAdditionalPositiveAnnFile = "/home/paul/programs/pedestrian detection/data/TUD/TUD-MotionPairs/additional-negative-bootstrap/annotation.idl";
    f.open(baseAdditionalPositiveAnnFile.c_str(), ios_base::in);
    for (size_t i = 0; i < 26; ++i)
    {
        string s;
        getline(f, s, '\n');
        ImageAnnotation ann;
        stringstream ss;
        ss << "img-" << setw(2) << setfill('0') << i << "-0.png";
        CV_Assert(static_cast<size_t>(1) == s.find(ss.str()));
        ann.sources[DataTypeTime("image", 0)] = "TUD-MotionPairs/additional-negative-bootstrap/" + ss.str();
        ss.str("");
        ss << "img-" << setw(2) << setfill('0') << i << "-1.png";
        ann.sources[DataTypeTime("image", 1)] = "TUD-MotionPairs/additional-negative-bootstrap/" + ss.str();

        size_t n = s.find(':');
        while ((n = s.find('(', n)) != string::npos)
        {
            Rect r;
            int x1, y1, x2, y2;
            n += sscanf(s.substr(n, string::npos).c_str(),
                        "(%d, %d, %d, %d)",
                        &x1, &y1, &x2, &y2) - 1;

            r.x = std::min(x1, x2);
            r.y = std::min(y1, y2);
            r.width = std::max(x1, x2);
            r.height = std::max(y1, y2);
            r.width -= r.x;
            r.height -= r.y;
            if (0 <= r.x && 0 <= r.y && 0 < r.width && 0 < r.height)
            {
                ann.addBBox(r, 1, ImageAnnotation::GROUND_TRUTH);
            }
            else
            {
                cout << "invalid bounding box for \""
                     << ann.sources.at(DataTypeTime("image", 0))
                     << "\": (" << r.x << ", " << r.y << ", "
                     << r.width << ", " << r.height << ")" << endl;
            }
        }
        positiveAnn.push_back(ann);
    }
    f.close();


    // negative TUD-MotionPairs
    vector<ImageAnnotation> negativeAnn;
    for (size_t i = 0; i < 192; ++i)
    {
        ImageAnnotation ann;
        stringstream ss;
        ss << "img-" << setw(3) << setfill('0') << i << "-0.png";
        ann.sources[DataTypeTime("image", 0)] = "TUD-MotionPairs/negative/" + ss.str();
        ss.str("");
        ss << "img-" << setw(3) << setfill('0') << i << "-1.png";
        ann.sources[DataTypeTime("image", 1)] = "TUD-MotionPairs/negative/" + ss.str();
        negativeAnn.push_back(ann);
    }


    // TUD-Brussels
    string baseAnnFile = "/home/paul/programs/pedestrian detection/data/TUD/TUD-Brussels/annotation.idl";
    f.open(baseAnnFile.c_str(), ios_base::in);
    for (size_t i = 0; i < 508; ++i)
    {
        string s;
        getline(f, s, '\n');

        ImageAnnotation ann;
        stringstream ss;
        ss << "img-" << setw(3) << setfill('0') << i << "-2.png";
        CV_Assert(s.find(ss.str()) == 1);
        ann.sources[DataTypeTime("image", 0)] = "TUD-Brussels/" + ss.str();
        ss.str("");
        ss << "img-" << setw(3) << setfill('0') << i << "-3.png";
        ann.sources[DataTypeTime("image", 1)] = "TUD-Brussels/" + ss.str();

        size_t n = s.find(':');
        while ((n = s.find('(', n)) != string::npos)
        {
            Rect r;
            int x1, y1, x2, y2;
            n += sscanf(s.substr(n, string::npos).c_str(),
                        "(%d, %d, %d, %d)",
                        &x1, &y1, &x2, &y2) - 1;

            r.x = std::min(x1, x2);
            r.y = std::min(y1, y2);
            r.width = std::max(x1, x2);
            r.height = std::max(y1, y2);
            r.width -= r.x;
            r.height -= r.y;
            ann.addBBox(r, 1, ImageAnnotation::GROUND_TRUTH);
        }

        if (ann.bboxes.size())
        {
            positiveAnn.push_back(ann);
        }
        else
        {
            negativeAnn.push_back(ann);
        }
    }
    f.close();

    saveDatasetAnnotation(resultPosAnnFile, "annotation", positiveAnn);
    saveDatasetAnnotation(resultNegAnnFile, "annotation", negativeAnn);
}
