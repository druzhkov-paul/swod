#include "swod/swod.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace cv;


void getCalTechAnn(const string & baseAnnFile,
                        const string & resultAnnFile)
{
    vector<string> baseAnn;
    fstream f(baseAnnFile.c_str(), ios_base::in);
    while (!f.eof())
    {
        string s;
        f >> s;
        baseAnn.push_back(s);
    }
    f.close();


    vector<ImageAnnotation> resultAnn;
    for (size_t i = 0; i < baseAnn.size(); ++i)
    {
        string s = baseAnn[i];
        //cout << s << endl;
        size_t fileStartPos = s.find("/set", 0);
        CV_Assert(string::npos != fileStartPos);
        s = s.substr(fileStartPos + 1);
        size_t frameIdxStart = s.find("/I", 0);
        CV_Assert(string::npos != frameIdxStart);
        size_t frameIdxEnd = s.rfind(".");
        CV_Assert(string::npos != frameIdxEnd);
        size_t fileNameEnd = s.rfind("\"");
        CV_Assert(string::npos != fileNameEnd);

        int frameIdx = stoi(s.substr(frameIdxStart + 2, frameIdxEnd));
        stringstream ss;
        ss << setw(5) << setfill('0') << frameIdx;
        ImageAnnotation ann;
        ann.sources[DataTypeTime("image", 0)] = "./" + s.substr(0, frameIdxStart + 2) +
                               ss.str() + s.substr(frameIdxEnd, fileNameEnd - frameIdxEnd);
        ss.str("");
        ss << setw(5) << setfill('0') << frameIdx + 1;
        ann.sources[DataTypeTime("image", 1)] = "./" + s.substr(0, frameIdxStart + 2) +
                                          ss.str() + s.substr(frameIdxEnd, fileNameEnd - frameIdxEnd);
        resultAnn.push_back(ann);
    }
    saveDatasetAnnotation(resultAnnFile, "annotation", resultAnn);
}


int main()
{
    getCalTechAnn("/home/paul/programs/pedestrian detection/data/CalTech/test_images/calTech-test.idl",
                  "./caltech_test.yml");
    getCalTechAnn("/home/paul/programs/pedestrian detection/data/CalTech/test_images/calTech-train.idl",
                  "./caltech_train.yml");
    return 0;
}
