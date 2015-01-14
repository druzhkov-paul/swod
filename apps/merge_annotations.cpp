#include "swod/swod.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <iomanip>


using namespace std;
using namespace cv;


int main(int argc, char ** argv)
{
    const string commandLineKeys = "{h|help|false|show help and exit}"
                                   "{n||0|count of files to merge}"
                                   "{a|name|annotation|internal annotation structure name}"
                                   "{i|in||prefix of input annotation files}"
                                   "{o|out||output file name}";
    CommandLineParser cmdParser(argc, argv, commandLineKeys.c_str());
    if (cmdParser.get<bool>("help"))
    {
        cmdParser.printParams();
        return 0;
    }
    CV_Assert(cmdParser.get<string>("in") != "" && cmdParser.get<string>("out") != "" && 0 < cmdParser.get<int>("n"));

    vector<ImageAnnotation> ann;
    stringstream s;
    for (int i = 0; i < cmdParser.get<int>("n"); ++i)
    {
        s.str("");
        s << cmdParser.get<string>("in") << "-" << i << ".yml";
        cout << "reading " << s.str() << endl;
        vector<ImageAnnotation> a;
        loadDatasetAnnotation(s.str(), cmdParser.get<string>("name"), a);
        for (auto j = a.begin(); j != a.end(); ++j)
        {
            ann.push_back(*j);
        }
    }

    saveDatasetAnnotation(cmdParser.get<string>("out"),
                          cmdParser.get<string>("name"),
                          ann);
    return 0;
}
