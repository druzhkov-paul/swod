#include <swod/swod.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <ctime>

using namespace std;
using namespace cv;


void readDatasetCSV(const string & fileName,
                    Mat & dataset,
                    Mat & responses)
{
    ifstream f(fileName);
    CV_Assert(f.is_open());

    // determine dataset size
    string s;
    getline(f, s);
    int featuresNum = static_cast<int>(count(s.begin(), s.end(), ';'));
    int samplesNum = 1;
    while (!f.eof())
    {
        getline(f, s);
        if (!f.eof() && s != "")
        {
            ++samplesNum;
        }
    }
    f.close();

    // allocate memory for dataset
    Mat data(samplesNum, featuresNum + 1, CV_32F);
    dataset = data.colRange(1, featuresNum + 1);
    responses = data.col(0);

    // read data
    char delimiter;
    f.open(fileName);
    CV_Assert(f.is_open());
    for (int i = 0; i < samplesNum; ++i)
    {
        f >> responses.at<float>(i);
        for (int j = 0; j < featuresNum; ++j)
        {
            f >> delimiter >> dataset.at<float>(i, j);
            CV_Assert(delimiter == ';');
        }
    }
    f.close();
}


void getClassifierError(const Classifier * cl,
                        const Mat & dataset,
                        const Mat & responses,
                        Mat & errors,
                        int classesNum = 2)
{

    errors.create(classesNum, classesNum, CV_32S);
    errors = Scalar();
    for (int i = 0; i < dataset.rows; ++i)
    {
        Mat sample = dataset.row(i);
        vector<float> confidence;
        int prediction = cl->predict(sample, confidence);
        int response = responses.at<float>(i);
        errors.at<int>(prediction, response) += 1;
    }
}


void printClassificationError(const Classifier * cl,
                              const Mat & dataset,
                              const Mat & responses,
                              int classesNum = 2)
{
    Mat errors;
    getClassifierError(cl, dataset, responses, errors, classesNum);
    for (int i = 0; i < classesNum; ++i)
    {
        for (int j = 0; j < classesNum - 1; ++j)
        {
            cout << errors.at<int>(i, j) << "\t";
        }
        cout << errors.at<int>(i, classesNum - 1) << endl;
    }

    float accuracy = static_cast<float>(sum(errors.diag())[0]) /
                     static_cast<float>(sum(errors)[0]);
    cout << "accuracy: " << accuracy << endl;
}


int main(int argc, char ** argv)
{
    initClassifiers();

    const string commandLineKeys = "{h|help|false|show help and exit}"
                                   "{c|config||.xml or .yml file containing "
                                   "configuration parameters, i.e. classifier type "
                                   "and params of the training algorithm}"
                                   "{d|dataset||path to dataset csv-file}"
                                   "{|seed||seed to initialize RNG}";
    CommandLineParser cmdParser(argc, argv, commandLineKeys.c_str());
    if (cmdParser.get<bool>("help"))
    {
        cmdParser.printParams();
        return 0;
    }

    if (cmdParser.get<string>("seed") == "")
    {
        theRNG() = RNG(time(0));
    }
    else
    {
        theRNG() = RNG(cmdParser.get<uint64>("seed"));
    }
    cout << "rng state: " << theRNG().state << endl;

    FileStorage config(cmdParser.get<string>("config"), FileStorage::READ);
    CV_Assert(config.isOpened());

    // configure classifier
    FileNode classifierFn = config["classifier"];
    FileNode classifierType = classifierFn["name"];
    CV_Assert(!classifierType.empty());
    string classifierName;
    classifierType >> classifierName;
    Ptr<Classifier> classifier = Algorithm::create<Classifier>(classifierName);
    classifier->read(classifierFn);
    cout << classifierName << " classifier is used" << endl;
    string classifierModelFile = classifierFn["modelFileName"];
    string classifierModelName = classifierFn["modelName"];


    Mat samples, responses;
    cout << "reading dataset..." << flush;
    readDatasetCSV(cmdParser.get<string>("dataset"), samples, responses);
    cout << "done" << endl;
    cout << "dataset size: " << samples.rows << " x " << samples.cols << endl;

    cout << "training classifier..." << flush;
    classifier->train(samples, responses);
    cout << "done" << endl;

    double maxResponse = 0.0;
    minMaxLoc(responses, 0, &maxResponse);
    printClassificationError(classifier,
                             samples,
                             responses,
                             static_cast<int>(maxResponse) + 1);

    classifier->saveModel(classifierModelFile, classifierModelName);

    return 0;
}
