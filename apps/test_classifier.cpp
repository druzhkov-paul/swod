#include <swod/swod.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


size_t getTotalBboxesNum(const vector<ImageAnnotation> & ann)
{
    size_t n = 0;
    for (size_t i = 0; i < ann.size(); ++i)
    {
        n += ann[i].bboxes.size();
    }
    return n;
}


void flipSources(SourcesMap & sources)
{
    for (auto i = sources.begin(); i != sources.end(); ++i)
    {
        if (i->first.first != "image")
        {
            cout << "warning. nonimage source: \""
                 << i->first.first << "\" : " << i->first.second
                 << ". flipping operation may lead to an error."
                 << endl;
        }
        Mat & image = i->second;
        flip(image, image, 1);
    }
}


ImageAnnotation getFlipedAnnotation(const ImageAnnotation & ann,
                                    const SourcesMap & sources)
{
    ImageAnnotation flipedAnn;
    for (size_t i = 0; i < ann.bboxes.size(); ++i)
    {
        if (ann.labels[i] == 0)
        {
            continue;
        }
        flipedAnn.labels.push_back(ann.labels[i]);
        flipedAnn.scores.push_back(ann.scores[i]);
        Rect bbox = ann.bboxes[i];
        Size imageSize = sources.at(DataTypeTime("image", 0)).size();
        bbox.x = imageSize.width - bbox.x - bbox.width;
        flipedAnn.bboxes.push_back(bbox);
    }
    return flipedAnn;
}


void extractSamples(ImageFileReader * fileReader,
                    const vector<ImageAnnotation> & annotations,
                    Features & features,
                    Mat & samples,
                    Mat & responses,
                    bool doMirror = false)
{
    CV_Assert(samples.cols == features.getTotalFeatureVectorLength());
    int samplesNum = 0;
    for (size_t i = 0; i < annotations.size(); ++i)
    {
        const ImageAnnotation & ann = annotations[i];
        for (auto j = ann.sources.begin(); j != ann.sources.end(); ++j)
        {
            fileReader->addSource(j->first, j->second);
        }
        fileReader->open();
        CV_Assert(fileReader->grab());
        SourcesMap sources;
        CV_Assert(fileReader->retrieve(sources, features));
        fileReader->release();

        size_t m = ann.bboxes.size();
        Mat sample = samples.rowRange(samplesNum, samplesNum + m);
        features.getROIDescription(sample, sources, ann.bboxes);

        for (size_t j = 0; j < m; ++j)
        {
            responses.at<float>(samplesNum + j) = static_cast<float>(ann.labels[j]);
        }
        samplesNum += m;


        if (doMirror)
        {
            flipSources(sources);
            ImageAnnotation flipedAnn = getFlipedAnnotation(ann, sources);
            m = flipedAnn.bboxes.size();
            Mat sample = samples.rowRange(samplesNum, samplesNum + m);
            features.getROIDescription(sample, sources, flipedAnn.bboxes);
            for (size_t j = 0; j < m; ++j)
            {
                responses.at<float>(samplesNum + j) = static_cast<float>(flipedAnn.labels[j]);
            }
            samplesNum += m;
        }
    }
}


void dumpDatasetCSV(const Mat & dataset,
                    const Mat & responses,
                    const string & fileName)
{
    CV_Assert(static_cast<size_t>(dataset.rows) == responses.total());
    ofstream f(fileName.c_str());
    for (int i = 0; i < dataset.rows; ++i)
    {
        f << responses.at<float>(i);
        for (int j = 0; j < dataset.cols; ++j)
        {
            f << ";" << dataset.at<float>(i, j);
        }
        f << endl;
    }
    f.close();
}


void getClassifierError(const Classifier * cl,
                        const Mat & dataset,
                        const Mat & responses,
                        Mat & confusion,
                        int classesNum = 2)
{

    confusion.create(classesNum, classesNum, CV_32S);
    confusion = Scalar();
    for (int i = 0; i < dataset.rows; ++i)
    {
        Mat sample = dataset.row(i);
        vector<float> confidence;
        int prediction = cl->predict(sample, confidence);
        int response = responses.at<float>(i);
        confusion.at<int>(prediction, response) += 1;
    }
}


void printClassificationError(const Classifier * cl,
                              const Mat & dataset,
                              const Mat & responses,
                              int classesNum = 2)
{
    Mat confusion;
    getClassifierError(cl, dataset, responses, confusion, classesNum);
    for (int i = 0; i < classesNum; ++i)
    {
        for (int j = 0; j < classesNum - 1; ++j)
        {
            cout << confusion.at<int>(i, j) << "\t";
        }
        cout << confusion.at<int>(i, classesNum - 1) << endl;
    }

    float accuracy = static_cast<float>(sum(confusion.diag())[0]) /
                     static_cast<float>(sum(confusion)[0]);
    cout << "accuracy: " << accuracy << endl;
}


int main(int argc, char ** argv)
{
    initFeatures();
    initClassifiers();
    initDataProviders();

    const string commandLineKeys = "{h|help|false|show help and exit}"
                                   "{c|config||.xml or .yml file containing "
                                   "configuration parameters, i.e. features to used "
                                   "and thier parameters, classifier type "
                                   "and params of the training algorithm}"
                                   "{a|ann||path to annotation with positive examples}"
                                   "{|annname|annotation|annotation name}"
                                   "{|dump||prefix of files to save dataset to}"
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

    // general parameters
    FileNode generalParamsFn = config["general_params"];
    if (generalParamsFn.empty())
    {
        cout << "Error. general_params tag is missed in config file" << endl;
        return 2;
    }
    cout << "reading general parameters..." << flush;
    Size winSize, winBorderSize;
    generalParamsFn["windowSize"] >> winSize;
    generalParamsFn["borderSize"] >> winBorderSize;
    cout << "done" << endl;

    // configure feature descriptors
    Features features;
    FileNode featuresFn = config["features"];
    for (FileNodeIterator i = featuresFn.begin(); i != featuresFn.end(); ++i)
    {
        FileNode featureType = (*i)["name"];
        CV_Assert(!featureType.empty());
        string featureName;
        featureType >> featureName;
        Ptr<Feature> feature = Algorithm::create<Feature>(featureName);
        feature->read(*i);
        features.featuresSet.push_back(feature);
        cout << featureName << " feature is used" << endl;
    }

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
    classifier->loadModel(classifierModelFile, classifierModelName);

    config.release();


    vector<ImageAnnotation> annotation;
    cout << "loading annotation..." << flush;
    loadDatasetAnnotation(cmdParser.get<string>("ann"),
                          cmdParser.get<string>("annname"),
                          annotation);
    for (size_t i = 0; i < annotation.size(); ++i)
    {
        annotation[i].addBBoxesBorder(winSize,
                                      winBorderSize);
    }
    cout << "done" << endl;


    Ptr<ImageFileReader> fileReader = Algorithm::create<ImageFileReader>("SWOD.DataProvider.ImageFileReader");
    Mat samples, responses;
    int samplesNum = getTotalBboxesNum(annotation);
    samples.create(samplesNum, features.getTotalFeatureVectorLength(), CV_32F);
    responses.create(samplesNum, 1, CV_32F);

    cout << "extracting samples..." << flush;
    extractSamples(fileReader, annotation, features, samples, responses, false);
    cout << "done" << endl;
    cout << "dataset size: " << samplesNum << " x " << samples.cols << endl;

    if (cmdParser.get<string>("dump") != "")
    {
        cout << "saving dataset to " << cmdParser.get<string>("dump") << ".csv..." << flush;
        dumpDatasetCSV(samples.rowRange(0, samplesNum),
                       responses.rowRange(0, samplesNum),
                       cmdParser.get<string>("dump") + ".csv");
        cout << "done" << endl;
    }

    cout << "testing classifier..." << endl;
    double maxResponse = 0.0;
    minMaxLoc(responses.rowRange(0, samplesNum), 0, &maxResponse);
    printClassificationError(classifier,
                             samples.rowRange(0, samplesNum),
                             responses.rowRange(0, samplesNum),
                             static_cast<int>(maxResponse) + 1);

    return 0;
}
