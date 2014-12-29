#include "swod/swod.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <ctime>

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
                    int & samplesNum,
                    bool doMirror = false)
{
    size_t n = 2 * getTotalBboxesNum(annotations);
    if (samples.empty())
    {
        samples.create(n, features.getTotalFeatureVectorLength(), CV_32F);
        responses.create(n, 1, CV_32F);
        samplesNum = 0;
    }
    else if (static_cast<size_t>(samples.rows) < samplesNum + n)
    {
        samples.resize(samplesNum + n);
        responses.resize(samplesNum + n);
    }
    CV_Assert(samples.cols == features.getTotalFeatureVectorLength());
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

        for (size_t j = 0; j < ann.bboxes.size(); ++j)
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
            for (size_t j = 0; j < flipedAnn.bboxes.size(); ++j)
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
    fstream f(fileName.c_str(), fstream::out);
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


void getWhiteningTransform(const Mat & dataset,
                           Mat & transf,
                           float epsilon)
{
    Mat cov, m;
    calcCovarMatrix(dataset, cov, m, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE, CV_64F);

    Mat eigenValues, eigenVectors;
    eigen(cov, eigenValues, eigenVectors);

    CV_Assert(eigenValues.total() == static_cast<size_t>(cov.cols));
    CV_Assert(eigenVectors.size() == cov.size());

    for (size_t i = 0; i < eigenValues.total(); ++i)
    {
        eigenValues.at<double>(i) = 1.0 / sqrt(eigenValues.at<double>(i) + epsilon);
    }

    transf = eigenVectors.t() * Mat::diag(eigenValues) * eigenVectors;
    transf.convertTo(transf, CV_32F);
}



int main(int argc, char ** argv)
{
    initFeatures();
    initDataProviders();

    const string commandLineKeys = "{h|help|false|show help and exit}"
                                   "{c|config||.xml or .yml file containing detector "
                                   "configuration parameters, i.e. features to used "
                                   "and thier parameters, classifier type "
                                   "and params of the training algorithm, "
                                   "general detection parameters}"
                                   "{|pos||path to annotation with positive examples}"
                                   "{|posname|annotation|name of the positive annotation file node to read from}"
                                   "{|neg||path to annotation with negative examples}"
                                   "{|negname|annotation|name of the negative annotation file node to read from}"
                                   "{|dump||prefix of files to save dataset to}"
                                   "{|rand|50|number of random samples to draw from each negative image}"
                                   "{|mirror|true|use horizontaly flip positives for training}"
                                   "{|mem|0|preallocate memory for specified "
                                   "number of samples in the dataset. "
                                   "If more samples are needed dataset resize "
                                   "is performed that may lead to additional memory usage}"
                                   "{|seed||seed to initialize RNG}"
                                   "{|whiteeps|0.01|whitening regularization constant}"
                                   "{|whitefile|whitening.yml|file to save whitening transform to}";
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

    // read detection parameters
    DetectionParams detectorParams;
    FileNode detectorParamsFn = config["detector_params"];
    if (detectorParamsFn.empty())
    {
        cout << "Error. detection_params tag is missed in config file" << endl;
        return 2;
    }
    cout << "reading general detector parameters..." << flush;
    detectorParamsFn >> detectorParams;
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

    vector<ImageAnnotation> positiveAnn;
    vector<ImageAnnotation> negativeAnn;
    cout << "loading annotations..." << flush;
    loadDatasetAnnotation(cmdParser.get<string>("pos"),
                          cmdParser.get<string>("posname"),
                          positiveAnn);
    for (size_t i = 0; i < positiveAnn.size(); ++i)
    {
        positiveAnn[i].addBBoxesBorder(detectorParams.winSize,
                                       detectorParams.winBorder);
    }
    loadDatasetAnnotation(cmdParser.get<string>("neg"),
                          cmdParser.get<string>("negname"),
                          negativeAnn);
    cout << "done" << endl;

    cout << "generating random negative samples..." << flush;
    for (size_t i = 0; i < negativeAnn.size(); ++i)
    {
        negativeAnn[i].generateRandomAnnotation(detectorParams.winSize,
                                                Size(),
                                                detectorParams.minObjectHeight,
                                                detectorParams.maxObjectHeight,
                                                cmdParser.get<int>("rand"));
    }
    cout << "done" << endl;

    Ptr<ImageFileReader> fileReader = Algorithm::create<ImageFileReader>("SWOD.DataProvider.ImageFileReader");
    Mat samples, responses;
    if (0 < cmdParser.get<int>("mem"))
    {
        int n = cmdParser.get<int>("mem");
        samples.create(n, features.getTotalFeatureVectorLength(), CV_32F);
        responses.create(n, 1, CV_32F);
    }
    int samplesNum = 0;
    cout << "extracting samples..." << flush;
    cout << "negative..." << flush;
    extractSamples(fileReader, negativeAnn, features, samples, responses,
                   samplesNum, false);
    cout << "positive..." << flush;
    extractSamples(fileReader, positiveAnn, features, samples, responses,
                   samplesNum, cmdParser.get<bool>("mirror"));
    cout << "done" << endl;
    cout << "dataset size: " << samplesNum << " x " << samples.cols << endl;

    if (cmdParser.get<string>("dump") != "")
    {
        cout << "saving dataset to " << cmdParser.get<string>("dump") << "-0.csv..." << flush;
        dumpDatasetCSV(samples.rowRange(0, samplesNum),
                       responses.rowRange(0, samplesNum),
                       cmdParser.get<string>("dump") + "-0.csv");
        cout << "done" << endl;
    }

    cout << "geting whitening transform..." << flush;
    Mat whiteningTransform;
    float whiteningEpsilon = cmdParser.get<float>("whiteeps");
    getWhiteningTransform(samples.rowRange(0, samplesNum),
                          whiteningTransform,
                          whiteningEpsilon);
    cout << "done" << endl;

    cout << "saving whitening matrix..." << flush;
    FileStorage fs(cmdParser.get<string>("whitefile"), FileStorage::READ);
    fs << "whiteningMat" << whiteningTransform;
    fs.release();
    cout << "done" << endl;

    return 0;
}
