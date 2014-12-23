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


void getBootstrapSample(vector<ImageAnnotation> & ann,
                        size_t sampleSize = 15000)
{
    size_t n = getTotalBboxesNum(ann);
    vector<int> sample(sampleSize);
    theRNG().fill(sample, RNG::UNIFORM, 0, n);
    std::sort(sample.begin(), sample.end());
    int m = 0;
    size_t j = 0;
    for (size_t i = 0; i < ann.size(); ++i)
    {
        ImageAnnotation a;
        a.sources = ann[i].sources;
        size_t k = ann[i].bboxes.size();
        size_t idx = sample[j] - m;
        while (idx < k)
        {
            a.addBBox(ann[i].bboxes[idx],
                      ann[i].labels[idx],
                      ann[i].scores[idx]);
            idx = sample[++j] - m;
        }
        ann[i] = a;
        m += k;
    }
}


void getClassifierError(const Classifier * cl,
                        const Mat & dataset,
                        const Mat & responses,
                        Mat & errors)
{
    errors.create(2, 2, CV_32F);
    errors = Scalar();
    for (int i = 0; i < dataset.rows; ++i)
    {
        Mat sample = dataset.row(i);
        vector<float> confidence;
        int prediction = cl->predict(sample, confidence);
        int response = responses.at<float>(i);
        errors.at<float>(prediction, response) += 1;
    }
}


void printClassificationError(const Classifier * cl,
                              const Mat & dataset,
                              const Mat & responses)
{
    Mat errors;
    getClassifierError(cl, dataset, responses, errors);
    cout << errors.at<float>(0, 0) << "\t" << errors.at<float>(0, 1) << endl;
    cout << errors.at<float>(1, 0) << "\t" << errors.at<float>(1, 1) << endl;
    cout << "accuracy: " << (errors.at<float>(0, 0) + errors.at<float>(1, 1)) / sum(errors)[0] << endl;
}



int main(int argc, char ** argv)
{
    initFeatures();
    initClassifiers();
    initDataProviders();
    initNonMaximumSuppressors();

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
                                   "{|bi|3|number of bootstrap iterations}"
                                   "{|samples|0|number of samples to draw from all false detections at each iteration. All false positives are used by default}"
                                   "{|mirror|true|use horizontaly flip positives for training}"
                                   "{|mem|0|preallocate memory for specified "
                                   "number of samples in the dataset. "
                                   "If more samples are needed dataset resize "
                                   "is performed that may lead to additional memory usage}"
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

    // read detection parameters
    DetectionParams detectorParams;
    FileNode detectorParamsFn = config["detector_params"];
    if (detectorParamsFn.empty())
    {
        cout << "Error. detection_params tag is missed in cofig file" << endl;
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

    // configure nonmaximum suppressor
    FileNode nmsFn = config["nonmaximum_suppressor"];
    Ptr<NonMaximumSuppressor> nms = 0;
    if (!nmsFn.empty())
    {
        FileNode nmsType = nmsFn["name"];
        CV_Assert(!nmsType.empty());
        string nmsName;
        nmsType >> nmsName;
        nms = Algorithm::create<NonMaximumSuppressor>(nmsName);
        nms->read(nmsFn);
        cout << nmsName << " nonmaximum suppressor is used" << endl;
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

    cout << "training classifier..." << flush;
    classifier->train(samples.rowRange(0, samplesNum),
                      responses.rowRange(0, samplesNum));
    cout << "done" << endl;
    printClassificationError(classifier, samples.rowRange(0, samplesNum),
                             responses.rowRange(0, samplesNum));

    classifier->saveModel(classifierModelFile + "-0.yml", classifierModelName);

    for (int i = 1; i <= cmdParser.get<int>("bi"); ++i)
    {
        cout << "detecting on background images..." << endl;
        for (size_t j = 0; j < negativeAnn.size(); ++j)
        {
            ImageAnnotation & ann = negativeAnn[j];
            ann.bboxes.clear();
            ann.labels.clear();
            ann.scores.clear();
            for (auto k = ann.sources.begin(); k != ann.sources.end(); ++k)
            {
                fileReader->addSource(k->first, k->second);
            }
            fileReader->open();
            CV_Assert(fileReader->grab());
            SourcesMap sources;
            fileReader->retrieve(sources, features);
            fileReader->release();
            cout << "detecting on "
                 << ann.sources.at(DataTypeTime("image", 0))
                 << "... " << flush;
            detect(features, classifier, sources, detectorParams, ann);
            cout << ann.bboxes.size() << " raw detections ... " << flush;

            if (nms != 0)
            {
                (*nms)(ann.bboxes, ann.scores);
            }
            cout << ann.bboxes.size() << " filtered detections" << endl;

            for (size_t k = 0; k < ann.labels.size(); ++k)
            {
                ann.labels[k] = 0;
            }
            ann.addBBoxesBorder(detectorParams.winSize,
                                detectorParams.winBorder);
        }
        cout << "done" << endl;

        if (0 < cmdParser.get<int>("samples"))
        {
            cout << "draw random sample from false detections..." << flush;
            getBootstrapSample(negativeAnn, cmdParser.get<int>("samples"));
            cout << "done" << endl;
        }

        cout << "extracting samples..." << flush;
        extractSamples(fileReader, negativeAnn, features, samples,
                       responses, samplesNum, false);
        cout << "done" << endl;
        cout << "dataset size: " << samplesNum << " x " << samples.cols << endl;

        if (cmdParser.get<string>("dump") != "")
        {
            stringstream ss;
            ss << cmdParser.get<string>("dump") << "-" << i << ".csv";
            cout << "saving dataset to " << ss.str() << "..." << flush;
            dumpDatasetCSV(samples.rowRange(0, samplesNum),
                           responses.rowRange(0, samplesNum),
                           ss.str());
            cout << "done" << endl;
        }

        cout << "training classifier..." << flush;
        classifier->train(samples.rowRange(0, samplesNum),
                          responses.rowRange(0, samplesNum));
        cout << "done" << endl;
        printClassificationError(classifier, samples.rowRange(0, samplesNum),
                                 responses.rowRange(0, samplesNum));

        stringstream ss;
        ss << classifierModelFile << "-" << i << ".yml";
        classifier->saveModel(ss.str(), classifierModelName);
    }
    return 0;
}
