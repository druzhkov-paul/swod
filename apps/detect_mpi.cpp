#include "swod/swod.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/internal.hpp"
#include <iostream>
#include <iomanip>
#include <set>
#include "mpi.h"

using namespace std;
using namespace cv;


int main(int argc, char ** argv)
{
    MPI_Init (&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    initFeatures();
    initClassifiers();
    initDataProviders();
    initNonMaximumSuppressors();

    const string commandLineKeys = "{h|help|false|show help and exit}"
                                   "{c|config| |.xml or .yml file containing detector "
                                   "configuration parameters, i.e. features to used "
                                   "and thier parameters, classifier with the path "
                                   "to the pretrained model, general detection parameters}"
                                   "{o|out|detections.yml|file to save detection results to}"
                                   "{|outname|detections|name of the node to save detections to"
                                   "in the result file}"
                                   "{v|video| |video file on frames of which detection "
                                   "should be done}"
                                   "{a|ann| |annotation of sources to detect on}"
                                   "{|annname|annotation|the name of the annFile node to read from}";
    CommandLineParser cmdParser(argc, argv, commandLineKeys.c_str());
    if (cmdParser.get<bool>("help"))
    {
        cmdParser.printParams();
        return 0;
    }
    CV_Assert(cmdParser.get<string>("config") != "");
    CV_Assert(cmdParser.get<string>("video") != "" || cmdParser.get<string>("ann") != "");

    FileStorage config(cmdParser.get<string>("config"), FileStorage::READ);

    // read detection parameters
    DetectionParams detectorParams;
    FileNode detectorParamsFn = config["detector_params"];
    if (detectorParamsFn.empty())
    {
        cout << "Error. detection_params tag is missed in cofig file" << endl;
        return 2;
    }
    detectorParamsFn >> detectorParams;

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
    string modelFileName = classifierFn["modelFileName"];
    string modelName = classifierFn["modelName"];
    CV_Assert(!modelFileName.empty());
    classifier->loadModel(modelFileName, modelName);

    // configure data providers
    /*
    FileNode dataProviderFn = config["data_provider"];
    FileNode dataProviderType = dataProviderFn["name"];
    CV_Assert(!dataProviderType.empty());
    string dataProviderName;
    dataProviderType >> dataProviderName;
    Ptr<DataProvider> dataProvider = Algorithm::create<DataProvider>(dataProviderName);
    dataProvider->read(dataProviderFn);
    cout << dataProviderName << " data provider is used" << endl;
    */

    // configure nonmaximum suppressor
    FileNode nmsFn = config["nonmaximum_suppressor"];
    FileNode nmsType = nmsFn["name"];
    CV_Assert(!nmsType.empty());
    string nmsName;
    nmsType >> nmsName;
    Ptr<NonMaximumSuppressor> nms = Algorithm::create<NonMaximumSuppressor>(nmsName);
    nms->read(nmsFn);
    cout << nmsName << " nonmaximum suppressor is used" << endl;


    vector<ImageAnnotation> ann;
    if (cmdParser.get<string>("video") != "")
    {
        Ptr<VideoProvider> dataProvider = Algorithm::create<VideoProvider>("SWOD.DataProvider.VideoFile");
        dataProvider->set("videoFilePath", cmdParser.get<string>("video"));
        // FIXME: set proper history size
        dataProvider->set("maxTimeCacheDepth", 2);
        dataProvider->open();
        CV_Assert(dataProvider->isOpened());
        // run detection
        for (size_t i = 0; dataProvider->grab(); ++i)
        {
            ImageAnnotation imageAnn;
            stringstream s;
            s << "frame " << setw(6) << setfill('0') << i;
			imageAnn.sources[DataTypeTime("image", 0)] = s.str();
            SourcesMap sources;
            CV_Assert(dataProvider->retrieve(sources, features));
            detect(features, classifier, sources, detectorParams, imageAnn);
            (*nms)(imageAnn.bboxes, imageAnn.scores);
            ann.push_back(imageAnn);
        }
        dataProvider->release();
    }
    else
    {
        // load annotation (i.e. the list of sources to run detector on)
        loadDatasetAnnotation(cmdParser.get<string>("ann"),
                              cmdParser.get<string>("annname"), ann);
        Ptr<ImageFileReader> fileReader = Algorithm::create<ImageFileReader>("SWOD.DataProvider.ImageFileReader");
        // run detection
        cout << "[" << rank << ":" << size << "] detect from "
            << (ann.size() * rank / size) << " to "
            << (ann.size() * (rank + 1) / size) << " of "
            << ann.size() << endl;
        for (size_t i = ann.size() * rank / size; i < ann.size() * (rank + 1) / size; ++i)
        {
            ImageAnnotation & a = ann[i];
            cout << "detecting on "
                 << a.sources.at(DataTypeTime("image", 0))
                 << "..." << endl;
            for (auto j = a.sources.begin(); j != a.sources.end(); ++j)
            {
                fileReader->addSource(j->first, j->second);
            }
            SourcesMap sources;
            fileReader->open();
            CV_Assert(fileReader->isOpened());
            CV_Assert(fileReader->grab());
            CV_Assert(fileReader->retrieve(sources, features));
            fileReader->release();
            detect(features, classifier, sources, detectorParams, a);
            cout << "raw detections: " << a.bboxes.size() << flush;
            (*nms)(a.bboxes, a.scores);
            // FIXME: make proper processing of labels inside NMS
            a.labels.resize(a.bboxes.size());
            cout << ", filtered detections: " << a.bboxes.size() << endl;
        }
    }

    // save result
    stringstream s;
    s << cmdParser.get<string>("out") << "-" << rank << ".yml";
    saveDatasetAnnotation(s.str(),
                          cmdParser.get<string>("outname"), ann);
    config.release();
    MPI_Finalize();
    return 0;
}
