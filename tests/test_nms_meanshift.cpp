#include "test_utils.hpp"
#include "swod/swod.hpp"
#include "swod/timing.hpp"
#include <cstdio>

using namespace std;
using namespace cv;

/*
TEST(swod, NMSMeanshift)
{
    initNonMaximumSuppressors();
    Ptr<NonMaximumSuppressor> ms =
            Algorithm::create<NonMaximumSuppressor>("SWOD.NMS.Meanshift");
    EXPECT_FALSE(ms.empty());

    const string classifierPath = "/home/paul/programs/swod-build/bin/1/classifier-0.yml";
    const string imagePath = "/home/paul/programs/pedestrian detection/data/TUD/TUD-MotionPairs/negative/img-002-0.png";
    const DataTypeTime SOURCE_IMAGE("image", 0);
    SourcesMap sources;
    sources[SOURCE_IMAGE] = imread(imagePath);
    EXPECT_FALSE(sources.at(SOURCE_IMAGE).empty());

    initClassifiers();
    Ptr<Classifier> svm = Algorithm::create<Classifier>("SWOD.Classifier.SVM");
    EXPECT_FALSE(svm.empty());
    svm->loadModel(classifierPath, "svm_model");

    initFeatures();
    Ptr<PiotrHOG> hog = Algorithm::create<PiotrHOG>("SWOD.Feature.PiotrHOG");
    EXPECT_FALSE(hog.empty());
    PiotrHOGParams paramsHOG;
    paramsHOG.winSizeH = 128;
    paramsHOG.winSizeW = 64;
    paramsHOG.orientBins = 9;
    paramsHOG.spatialStride = 8;
    paramsHOG.featureSubsetType = PiotrHOG::SUBSET_10;
    paramsHOG.mask = Mat();
    hog->setParams(paramsHOG);
    Features features;
    features.featuresSet.push_back(hog);

    DetectionParams params;
    params.maxObjectHeight = -1;
    params.minObjectHeight = 50;
    params.scaleStep = 1.1f;
    params.spatialStride = 8;
    params.winBorder = Size(16, 16);
    params.winSize = Size(64, 128);


    ImageAnnotation ann;
    ann.sources[SOURCE_IMAGE] = imagePath;
    detect(features, svm, sources, params, ann);

    cout << "raw detections: " << ann.bboxes.size() << endl;

    Mat & im = sources.at(SOURCE_IMAGE);
    for (size_t i = 0; i < ann.bboxes.size(); ++i)
    {
        rectangle(im, ann.bboxes[i], CV_RGB(255, 0, 0));
    }
    imshow("tmp1", im);


    TIMER_START(MEANSHIFT);
    (*ms)(ann.bboxes, ann.scores);
    TIMER_END(MEANSHIFT);
    cout << "filtered detections: " << ann.bboxes.size() << endl;

    for (size_t i = 0; i < ann.bboxes.size(); ++i)
    {
        rectangle(im, ann.bboxes[i], CV_RGB(0, 255, 0));
    }
    imshow("tmp2", im);
    waitKey();


}


TEST(swod, NMSMeanshiftNew)
{
    initNonMaximumSuppressors();
    Ptr<NonMaximumSuppressor> ms =
            Algorithm::create<NonMaximumSuppressor>("SWOD.NMS.Meanshift");
    EXPECT_FALSE(ms.empty());

    const string imagePath = "/home/paul/programs/pedestrian detection/data/TUD/TUD-MotionPairs/negative/img-002-0.png";
    const DataTypeTime SOURCE_IMAGE("image", 0);
    SourcesMap sources;
    sources[SOURCE_IMAGE] = imread(imagePath);
    EXPECT_FALSE(sources.at(SOURCE_IMAGE).empty());

    ImageAnnotation ann;
    ann.sources[SOURCE_IMAGE] = imagePath;
    ann.addBBox(Rect(100, 200, 2 * 32, 2 * 96), 1, 1.0f);
    ann.addBBox(Rect(350, 50, 1.3f * 32, 1.3f * 96), 2, 2.0f);

    cout << "raw detections: " << ann.bboxes.size() << endl;
    Mat & im = sources.at(SOURCE_IMAGE);
    for (size_t i = 0; i < ann.bboxes.size(); ++i)
    {
        rectangle(im, ann.bboxes[i], CV_RGB(255, 0, 0));
    }
    imshow("tmp1", im);

    (*ms)(ann.bboxes, ann.scores);
    cout << "filtered detections: " << ann.bboxes.size() << endl;

    for (size_t i = 0; i < ann.bboxes.size(); ++i)
    {
        rectangle(im, ann.bboxes[i], CV_RGB(0, 255, 0));
    }
    imshow("tmp2", im);
    waitKey();
}
*/
