#include "test_utils.hpp"
#include "swod/swod.hpp"
#include "swod/timing.hpp"
#include <cstdio>

using namespace std;
using namespace cv;


TEST(swod, opencvHOGCreateInstance)
{
    initFeatures();
    Ptr<Algorithm> opencvHOG =
            Algorithm::create<Algorithm>("SWOD.Feature.OpenCVHOG");
    EXPECT_FALSE(opencvHOG.empty());
}


TEST(swod, opencvHOGSizes)
{
    initFeatures();
    Ptr<OpenCVHOG> opencvHOG =
            Algorithm::create<OpenCVHOG>("SWOD.Feature.OpenCVHOG");
    EXPECT_FALSE(opencvHOG.empty());

    OpenCVHOGParams params;
    params.winSizeW = 48;
    params.winSizeH = 96;
    params.winStrideW = 8;
    params.winStrideH = 8;
    params.blockSizeW = 16;
    params.blockSizeH = 16;
    params.blockStrideW = 8;
    params.blockStrideH = 8;
    params.cellSizeW = 8;
    params.cellSizeH = 8;
    params.nbins = 9;
    params.mask = Mat();
    opencvHOG->setParams(params);

    EXPECT_EQ(48, opencvHOG->get<int>("winSizeW"));
    EXPECT_EQ(96, opencvHOG->get<int>("winSizeH"));

    SourcesMap srcs;
    srcs[DataTypeTime("image", 0)] = Mat(480, 640, CV_8UC3);
    opencvHOG->computeOnNewImage(srcs);
    opencvHOG->computeOnNewScale(1.0f);
    Size winStepsNum = opencvHOG->getNumOfSpatialSteps();
    EXPECT_EQ(75, winStepsNum.width);
    EXPECT_EQ(49, winStepsNum.height);

    opencvHOG->computeOnNewScale(2.0f);
    winStepsNum = opencvHOG->getNumOfSpatialSteps();
    EXPECT_EQ(35, winStepsNum.width);
    EXPECT_EQ(19, winStepsNum.height);

    // block grid knots * nbins * cells in block
    EXPECT_EQ(5 * 11 * 9 * 4, opencvHOG->getFeatureVectorLength());

    Mat mask(1, opencvHOG->getFeatureVectorLength(), CV_8U, Scalar(0));
    mask.at<uchar>(1) = 1;
    mask.at<uchar>(10) = 1;
    mask.at<uchar>(1234) = 1;
    params.mask = mask;
    opencvHOG->setParams(params);
    opencvHOG->computeOnNewImage(srcs);
    opencvHOG->computeOnNewScale(1.0f);
    EXPECT_EQ(3, opencvHOG->getFeatureVectorLength());
}


TEST(swod, opencvHOGReadWrite)
{
    initFeatures();
    Ptr<OpenCVHOG> opencvHOG =
            Algorithm::create<OpenCVHOG>("SWOD.Feature.OpenCVHOG");
    EXPECT_FALSE(opencvHOG.empty());

    OpenCVHOGParams params;
    params.winSizeW = 48;
    params.winSizeH = 96;
    params.winStrideW = 8;
    params.winStrideH = 8;
    params.blockSizeW = 16;
    params.blockSizeH = 16;
    params.blockStrideW = 8;
    params.blockStrideH = 8;
    params.cellSizeW = 8;
    params.cellSizeH = 8;
    params.nbins = 9;
    params.mask = Mat();
    opencvHOG->setParams(params);

    char tempFileName[L_tmpnam];
    ASSERT_EQ(tempFileName, tmpnam(tempFileName));
    string fileName = string(tempFileName) + ".yml";
    FileStorage fs(fileName, FileStorage::WRITE);
    fs << "opencv_hog_feature" << "{";
    opencvHOG->write(fs);
    fs << "}";
    fs.release();

    opencvHOG->setParams(OpenCVHOGParams());
    fs.open(fileName, FileStorage::READ);
    EXPECT_FALSE(fs["opencv_hog_feature"].empty());
    opencvHOG->read(fs["opencv_hog_feature"]);
    fs.release();
    remove(fileName.c_str());

    EXPECT_EQ(48, opencvHOG->get<int>("winSizeW"));
    EXPECT_EQ(96, opencvHOG->get<int>("winSizeH"));
    EXPECT_EQ(8, opencvHOG->get<int>("winStrideW"));
    EXPECT_EQ(8, opencvHOG->get<int>("winStrideH"));
    EXPECT_EQ(16, opencvHOG->get<int>("blockSizeW"));
    EXPECT_EQ(16, opencvHOG->get<int>("blockSizeH"));
    EXPECT_EQ(8, opencvHOG->get<int>("blockStrideW"));
    EXPECT_EQ(8, opencvHOG->get<int>("blockStrideH"));
    EXPECT_EQ(8, opencvHOG->get<int>("cellSizeW"));
    EXPECT_EQ(8, opencvHOG->get<int>("cellSizeH"));
    EXPECT_EQ(9, opencvHOG->get<int>("nbins"));
    EXPECT_TRUE(opencvHOG->get<Mat>("mask").empty());
}

// TODO. add regression test


TEST(swod, opencvHOGDetect)
{
    initFeatures();
    Ptr<OpenCVHOG> opencvHOG =
            Algorithm::create<OpenCVHOG>("SWOD.Feature.OpenCVHOG");
    EXPECT_FALSE(opencvHOG.empty());

    OpenCVHOGParams params;
    params.winSizeW = 64;
    params.winSizeH = 128;
    params.winStrideW = 8;
    params.winStrideH = 8;
    params.blockSizeW = 16;
    params.blockSizeH = 16;
    params.blockStrideW = 8;
    params.blockStrideH = 8;
    params.cellSizeW = 8;
    params.cellSizeH = 8;
    params.nbins = 9;
    params.mask = Mat();
    opencvHOG->setParams(params);
    EXPECT_EQ(3780, opencvHOG->getFeatureVectorLength());

    Features features;
    features.featuresSet.push_back(opencvHOG);

    initClassifiers();
    Ptr<Classifier> svm = Algorithm::create<Classifier>("SWOD.Classifier.SVM");
    EXPECT_FALSE(svm.empty());
    const string modelPath = "./testdata/linsvm_classifier.yml";
    const string modelName = "svm_model";
    svm->loadModel(modelPath, modelName);

    const string imagePath = "./testdata/test_image_1.png";
    SourcesMap sources;
    Mat image = imread(imagePath);
    EXPECT_FALSE(image.empty());
    const DataTypeTime SOURCE_IMAGE("image", 0);
    sources[SOURCE_IMAGE] = image;

    DetectionParams detectorParams;
    detectorParams.minObjectHeight = 50;
    detectorParams.maxObjectHeight = -1;
    detectorParams.scaleStep = 1.1f;
    detectorParams.winSize = Size(64, 128);
    detectorParams.winBorder = Size(16, 16);
    detectorParams.spatialStride = 8;

    ImageAnnotation ann;
    ann.sources[SOURCE_IMAGE] = imagePath;
    detect(features, svm, sources, detectorParams, ann);
}
