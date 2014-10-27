#include "test_utils.hpp"
#include "swod/swod.hpp"
#include <cstdio>

using namespace std;
using namespace cv;


TEST(swod, piotrHOGCreateInstance)
{
    initFeatures();
    Ptr<Algorithm> piotrHOG =
            Algorithm::create<Algorithm>("SWOD.Feature.PiotrHOG");
    EXPECT_FALSE(piotrHOG.empty());
}


TEST(swod, piotrHOGSizes)
{
    initFeatures();
    Ptr<PiotrHOG> piotrHOG =
            Algorithm::create<PiotrHOG>("SWOD.Feature.PiotrHOG");
    EXPECT_FALSE(piotrHOG.empty());

    PiotrHOGParams params;
    params.winSizeH = 96;
    params.winSizeW = 48;
    params.orientBins = 9;
    params.spatialStride = 8;
    params.featureSubsetType = PiotrHOG::ALL_FEATURES;
    params.mask = Mat();
    piotrHOG->setParams(params);

    EXPECT_EQ(48, piotrHOG->get<int>("winSizeW"));
    EXPECT_EQ(96, piotrHOG->get<int>("winSizeH"));
    EXPECT_EQ(9, piotrHOG->get<int>("orientBins"));
    EXPECT_EQ(8, piotrHOG->get<int>("spatialStride"));
    EXPECT_EQ((int)PiotrHOG::ALL_FEATURES, piotrHOG->get<int>("featureSubsetType"));
    EXPECT_TRUE(piotrHOG->get<Mat>("mask").empty());

    SourcesMap srcs;
    srcs[DataTypeTime("image", 0)] = Mat(480, 640, CV_8UC3);
    piotrHOG->computeOnNewImage(srcs);
    piotrHOG->computeOnNewScale(1.0f);
    Size winStepsNum = piotrHOG->getNumOfSpatialSteps();
    EXPECT_EQ(75, winStepsNum.width);
    EXPECT_EQ(49, winStepsNum.height);

    piotrHOG->computeOnNewScale(2.0f);
    winStepsNum = piotrHOG->getNumOfSpatialSteps();
    EXPECT_EQ(35, winStepsNum.width);
    EXPECT_EQ(19, winStepsNum.height);

    // block grid knots * nbins * cells in block
    EXPECT_EQ(4 * 10 * 9 * 4, piotrHOG->getFeatureVectorLength());

    Mat mask(1, piotrHOG->getFeatureVectorLength(), CV_8U, Scalar(0));
    mask.at<uchar>(1) = 1;
    mask.at<uchar>(10) = 1;
    mask.at<uchar>(1234) = 1;
    params.mask = mask;
    params.featureSubsetType = PiotrHOG::SUBSET_BY_MASK;
    piotrHOG->setParams(params);
    piotrHOG->computeOnNewImage(srcs);
    piotrHOG->computeOnNewScale(1.0f);
    EXPECT_EQ(3, piotrHOG->getFeatureVectorLength());
}


TEST(swod, piotrHOGReadWrite)
{
    initFeatures();
    Ptr<PiotrHOG> piotrHOG =
            Algorithm::create<PiotrHOG>("SWOD.Feature.PiotrHOG");
    EXPECT_FALSE(piotrHOG.empty());

    PiotrHOGParams params;
    params.winSizeH = 96;
    params.winSizeW = 48;
    params.orientBins = 9;
    params.spatialStride = 8;
    params.featureSubsetType = PiotrHOG::ALL_FEATURES;
    params.mask = Mat();
    piotrHOG->setParams(params);

    char tempFileName[L_tmpnam];
    ASSERT_EQ(tempFileName, tmpnam(tempFileName));
    string fileName = string(tempFileName) + ".yml";
    FileStorage fs(fileName, FileStorage::WRITE);
    fs << "piotr_hog_feature" << "{";
    piotrHOG->write(fs);
    fs << "}";
    fs.release();

    piotrHOG->setParams(PiotrHOGParams());
    fs.open(fileName, FileStorage::READ);
    EXPECT_FALSE(fs["piotr_hog_feature"].empty());
    piotrHOG->read(fs["piotr_hog_feature"]);
    fs.release();
    remove(fileName.c_str());

    EXPECT_EQ(48, piotrHOG->get<int>("winSizeW"));
    EXPECT_EQ(96, piotrHOG->get<int>("winSizeH"));
    EXPECT_EQ(9, piotrHOG->get<int>("orientBins"));
    EXPECT_EQ(8, piotrHOG->get<int>("spatialStride"));
    EXPECT_EQ((int)PiotrHOG::ALL_FEATURES, piotrHOG->get<int>("featureSubsetType"));
    EXPECT_TRUE(piotrHOG->get<Mat>("mask").empty());
}

// TODO. add regression test
