#include "test_utils.hpp"
#include "swod/swod.hpp"

using namespace std;
using namespace cv;


TEST(swod, FeatureRawPixelCreateInstance)
{
    initFeatures();
    Ptr<Algorithm> raw =
            Algorithm::create<Algorithm>("SWOD.Feature.RawPixel");
    EXPECT_FALSE(raw.empty());
}


TEST(swod, FeatureRawPixelReadWrite)
{
    initFeatures();
    Ptr<RawPixel> raw =
            Algorithm::create<RawPixel>("SWOD.Feature.RawPixel");
    EXPECT_FALSE(raw.empty());

    RawPixelParams params;
    params.winSizeH = 3;
    params.winSizeW = 5;
    params.winStrideH = 2;
    params.winStrideW = 4;
    raw->setParams(params);

    char tempFileName[L_tmpnam];
    ASSERT_EQ(tempFileName, tmpnam(tempFileName));
    string fileName = string(tempFileName) + ".yml";
    FileStorage fs(fileName, FileStorage::WRITE);
    EXPECT_TRUE(fs.isOpened());
    fs << "raw_pixel_feature" << "{";
    raw->write(fs);
    fs << "}";
    fs.release();

    raw->setParams(RawPixelParams());
    fs.open(fileName, FileStorage::READ);
    EXPECT_FALSE(fs["raw_pixel_feature"].empty());
    raw->read(fs["raw_pixel_feature"]);
    fs.release();
    remove(fileName.c_str());

    EXPECT_EQ(3, raw->get<int>("winSizeH"));
    EXPECT_EQ(5, raw->get<int>("winSizeW"));
    EXPECT_EQ(2, raw->get<int>("winStrideH"));
    EXPECT_EQ(4, raw->get<int>("winStrideW"));
}


TEST(swod, FeatureRawPixelDescription)
{
    initFeatures();
    Ptr<RawPixel> raw =
            Algorithm::create<RawPixel>("SWOD.Feature.RawPixel");
    EXPECT_FALSE(raw.empty());

    RawPixelParams params;
    params.winSizeH = 3;
    params.winSizeW = 5;
    params.winStrideH = 2;
    params.winStrideW = 3;
    raw->setParams(params);

    Mat img(5, 11, CV_8UC1);
    uchar k = 0;
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            img.at<uchar>(i, j) = k++;
        }
    }
    SourcesMap sources;
    sources[DataTypeTime("image", 0)] = img;

    raw->computeOnNewImage(sources);
    raw->computeOnNewScale(1.0f);

    Size grid = raw->getNumOfSpatialSteps();
    EXPECT_EQ(grid.width, 3);
    EXPECT_EQ(grid.height, 2);

    EXPECT_EQ(raw->getFeatureVectorLength(), 15);
    Mat featureVector;
    raw->getFeatureVector(0, 0, featureVector);
    EXPECT_EQ(1, featureVector.rows);
    EXPECT_EQ(15, featureVector.cols);
    EXPECT_EQ(0, featureVector.at<float>(0));
    EXPECT_EQ(26, featureVector.at<float>(14));
}
