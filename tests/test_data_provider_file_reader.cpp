#include "test_utils.hpp"
#include "swod/swod.hpp"
#include <cstdio>

using namespace std;
using namespace cv;


TEST(swod, DPFileReaderCreateInstance)
{
    initDataProviders();
    Ptr<Algorithm> fileReader =
            Algorithm::create<Algorithm>("SWOD.DataProvider.ImageFileReader");
    EXPECT_FALSE(fileReader.empty());
}


TEST(swod, DPFileReaderOpen)
{
    initDataProviders();
    Ptr<ImageFileReader> fileReader =
            Algorithm::create<ImageFileReader>("SWOD.DataProvider.ImageFileReader");
    EXPECT_FALSE(fileReader.empty());

    fileReader->addSource("image", 0, "./testdata/test_image_1.png");
    fileReader->addSource("image", -1, "./testdata/test_image_2.png");
    fileReader->open();
    EXPECT_TRUE(fileReader->isOpened());

    Mat image1, image2;
    fileReader->grab();
    EXPECT_TRUE(fileReader->retrieve(image1, "image", -1));
    EXPECT_TRUE(fileReader->retrieve(image2, "image", 0));
    EXPECT_FALSE(image1.empty());
    EXPECT_FALSE(image2.empty());

    EXPECT_FALSE(fileReader->retrieve(image1, "nonimage", 0));
    EXPECT_FALSE(fileReader->retrieve(image1, "image", -2));
}
