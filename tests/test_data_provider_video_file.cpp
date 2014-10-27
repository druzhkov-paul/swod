#include "test_utils.hpp"
#include "swod/swod.hpp"
#include <cstdio>

using namespace std;
using namespace cv;


TEST(swod, DPVideoFileCreateInstance)
{
    initDataProviders();
    Ptr<Algorithm> videoFile =
            Algorithm::create<Algorithm>("SWOD.DataProvider.VideoFile");
    EXPECT_FALSE(videoFile.empty());
}


TEST(swod, DISABLED_DPVideoFileOpenVideo)
{
    initDataProviders();
    Ptr<DataProvider> videoFile =
            Algorithm::create<DataProvider>("SWOD.DataProvider.VideoFile");
    EXPECT_FALSE(videoFile.empty());

    videoFile->set("maxTimeCacheDepth", 2);
    string testVideoFileName = "./testdata/test_video_sample.avi";
    videoFile->set("videoFilePath", testVideoFileName);

    EXPECT_EQ(2, videoFile->get<int>("maxTimeCacheDepth"));
    EXPECT_EQ(0, testVideoFileName.compare(videoFile->get<string>("videoFilePath")));

    videoFile->open();
    EXPECT_TRUE(videoFile->isOpened());

    int framesCounter = 0;
    for (; videoFile->grab(); ++framesCounter)
    {
        Mat image;
        EXPECT_TRUE(videoFile->retrieve(image, "image", 0));
        EXPECT_FALSE(image.empty());
    }
    EXPECT_EQ(150, framesCounter);

    videoFile->release();
}
