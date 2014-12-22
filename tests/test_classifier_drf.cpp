#include "test_utils.hpp"
#include "swod/swod.hpp"
#include <cstdio>
#include <iostream>

using namespace std;
using namespace cv;


TEST(swod, ClassifierDRFCreateInstance)
{
    initClassifiers();
    Ptr<Algorithm> drf =
            Algorithm::create<Algorithm>("SWOD.Classifier.DRF");
    EXPECT_FALSE(drf.empty());
}


TEST(swod, ClassifierDRFReadWriteParams)
{
    Ptr<Algorithm> drf =
            Algorithm::create<Algorithm>("SWOD.Classifier.DRF");
    EXPECT_FALSE(drf.empty());

    char tempFileName[L_tmpnam];
    ASSERT_EQ(tempFileName, tmpnam(tempFileName));
    string fileName = string(tempFileName) + ".yml";

    FileStorage outFs(fileName, FileStorage::WRITE);
    EXPECT_TRUE(outFs.isOpened());
    outFs << "drf" << "{";
    drf->write(outFs);
    outFs << "}";
    outFs.release();

    FileStorage inFs(fileName, FileStorage::READ);
    EXPECT_TRUE(inFs.isOpened());
    EXPECT_FALSE(inFs["drf"].empty());
    drf->read(inFs["drf"]);
    inFs.release();

    remove(fileName.c_str());
}
