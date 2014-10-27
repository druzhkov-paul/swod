#include "test_utils.hpp"
#include "swod/swod.hpp"
#include <cstdio>

using namespace std;
using namespace cv;


TEST(swod, ClassifierGBTCreateInstance)
{
    initClassifiers();
    Ptr<Algorithm> gbt =
            Algorithm::create<Algorithm>("SWOD.Classifier.GBT");
    EXPECT_FALSE(gbt.empty());
}


TEST(swod, ClassifierGBTReadWriteParams)
{
    Ptr<Algorithm> gbt =
            Algorithm::create<Algorithm>("SWOD.Classifier.GBT");
    EXPECT_FALSE(gbt.empty());

    char tempFileName[L_tmpnam];
    ASSERT_EQ(tempFileName, tmpnam(tempFileName));
    string fileName = string(tempFileName) + ".нml";

    FileStorage outFs(fileName, FileStorage::WRITE);
    outFs << "gbt" << "{";
    gbt->write(outFs);
    outFs << "}";
    outFs.release();

    FileStorage inFs(fileName, FileStorage::READ);
    gbt->read(inFs["gbt"]);
    inFs.release();

    remove(fileName.c_str());
}
