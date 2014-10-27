#include "test_utils.hpp"
#include "swod/swod.hpp"
#include <cstdio>

using namespace std;
using namespace cv;


TEST(swod, ClassifierSVMCreateInstance)
{
    initClassifiers();
    Ptr<Algorithm> svm =
            Algorithm::create<Algorithm>("SWOD.Classifier.SVM");
    EXPECT_FALSE(svm.empty());
}

