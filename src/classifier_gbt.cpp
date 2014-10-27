#include "swod/classifier_gbt.hpp"

using namespace cv;
using namespace std;


GBTClassifier::GBTClassifier()
    : GBTrees(),
      modelFileName(""),
      modelName("")
{}


int GBTClassifier::predict(const Mat & sample,
                           vector<float> & weights) const
{
    float prediction = GBTrees::predict(sample, Mat(), &weights);
    float score = weights[0] - weights[1];
    weights[0] = score;
    weights[1] = -score;
    return static_cast<int>(prediction);
}


void GBTClassifier::train(const Mat & dataset,
                          const Mat & responses)
{
    GBTrees::train(dataset, CV_ROW_SAMPLE, responses, Mat(), Mat(), Mat(), Mat(), params);
}


void GBTClassifier::saveModel(string filePath,
                              string modelName) const
{
    GBTrees::save(filePath.c_str(), modelName.c_str());
}


void GBTClassifier::loadModel(string filePath,
                              string modelName)
{
    GBTrees::load(filePath.c_str(), modelName.c_str());
}


void GBTClassifier::setParams(const CvGBTreesParams & p)
{
    params = p;
}
