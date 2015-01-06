#include "swod/classifier_svm.hpp"

using namespace cv;
using std::vector;
using std::string;


SVMClassifier::SVMClassifier()
    : CvSVM(),
      crossValidationFolds(5),
      modelFileName(""),
      modelName("")
{}


int SVMClassifier::predict(const Mat & sample,
                           vector<float> & weights) const
{
    int classNum = class_labels->cols;
    weights.assign(classNum, 0.0f);

    float confidence = CvSVM::predict(sample, true);
    int prediction = 0;
    if (classNum == 2)
    {
        weights[0] = confidence;
        weights[1] = -confidence;
        prediction = 0.0f < confidence ? 0 : 1;
    }
    else
    {
        prediction = static_cast<int>(confidence);
        weights[prediction] = 1.0f;
    }
    return prediction;
}


void SVMClassifier::train(const Mat & dataset,
                          const Mat & responses)
{
    train_auto(dataset, responses, Mat(), Mat(), params, crossValidationFolds);
}


void SVMClassifier::saveModel(string filePath,
                              string modelName) const
{
    CvSVM::save(filePath.c_str(), modelName.c_str());
}


void SVMClassifier::loadModel(string filePath,
                              string modelName)
{
    CvSVM::load(filePath.c_str(), modelName.c_str());
}


void SVMClassifier::setParams(const CvSVMParams & p)
{
    set_params(p);
}
