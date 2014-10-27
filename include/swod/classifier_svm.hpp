#pragma once

#include "swod/classifier.hpp"
#include "opencv2/ml/ml.hpp"


class SVMClassifier : public Classifier, private CvSVM
{
public:
    SVMClassifier();
    int predict(const cv::Mat & sample,
                std::vector<float> & weights) const;
    void train(const cv::Mat & dataset,
               const cv::Mat & responses);
    void saveModel(std::string filePath,
                   std::string modelName) const;
    void loadModel(std::string filePath,
                   std::string modelName);
    void setParams(const CvSVMParams & p);

    cv::AlgorithmInfo * info() const;

private:
    int crossValidationFolds;
    std::string modelFileName;
    std::string modelName;
};
