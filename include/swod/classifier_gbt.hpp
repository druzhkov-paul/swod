#pragma once

#include "swod/classifier.hpp"
#include "gbt.hpp"


class GBTClassifier : public Classifier, private GBTrees
{
public:
    GBTClassifier();
    int predict(const cv::Mat & sample,
                std::vector<float> & weights) const;
    void train(const cv::Mat & dataset,
               const cv::Mat & responses);
    void saveModel(std::string filePath,
                   std::string modelName) const;
    void loadModel(std::string filePath,
                   std::string modelName);
    void setParams(const CvGBTreesParams & p);

    cv::AlgorithmInfo * info() const;

private:
    std::string modelFileName;
    std::string modelName;
};
