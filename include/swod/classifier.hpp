#pragma once

#include "opencv2/core/core.hpp"
#include <vector>
#include <string>


class Classifier : public cv::Algorithm
{
public:
    virtual int predict(const cv::Mat & sample,
                        std::vector<float> & weights) const = 0;
    virtual void train(const cv::Mat & dataset,
                       const cv::Mat & responses) = 0;
    virtual void saveModel(std::string filePath,
                           std::string modelName) const = 0;
    virtual void loadModel(std::string filePath,
                           std::string modelName) = 0;

};

bool initClassifiers();
