#pragma once

#include "swod/classifier.hpp"
#include "swod/random_forest.hpp"


struct DRFClassifierParams
{
    DRFClassifierParams();
    int layersNum;
    std::vector<CvRTParams> rfParams;
};


class DRFClassifier : public Classifier
{
public:
    DRFClassifier();
    int predict(const cv::Mat & sample,
                std::vector<float> & weights) const;
    void train(const cv::Mat & dataset,
               const cv::Mat & responses);
    void saveModel(std::string filePath,
                   std::string modelName) const;
    void loadModel(std::string filePath,
                   std::string modelName);
    void setParams(const DRFClassifierParams & p);

    void read(const cv::FileNode & fn);
    void write(cv::FileStorage & fs) const;

    cv::AlgorithmInfo * info() const;

private:
    DRFClassifierParams params;
    std::vector<RandomForest> randomForests;
    std::string modelFileName;
    std::string modelName;
};
