#pragma once

#include "swod/feature.hpp"


struct HueParams
{
    HueParams();

    cv::Size winSize;
    cv::Size spatialStride;
    cv::Size blockSize;
    cv::Size blockStride;
    int bins;
};


class Hue : public Feature
{
public:
    Hue();
    cv::Size getNumOfSpatialSteps() const;
    void getFeatureVector(int positionX,
                          int positionY,
                          cv::Mat & featureVector) const;
    int getFeatureVectorLength() const;
    void computeOnNewImage(const SourcesMap & sources);
    void computeOnNewScale(const float scale);
    void getROIDescription(cv::Mat & featureDescription,
                           const SourcesMap & sources,
                           const cv::Rect & roi);
    void getROIDescription(cv::Mat & featureDescription,
                           const SourcesMap & sources,
                           const std::vector<cv::Rect> & roi);
    void setParams(const HueParams & params);
    std::vector<std::vector<DataTypeTime> > getRequiredSources() const;
    
    cv::AlgorithmInfo * info() const;

private:

    cv::Mat image;
    cv::Mat sourceImage;
    int featureVectorLength;
    std::vector<cv::Rect> roi;

    HueParams params;
};
