#pragma once

#include "swod/feature.hpp"
#include "opencv2/core/core.hpp"
#include <map>
#include <vector>
#include <string>


class Features
{
public:
    ~Features();
    int getTotalFeatureVectorLength() const;
    void getTotalFeatureVector(int positionX,
                               int positionY,
                               cv::Mat & featureVector) const;
    cv::Size getNumOfSpatialSteps() const;
    void computeOnNewImage(const SourcesMap & sources);
    void computeOnNewScale(const float scale);
    virtual void getROIDescription(cv::Mat & featureDescription,
                                   const SourcesMap & sources,
                                   const cv::Rect & roi);
    virtual void getROIDescription(cv::Mat & featureDescription,
                                   const SourcesMap & sources,
                                   const std::vector<cv::Rect> & roi);

    std::vector<cv::Ptr<Feature> > featuresSet;
};
