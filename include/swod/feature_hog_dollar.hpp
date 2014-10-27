#pragma once

#include "swod/feature.hpp"


struct PiotrHOGParams
{
    PiotrHOGParams();
    PiotrHOGParams(const PiotrHOGParams & p);
    PiotrHOGParams & operator= (const PiotrHOGParams & p);

    int winSizeW;
    int winSizeH;
    int spatialStride;
    int orientBins;
    int featureSubsetType;
    cv::Mat mask;
};


class PiotrHOG : public Feature
{
public:
    enum {ALL_FEATURES = 0, SUBSET_BY_MASK, SUBSET_10};

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
    void setParams(const PiotrHOGParams & params);
    std::vector<std::vector<DataTypeTime> > getRequiredSources() const;
    
    cv::AlgorithmInfo * info() const;

private:
    // data
    cv::Mat img;
    cv::Mat hog;

    // params
    PiotrHOGParams params;
};
