#pragma once

#include "swod/feature.hpp"

struct RawPixelParams
{
    RawPixelParams();
    RawPixelParams(const RawPixelParams & p);
    RawPixelParams & operator= (const RawPixelParams & p);

    int winSizeW;
    int winSizeH;
    int winStrideW;
    int winStrideH;
};


class RawPixel : public Feature
{
public:
    RawPixel();
    cv::Size getNumOfSpatialSteps() const;
    void getFeatureVector(int detectionWindowIndexX,
                          int detectionWindowIndexY,
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
    void setParams(const RawPixelParams & params);
    void read(const cv::FileNode & fn);
    std::vector<std::vector<DataTypeTime> > getRequiredSources() const;

    cv::AlgorithmInfo * info() const;

private:
    void initDescriptor();

    cv::Mat img;
    cv::Mat scaledImg;
    int featureVectorLength;
    RawPixelParams params;
};