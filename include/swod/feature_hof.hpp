#pragma once

#include "swod/feature.hpp"
#include "opencv2/video/video.hpp"


struct HOFParams
{
    HOFParams();
    HOFParams(const HOFParams & p);
    HOFParams & operator= (const HOFParams & p);

    int winSizeW;
    int winSizeH;
    int winStrideW;
    int winStrideH;
    int blockSizeW;
    int blockSizeH;
    int blockStrideW;
    int blockStrideH;
    int orientBins;
    double opticalFlowEpsilon;
    float l2HysThreshold;
    cv::Mat mask;
};


class HOF : public Feature
{
public:
    HOF();
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
    void setParams(const HOFParams & params);
    void read(const cv::FileNode & fn);
    std::vector<std::vector<DataTypeTime> > getRequiredSources() const;

    cv::AlgorithmInfo * info() const;
    cv::Mat visualizeOpticalFlow() const;

private:
    void initDescriptor();

    cv::Ptr<cv::DenseOpticalFlow> opticalFlowComputer;
    cv::Mat secondImage;
    cv::Mat flow;
    cv::Mat hof;
    cv::Size winSizeInBlocks;
    cv::Size winGridSize;
    int featureVectorLength;

    HOFParams params;
};
