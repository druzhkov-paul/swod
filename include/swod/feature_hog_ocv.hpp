#pragma once

#include "swod/feature.hpp"
#include "opencv2/objdetect/objdetect.hpp"


struct OpenCVHOGParams
{
    OpenCVHOGParams();
    OpenCVHOGParams(const OpenCVHOGParams & p);
    OpenCVHOGParams & operator= (const OpenCVHOGParams & p);

    int winSizeW, winSizeH;
    int winStrideW, winStrideH;
    int blockSizeW, blockSizeH;
    int blockStrideW, blockStrideH;
    int cellSizeW, cellSizeH;
    int nbins;
    int derivAperture;
    double winSigma;
    int histogramNormType;
    double L2HysThreshold;
    bool gammaCorrection;
    int nlevels;
    cv::Mat mask;
};



class OpenCVHOG : public Feature
{
public:
    OpenCVHOG();
    virtual ~OpenCVHOG();
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
    void setParams(const OpenCVHOGParams & params);
    void read(const cv::FileNode & fn);
    std::vector<std::vector<DataTypeTime> > getRequiredSources() const;

    cv::AlgorithmInfo * info() const;

private:

    void initDescriptor();

    // data
    cv::Mat img;
    cv::HOGDescriptor hog;
    std::vector<float> features;
    cv::Mat featuresMat;
    int featureVectorSize;
    int stepsX;
    int stepsY;
    
    // params
    OpenCVHOGParams params;
};
