#pragma once

#include "swod/type.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <map>
#include <vector>
#include <string>


class Feature : public cv::Algorithm
{
public:
    virtual ~Feature() {}
    virtual cv::Size getNumOfSpatialSteps() const = 0;
    virtual void getFeatureVector(int positionX,
                                  int positionY,
                                  cv::Mat & featureVector) const = 0;
    virtual int getFeatureVectorLength() const = 0;
    virtual void computeOnNewImage(const SourcesMap & sources) = 0;
    virtual void computeOnNewScale(const float scale) = 0;
    virtual void getROIDescription(cv::Mat & featureDescription,
                                   const SourcesMap & sources,
                                   const cv::Rect & roi) = 0;
    virtual void getROIDescription(cv::Mat & featureDescription,
                                   const SourcesMap & sources,
                                   const std::vector<cv::Rect> & roi) = 0;
    virtual std::vector<std::vector<DataTypeTime> > getRequiredSources() const = 0;
};


// TODO: remove this function
void addBorder(const cv::Mat & src,
               cv::Mat & dst,
               const cv::Size & detectionWindowSize,
               const cv::Size & detectionWindowBorder,
               int borderType,
               const cv::Scalar & value = cv::Scalar());

void getROI(const cv::Mat & src,
            cv::Mat & dst,
            const cv::Rect & roi,
            const int borderType = cv::BORDER_CONSTANT,
            const cv::Scalar & value = cv::Scalar());


bool initFeatures();
