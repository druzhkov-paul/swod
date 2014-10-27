#pragma once

#include "swod/annotation.hpp"
#include "swod/classifier.hpp"
#include "swod/multiple_features.hpp"
#include <string>
#include <map>


struct DetectionParams
{
    DetectionParams();
    virtual void write(cv::FileStorage & fs) const;
    virtual void read(const cv::FileNode & fn);

    cv::Size winSize;
    cv::Size winBorder;
    int spatialStride;
    int minObjectHeight;
    int maxObjectHeight;
    float scaleStep;
};


void write(cv::FileStorage& fs,
           const std::string &,
           const DetectionParams & x);

void read(const cv::FileNode & fn,
          DetectionParams & x,
          const DetectionParams & defaultValue = DetectionParams());


void detect(Features & features,
            const cv::Ptr<Classifier> classifier,
            const SourcesMap & sources,
            const DetectionParams & params,
            ImageAnnotation & ann);
