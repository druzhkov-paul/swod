#pragma once

#include "swod/nms.hpp"


class MeanshiftNMS : public NonMaximumSuppressor
{
public:
    MeanshiftNMS();
    ~MeanshiftNMS();
    void operator() (std::vector<cv::Rect> & bboxes,
                     std::vector<float> & scores) const;

    cv::AlgorithmInfo * info() const;

private:
    float threshold;
    cv::Size baseObjectSize;
    cv::Point3f kernelDiag;
};
