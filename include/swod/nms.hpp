#pragma once

#include "opencv2/core/core.hpp"
#include <vector>


class NonMaximumSuppressor : public cv::Algorithm
{
public:
    virtual ~NonMaximumSuppressor() {}
    virtual void operator() (std::vector<cv::Rect> & bboxes,
                             std::vector<float> & scores) const = 0;
};


bool initNonMaximumSuppressors();
