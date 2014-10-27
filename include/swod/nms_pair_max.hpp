#pragma once

#include "swod/nms.hpp"


class PairMaxNMS : public NonMaximumSuppressor
{
public:
    PairMaxNMS();
    ~PairMaxNMS();
    void operator() (std::vector<cv::Rect> & bboxes,
                     std::vector<float> & scores) const;

    cv::AlgorithmInfo * info() const;

private:
    float threshold;
};
