#pragma once

#include "opencv2/core/core.hpp"

namespace piotrhog
{
    cv::Mat hog(const cv::Mat & image, int sBin = 8, int oBin = 9);
    int getHogDescriptorSize(int detWinWidth = 64, int detWinHeight = 128,
                             int sBin = 8, int oBin = 9);
}
