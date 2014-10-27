#include "swod/feature.hpp"

using namespace std;
using namespace cv;

void addBorder(const Mat & src,
               Mat & dst,
               const Size & detectionWindowSize,
               const Size & detectionWindowBorder,
               int borderType,
               const Scalar & value)
{
    float borderFractionWidth = static_cast<float>(detectionWindowBorder.width) /
                                (detectionWindowSize.width - 2 * detectionWindowBorder.width);
    float borderFractionHeight = static_cast<float>(detectionWindowBorder.height) /
                                 (detectionWindowSize.height - 2 * detectionWindowBorder.height);

    int borderX = static_cast<int>(borderFractionWidth * src.cols);
    int borderY = static_cast<int>(borderFractionHeight * src.rows);

    copyMakeBorder(src, dst, borderY, borderY, borderX, borderX, borderType, value);
}


void getROI(const Mat & src,
            Mat & dst,
            const Rect & roi,
            const int borderType,
            const Scalar & value)
{
    Point topLeft = roi.tl();
    Point bottomRight = roi.br();
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;
    if (topLeft.x < 0)
    {
        left = -topLeft.x;
        topLeft.x = 0;
    }
    if (topLeft.y < 0)
    {
        top = -topLeft.y;
        topLeft.y = 0;
    }
    if (src.cols < bottomRight.x)
    {
        right = bottomRight.x - src.cols;
        bottomRight.x = src.cols;
    }
    if (src.rows < bottomRight.y)
    {
        bottom = bottomRight.y - src.rows;
        bottomRight.y = src.rows;
    }
    copyMakeBorder(src(Rect(topLeft, bottomRight)), dst,
                   top, bottom, left, right, borderType, value);
}
