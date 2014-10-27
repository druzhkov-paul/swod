#include "swod/swod.hpp"
#include "opencv2/core/internal.hpp"

using namespace cv;


CV_INIT_ALGORITHM(PairMaxNMS, "SWOD.NMS.PairMax",
                  obj.info()->addParam(obj, "threshold", obj.threshold, false));


CV_INIT_ALGORITHM(MeanshiftNMS, "SWOD.NMS.Meanshift",
                  obj.info()->addParam(obj, "baseObjectWidth", obj.baseObjectSize.width, false);
                  obj.info()->addParam(obj, "baseObjectHeight", obj.baseObjectSize.height, false);
                  obj.info()->addParam(obj, "standardDeviationX", obj.kernelDiag.x, false);
                  obj.info()->addParam(obj, "standardDeviationY", obj.kernelDiag.y, false);
                  obj.info()->addParam(obj, "standardDeviationScale", obj.kernelDiag.z, false);
                  obj.info()->addParam(obj, "threshold", obj.threshold, false));


bool initNonMaximumSuppressors()
{
    Ptr<Algorithm> pairMax = createPairMaxNMS();
    Ptr<Algorithm> meanshift = createMeanshiftNMS();
    return (pairMax->info() != 0) && (meanshift->info() != 0);
}
