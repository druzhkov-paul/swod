#include "swod/swod.hpp"
#include "opencv2/core/internal.hpp"

using namespace cv;


CV_INIT_ALGORITHM(VideoProvider, "SWOD.DataProvider.VideoFile",
                  obj.info()->addParam(obj, "maxTimeCacheDepth", obj.maxTimeCacheDepth, false);
                  obj.info()->addParam(obj, "videoFilePath", obj.videoFilePath, false));

CV_INIT_ALGORITHM(ImageFileReader, "SWOD.DataProvider.ImageFileReader",);

bool initDataProviders()
{
    Ptr<Algorithm> videoProvider = createVideoProvider();
    Ptr<Algorithm> fileReader = createImageFileReader();
    return videoProvider->info() != 0 &&
                                    fileReader->info() != 0;
}
