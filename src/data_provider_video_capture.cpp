#include "swod/data_provider_video_capture.hpp"

using namespace cv;
using namespace std;


VideoProvider::VideoProvider()
    : maxTimeCacheDepth(2),
      historyIdx(-1)
{
	channelsMap["image"] = 0;
}


VideoProvider::~VideoProvider()
{
    release();
}


void VideoProvider::open()
{
    capture.open(videoFilePath);
}


void VideoProvider::release()
{
    capture.release();
    history.clear();
}


bool VideoProvider::isOpened() const
{
    return capture.isOpened();
}


bool VideoProvider::grab()
{
    bool result = capture.grab();

    ++historyIdx;
    historyIdx = historyIdx % maxTimeCacheDepth;
    map<string, Mat> srcs;
    for (auto i = channelsMap.begin(); i != channelsMap.end(); ++i)
    {
        Mat image;
        capture.retrieve(image, i->second);
        srcs[i->first] = image;
    }
    if (static_cast<size_t>(historyIdx) == history.size())
    {
        history.push_back(srcs);
    }
    else
    {
        history.at(historyIdx) = srcs;
    }
    return result;
}


bool VideoProvider::retrieve(Mat & src,
                             string dataType,
                             int timeOffset) const
{
    CV_Assert(timeOffset <= 0);
    CV_Assert(-timeOffset < maxTimeCacheDepth);
    int n = (historyIdx + timeOffset + maxTimeCacheDepth) % maxTimeCacheDepth;
    CV_Assert(static_cast<size_t>(n) < history.size());
    CV_Assert(history[n].count(dataType));
    src = history[n].at(dataType);
    return true;
}


bool VideoProvider::retrieve(cv::Mat & src,
                             DataTypeTime data) const
{
    return retrieve(src, data.first, data.second);
}


bool VideoProvider::retrieve(SourcesMap & sources,
                             const Features & features) const
{
    return DataProvider::retrieve(sources, features);
}


int VideoProvider::getTimeCacheDepth() const
{
    return maxTimeCacheDepth;
}


set<DataTypeTime> VideoProvider::getAvailableSourcesList() const
{
    std::set<DataTypeTime> sourcesList;
    for (auto i = channelsMap.begin(); i != channelsMap.end(); ++i)
    {
        for (int j = 0; j < maxTimeCacheDepth; ++j)
        {
            sourcesList.insert(DataTypeTime(i->first, j));
        }
    }
    return sourcesList;
}
