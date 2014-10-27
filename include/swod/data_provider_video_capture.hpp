#pragma once

#include "swod/data_provider.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <map>

class VideoProvider : public DataProvider
{
public:
    VideoProvider();
    ~VideoProvider();
    void open();
    void release();
    bool isOpened() const;
    bool grab();
    bool retrieve(cv::Mat & src,
                  std::string dataType,
                  int timeOffset) const;
    bool retrieve(cv::Mat & src,
                  DataTypeTime data) const;
    bool retrieve(SourcesMap & sources,
                  const Features & features) const;
    int getTimeCacheDepth() const;
    std::set<DataTypeTime> getAvailableSourcesList() const;

    cv::AlgorithmInfo * info() const;

private:
    int maxTimeCacheDepth;
    std::string videoFilePath;

    cv::VideoCapture capture;
    std::vector<std::map<std::string, cv::Mat> > history;
    int historyIdx;

    std::map<std::string, int> channelsMap;
};
