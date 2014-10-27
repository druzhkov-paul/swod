#pragma once

#include "swod/data_provider.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <map>

class ImageFileReader : public DataProvider
{
public:
    ImageFileReader();
    ~ImageFileReader();
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

    virtual void addSource(std::string dataType,
                           int timeOffset,
                           std::string path);
    virtual void addSource(DataTypeTime data,
                           std::string path);

    cv::AlgorithmInfo * info() const;

private:
    std::map<DataTypeTime, std::string> sourcesPaths;
    std::map<DataTypeTime, cv::Mat> sources;
    bool isOpenedFlag;
};
