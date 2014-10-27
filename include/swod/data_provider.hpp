#pragma once

#include "swod/type.hpp"
#include "swod/multiple_features.hpp"
#include "opencv2/core/core.hpp"
#include <set>
#include <string>


class DataProvider : public cv::Algorithm
{
public:
    virtual ~DataProvider() {}
    virtual void open() = 0;
    virtual void release() = 0;
    virtual bool isOpened() const = 0;
    virtual bool grab() = 0;
    virtual bool retrieve(cv::Mat & src,
                          std::string dataType,
                          int timeOffset) const = 0;
    virtual bool retrieve(cv::Mat & src,
                          DataTypeTime data) const = 0;
    virtual bool retrieve(SourcesMap & sources,
                          const Features & features) const;
    virtual int getTimeCacheDepth() const = 0;
    virtual std::set<DataTypeTime> getAvailableSourcesList() const = 0;
};

bool initDataProviders();
