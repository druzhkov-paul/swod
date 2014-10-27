#include "swod/data_provider_file_reader.hpp"

using namespace cv;
using namespace std;



ImageFileReader::ImageFileReader()
    : isOpenedFlag(false)
{}


ImageFileReader::~ImageFileReader()
{
    release();
}


void ImageFileReader::open()
{
    isOpenedFlag = true;
    for (auto i = sourcesPaths.begin(); i != sourcesPaths.end(); ++i)
    {
        Mat image = imread(i->second);
        sources[i->first] = image;
        isOpenedFlag = isOpenedFlag && !image.empty();
    }
}


void ImageFileReader::release()
{
    sourcesPaths.clear();
    sources.clear();
    isOpenedFlag = false;
}


bool ImageFileReader::isOpened() const
{
    return isOpenedFlag;
}


bool ImageFileReader::grab()
{
    return true;
}


bool ImageFileReader::retrieve(Mat & src,
                               string dataType,
                               int timeOffset) const
{
    return retrieve(src, DataTypeTime(dataType, timeOffset));
}


bool ImageFileReader::retrieve(cv::Mat & src,
                               DataTypeTime data) const
{
    if (!sources.count(data))
    {
        return false;
    }
    src = sources.at(data);
    return true;
}


bool ImageFileReader::retrieve(SourcesMap & sources,
                               const Features & features) const
{
    return DataProvider::retrieve(sources, features);
}


int ImageFileReader::getTimeCacheDepth() const
{
    return 0;
}


set<DataTypeTime> ImageFileReader::getAvailableSourcesList() const
{
    std::set<DataTypeTime> sourcesList;
    for (auto i = sources.begin(); i != sources.end(); ++i)
    {
        sourcesList.insert(i->first);
    }
    return sourcesList;
}


void ImageFileReader::addSource(string dataType,
                                int timeOffset,
                                string path)
{
    sourcesPaths[DataTypeTime(dataType, timeOffset)] = path;
}


void ImageFileReader::addSource(DataTypeTime data,
                                std::string path)
{
    sourcesPaths[data] = path;
}
