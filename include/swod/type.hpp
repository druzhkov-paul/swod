#pragma once

#include "opencv2/core/core.hpp"
#include <map>
#include <string>


typedef std::pair<std::string, int> DataTypeTime;
typedef std::map<DataTypeTime, cv::Mat> SourcesMap;
