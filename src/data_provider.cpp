#include "swod/data_provider.hpp"

using namespace std;
using namespace cv;


bool DataProvider::retrieve(SourcesMap & sources,
                            const Features & features) const
{
    sources.clear();
    std::set<DataTypeTime> availableData = getAvailableSourcesList();
    for (auto i = features.featuresSet.begin(); i != features.featuresSet.end(); ++i)
    {
        bool haveSources = true;
        vector<vector<DataTypeTime> > requiredData = (*i)->getRequiredSources();
        for (size_t j = 0; j < requiredData.size(); ++j)
        {
            haveSources = true;
            // check if configuration available
            for (size_t k = 0; k < requiredData[j].size(); ++k)
            {
                haveSources = haveSources &&
                              (availableData.find(requiredData[j][k]) != availableData.end());
            }
            if (haveSources)
            {
                // load data
                for (size_t k = 0; k < requiredData[j].size(); ++k)
                {
                    DataTypeTime sourceType = requiredData[j][k];
                    Mat src;
                    retrieve(src, sourceType);
                    sources[sourceType] = src;
                }
                break;
            }
        }
        if (!haveSources)
        {
            return false;
        }
    }
    return true;
}
