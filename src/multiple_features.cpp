#include "swod/multiple_features.hpp"
#include <stdio.h>

using namespace cv;
using std::map;
using std::string;
using std::vector;


Features::~Features()
{
    featuresSet.clear();
}


int Features::getTotalFeatureVectorLength() const
{
    int length = 0;
    for (size_t i = 0; i < featuresSet.size(); ++i)
    {
        length += featuresSet[i]->getFeatureVectorLength();
    }
    return length;
}


void Features::getTotalFeatureVector(int positionX,
                                     int positionY,
                                     Mat & fv) const
{
    int len = getTotalFeatureVectorLength();
    fv.create(1, len, CV_32F);
    int pos = 0;
    for (size_t i = 0; i < featuresSet.size(); ++i)
    {
        int n = featuresSet[i]->getFeatureVectorLength();
        Mat fvColRange = fv.colRange(pos, pos + n);
        featuresSet[i]->getFeatureVector(positionX, positionY, fvColRange);
        pos += n;
    }
}


Size Features::getNumOfSpatialSteps() const
{
    CV_Assert(featuresSet.size());
    Size steps = featuresSet[0]->getNumOfSpatialSteps();
    for (size_t i = 1; i < featuresSet.size(); ++i)
    {
        Size s = featuresSet[i]->getNumOfSpatialSteps();
        CV_Assert(s == steps);
    }
    return steps;
}


void Features::computeOnNewImage(const SourcesMap & sources)
{
    for (size_t i = 0; i < featuresSet.size(); ++i)
    {
        featuresSet[i]->computeOnNewImage(sources);
    }
}


void Features::computeOnNewScale(const float scale)
{
    for (size_t i = 0; i < featuresSet.size(); ++i)
    {
        featuresSet[i]->computeOnNewScale(scale);
    }
}


void Features::getROIDescription(Mat & featureDescription,
                                 const SourcesMap & sources,
                                 const Rect & roi)
{
    featureDescription.create(1, getTotalFeatureVectorLength(), CV_32F);
    int k = 0;
    for (size_t i = 0; i < featuresSet.size(); ++i)
    {
        int descriptionLength = featuresSet[i]->getFeatureVectorLength();
        Mat currentFeatureDescription = featureDescription.colRange(k, k + descriptionLength);
        featuresSet[i]->getROIDescription(currentFeatureDescription, sources, roi);
        k += descriptionLength;
    }
}


void Features::getROIDescription(Mat & featureDescription,
                                 const SourcesMap & sources,
                                 const vector<Rect> & roi)
{
    featureDescription.create(roi.size(), getTotalFeatureVectorLength(), CV_32F);
    int k = 0;
    for (size_t i = 0; i < featuresSet.size(); ++i)
    {
        int descriptionLength = featuresSet[i]->getFeatureVectorLength();
        Mat currentFeatureDescription = featureDescription.colRange(k, k + descriptionLength);
        featuresSet[i]->getROIDescription(currentFeatureDescription, sources, roi);
        k += descriptionLength;
    }
}
