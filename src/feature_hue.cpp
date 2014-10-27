#include "swod/feature_hue.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using std::map;
using std::string;
using std::vector;


namespace
{
    DataTypeTime SOURCE_IMAGE("image", 0);
}


HueParams::HueParams()
    : winSize(64, 128),
      spatialStride(8, 8),
      blockSize(16, 16),
      blockStride(8, 8),
      bins(9)
{}


Hue::Hue()
{
    image = Mat();
    sourceImage = Mat();
    featureVectorLength = 0;
}


Size Hue::getNumOfSpatialSteps() const
{
    CV_Assert(!image.empty());
    return Size((image.cols - params.winSize.width) / params.spatialStride.width + 1,
                (image.rows - params.winSize.height) / params.spatialStride.height + 1);
}


void Hue::setParams(const HueParams & p)
{
    params = p;
    image = Mat();
    for (int i = 0; i < params.winSize.height - params.blockSize.height; i += params.blockStride.height)
    {
        for (int j = 0; j < params.winSize.width - params.blockSize.width; j += params.blockStride.width)
        {
            roi.push_back(Rect(j, i, params.blockSize.width, params.blockSize.height));
        }
    }
    featureVectorLength = params.bins * roi.size();
}


void Hue::getFeatureVector(int positionX,
                           int positionY,
                           Mat & featureVector) const
{
    CV_Assert(!image.empty());
    CV_Assert(0 <= positionX);
    CV_Assert(0 <= positionY);
    CV_Assert(positionX * params.spatialStride.width + params.winSize.width <= image.cols);
    CV_Assert(positionY * params.spatialStride.height + params.winSize.height <= image.rows);

    featureVector.create(1, featureVectorLength, CV_32F);
    featureVector = Scalar(0);

    int detWinPosX = positionX * params.spatialStride.width;
    int detWinPosY = positionY * params.spatialStride.height;

    const int maxHueValue = 180;
    int binSize = maxHueValue / params.bins;
    int halfBinSize = binSize / 2;

    for (size_t i = 0; i < roi.size(); ++i)
    {
        Rect r = roi[i];
        Mat fv = featureVector.colRange(i * params.bins, i * params.bins + params.bins);

        for (int y = detWinPosY + r.y; y < detWinPosY + r.y + r.height; ++y)
        {
            for (int x = detWinPosX + r.x; x < detWinPosX + r.x + r.width; ++x)
            {
                CV_DbgAssert(0 <= x && x < image.cols);
                CV_DbgAssert(0 <= y && y < image.rows);
                int hueValue = image.at<Vec3b>(y, x)[0] + halfBinSize;
                CV_DbgAssert(halfBinSize <= hueValue && hueValue <= maxHueValue + halfBinSize);
                int saturationValue = image.at<Vec3b>(y, x)[1];

                int binIdx = hueValue / binSize;
                float dist = float(hueValue) / float(binSize) - binIdx;
                --binIdx;
                CV_Assert(0.0f <= dist && dist <= 1.0f);
                fv.at<float>(0 < binIdx ? binIdx : params.bins - 1) += (1.0f - dist) * saturationValue;
                binIdx = (binIdx + 1 < params.bins) ? binIdx + 1 : 0;
                fv.at<float>(binIdx) += dist * saturationValue;
            }
        }
        normalize(fv, fv, 1.0, 0.0, NORM_L2);
    }
}


int Hue::getFeatureVectorLength() const
{
    return featureVectorLength;
}


void Hue::computeOnNewImage(const SourcesMap & sources)
{
    CV_Assert(sources.count(SOURCE_IMAGE));
    cvtColor(sources.at(SOURCE_IMAGE), sourceImage, CV_BGR2HSV);
    double minH, maxH;
    vector<Mat> channels;
    split(sourceImage, channels);
    minMaxIdx(channels[0], &minH, &maxH);
    CV_Assert(0 <= minH && minH < 180);
    CV_Assert(0 <= maxH && maxH < 180);
}


void Hue::computeOnNewScale(const float scale)
{
    Size scaledImageSize((int)(sourceImage.cols / scale), (int)(sourceImage.rows / scale));
    int method = (scale <= 1.0f) ? INTER_CUBIC : INTER_AREA;
    resize(sourceImage, image, scaledImageSize, 0.0, 0.0, method);
}


void Hue::getROIDescription(Mat & featureDescription,
                            const SourcesMap & sources,
                            const Rect & roi)
{
    CV_Assert(0 < roi.width);
    CV_Assert(0 < roi.height);
    CV_Assert(sources.count(SOURCE_IMAGE));
    const Mat & image = sources.at(SOURCE_IMAGE);
    Mat imageROI;
    getROI(image, imageROI, roi);
    resize(imageROI, imageROI, params.winSize);

    SourcesMap imageSource;
    imageSource[SOURCE_IMAGE] = imageROI;
    computeOnNewImage(imageSource);
    computeOnNewScale(1.0f);
    featureDescription.create(1, featureVectorLength, CV_32F);
    getFeatureVector(0, 0, featureDescription);
}


void Hue::getROIDescription(Mat & featureDescription,
                            const SourcesMap & sources,
                            const vector<Rect> & roi)
{
    featureDescription.create(roi.size(), featureVectorLength, CV_32F);
    for (size_t i = 0; i < roi.size(); ++i)
    {
        Mat featureVector = featureDescription.row(i);
        getROIDescription(featureVector, sources, roi[i]);
    }
}


vector<vector<DataTypeTime> > Hue::getRequiredSources() const
{
    vector<vector<DataTypeTime> > sources(1);
    sources[0].push_back(SOURCE_IMAGE);
    return sources;
}
