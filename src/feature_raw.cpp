#include "swod/feature_raw.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;


namespace
{
    DataTypeTime SOURCE_IMAGE("image", 0);
}


RawPixelParams::RawPixelParams()
    : winSizeW(64), winSizeH(128), winStrideW(8), winStrideH(8)
{}


RawPixelParams::RawPixelParams(const RawPixelParams & p)
{
    winSizeW = p.winSizeW;
    winSizeH = p.winSizeH;
    winStrideW = p.winStrideW;
    winStrideH = p.winStrideH;
}


RawPixelParams & RawPixelParams::operator= (const RawPixelParams & p)
{
    winSizeW = p.winSizeW;
    winSizeH = p.winSizeH;
    winStrideW = p.winStrideW;
    winStrideH = p.winStrideH;
    return *this;
}


RawPixel::RawPixel()
{}


void RawPixel::setParams(const RawPixelParams & p)
{
    params = p;
    initDescriptor();
}


void RawPixel::read(const cv::FileNode & fn)
{
    Algorithm::read(fn);
    initDescriptor();
}


Size RawPixel::getNumOfSpatialSteps() const
{
    return Size((scaledImg.cols - params.winSizeW) / params.winStrideW + 1,
                (scaledImg.rows - params.winSizeH) / params.winStrideH + 1);
}


void RawPixel::getFeatureVector(int positionX,
                                int positionY,
                                Mat & featureVector) const
{
    CV_Assert(0 <= positionX);
    CV_Assert(0 <= positionY);
    int featureVectorSize = params.winSizeW * params.winSizeH;
    featureVector.create(1, featureVectorSize, CV_32F);

    Rect roi(positionX * params.winStrideW,
             positionY * params.winStrideH,
             positionX * params.winStrideW + params.winSizeW,
             positionY * params.winStrideH + params.winSizeH);
    Mat patch = scaledImg(roi).clone();
    patch = patch.reshape(1, 1);
    patch.convertTo(featureVector, CV_32F);
}


int RawPixel::getFeatureVectorLength() const
{
    return params.winSizeW * params.winSizeH;
}


void RawPixel::initDescriptor()
{}


void RawPixel::computeOnNewImage(const SourcesMap & sources)
{
    CV_Assert(sources.count(SOURCE_IMAGE));
    CV_Assert(!sources.at(SOURCE_IMAGE).empty());
    sources.at(SOURCE_IMAGE).copyTo(img);
    scaledImg = img;
}


void RawPixel::computeOnNewScale(const float scale)
{
    CV_Assert(0.0f < scale);
    Size scaledImageSize(static_cast<int>(img.cols / scale),
                         static_cast<int>(img.rows / scale));
    CV_Assert(0 < scaledImageSize.width && 0 < scaledImageSize.height);
    int method = (scale <= 1.0f) ? INTER_CUBIC : INTER_AREA;
    resize(img, scaledImg, scaledImageSize, 0.0, 0.0, method);
}


void RawPixel::getROIDescription(cv::Mat & featureDescription,
                                 const SourcesMap & sources,
                                 const cv::Rect & roi)
{
    CV_Assert(sources.count(SOURCE_IMAGE));
    CV_Assert(0 < roi.width);
    CV_Assert(0 < roi.height);
    const Mat & im = sources.at(SOURCE_IMAGE);
    CV_Assert(!im.empty());

    Mat imageROI;
    getROI(im, imageROI, roi);
    resize(imageROI, imageROI, Size(params.winSizeW, params.winSizeH));

    SourcesMap imageSource;
    imageSource[SOURCE_IMAGE] = imageROI;
    computeOnNewImage(imageSource);
    computeOnNewScale(1.0f);
    featureDescription.create(1, getFeatureVectorLength(), CV_32F);
    getFeatureVector(0, 0, featureDescription);
}


void RawPixel::getROIDescription(cv::Mat & featureDescription,
                                 const SourcesMap & sources,
                                 const std::vector<cv::Rect> & roi)
{
    featureDescription.create(roi.size(), getFeatureVectorLength(), CV_32F);
    for (size_t i = 0; i < roi.size(); ++i)
    {
        Mat featureVector = featureDescription.row(i);
        getROIDescription(featureVector, sources, roi[i]);
    }
}


vector<vector<DataTypeTime> > RawPixel::getRequiredSources() const
{
    vector<vector<DataTypeTime> > sources(1);
    sources[0].push_back(SOURCE_IMAGE);
    return sources;
}
