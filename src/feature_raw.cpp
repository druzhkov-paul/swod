#include "swod/feature_raw.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>

using namespace cv;
using namespace std;


namespace
{
    DataTypeTime SOURCE_IMAGE("image", 0);
}


RawPixelParams::RawPixelParams()
    : winSizeW(64),
      winSizeH(128),
      winStrideW(8),
      winStrideH(8),
      doNormalization(true),
      normalizationRegularizer(1e-5),
      doWhitening(true),
      whiteningTransform(Mat::eye(64 * 128, 64 * 128, CV_32F))
{}


RawPixelParams::RawPixelParams(const RawPixelParams & p)
{
    winSizeW = p.winSizeW;
    winSizeH = p.winSizeH;
    winStrideW = p.winStrideW;
    winStrideH = p.winStrideH;
    doNormalization = p.doNormalization;
    normalizationRegularizer = p.normalizationRegularizer;
    doWhitening = p.doWhitening;
    whiteningTransform = p.whiteningTransform.clone();
}


RawPixelParams & RawPixelParams::operator= (const RawPixelParams & p)
{
    winSizeW = p.winSizeW;
    winSizeH = p.winSizeH;
    winStrideW = p.winStrideW;
    winStrideH = p.winStrideH;
    doNormalization = p.doNormalization;
    normalizationRegularizer = p.normalizationRegularizer;
    doWhitening = p.doWhitening;
    whiteningTransform = p.whiteningTransform.clone();
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
    if (params.doWhitening && fn["whiteningTransform"].isString())
    {
        string whiteningFile, whiteningName;
        fn["whiteningTransform"] >> whiteningFile;
        fn["whiteningTransformName"] >> whiteningName;
        FileStorage fs(whiteningFile, FileStorage::READ);
        CV_Assert(fs.isOpened());
        fs[whiteningName] >> params.whiteningTransform;
        fs.release();
    }
    else
    {
        fn["whiteningTransform"] >> params.whiteningTransform;
    }
    initDescriptor();
}


Size RawPixel::getNumOfSpatialSteps() const
{
    CV_Assert(!scaledImg.empty());
    return Size((scaledImg.cols - params.winSizeW) / params.winStrideW + 1,
                (scaledImg.rows - params.winSizeH) / params.winStrideH + 1);
}


void RawPixel::getFeatureVector(int positionX,
                                int positionY,
                                Mat & featureVector) const
{
    CV_Assert(!scaledImg.empty());
    CV_Assert(0 <= positionX);
    CV_Assert(0 <= positionY);
    CV_Assert(positionX * params.winStrideW + params.winSizeW <= scaledImg.cols);
    CV_Assert(positionY * params.winStrideH + params.winSizeH <= scaledImg.rows);

    featureVector.create(1, getFeatureVectorLength(), CV_32F);

    Rect roi(positionX * params.winStrideW,
             positionY * params.winStrideH,
             params.winSizeW,
             params.winSizeH);
    Mat patch = scaledImg(roi).clone();
    patch = patch.reshape(1, 1);
    CV_Assert(static_cast<int>(patch.total()) == getFeatureVectorLength());
    patch.convertTo(featureVector, CV_32F);
    if (params.doNormalization)
    {
        normalizeSample(featureVector);
    }
    if (params.doWhitening)
    {
        whitenSample(featureVector);
    }
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
    const Mat & im = sources.at(SOURCE_IMAGE);
    CV_Assert((im.type() == CV_8UC1) || (im.type() == CV_8UC3));
    if (im.type() == CV_8UC1)
    {
        im.copyTo(img);
    }
    else
    {
        cvtColor(im, img, CV_BGR2GRAY);
        CV_Assert(img.type() == CV_8U);
    }
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


void RawPixel::normalizeSample(Mat & sample) const
{
    Scalar mean, sd;
    meanStdDev(sample, mean, sd);
    float m = static_cast<float>(mean[0]);
    float var = static_cast<float>(sd[0]);
    var *= var;
    sample -= m;
    sample /= sqrt(var + params.normalizationRegularizer);
}


void RawPixel::whitenSample(Mat & sample) const
{
    sample = params.whiteningTransform * sample;
}
