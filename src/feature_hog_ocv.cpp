#include "swod/feature_hog_ocv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/internal.hpp"

using namespace cv;
using namespace std;


namespace
{
    DataTypeTime SOURCE_IMAGE("image", 0);
}


OpenCVHOGParams::OpenCVHOGParams()
    : winSizeW(64), winSizeH(128), winStrideW(8), winStrideH(8),
      blockSizeW(16), blockSizeH(16), blockStrideW(8), blockStrideH(8),
      cellSizeW(8), cellSizeH(8), nbins(9), derivAperture(1), winSigma(-1),
      histogramNormType(cv::HOGDescriptor::L2Hys), L2HysThreshold(0.2),
      gammaCorrection(true), nlevels(cv::HOGDescriptor::DEFAULT_NLEVELS)
{}


OpenCVHOGParams::OpenCVHOGParams(const OpenCVHOGParams & p)
{
    winSizeW = p.winSizeW;
    winSizeH = p.winSizeH;
    winStrideW = p.winStrideW;
    winStrideH = p.winStrideH;
    blockSizeW = p.blockSizeW;
    blockSizeH = p.blockSizeH;
    blockStrideW = p.blockStrideW;
    blockStrideH = p.blockStrideH;
    cellSizeW = p.cellSizeW;
    cellSizeH = p.cellSizeH;
    nbins = p.nbins;
    derivAperture = p.derivAperture;
    winSigma = p.winSigma;
    histogramNormType = p.histogramNormType;
    L2HysThreshold = p.L2HysThreshold;
    gammaCorrection = p.gammaCorrection;
    nlevels = p.nlevels;
    p.mask.copyTo(mask);
}


OpenCVHOGParams & OpenCVHOGParams::operator= (const OpenCVHOGParams & p)
{
    winSizeW = p.winSizeW;
    winSizeH = p.winSizeH;
    winStrideW = p.winStrideW;
    winStrideH = p.winStrideH;
    blockSizeW = p.blockSizeW;
    blockSizeH = p.blockSizeH;
    blockStrideW = p.blockStrideW;
    blockStrideH = p.blockStrideH;
    cellSizeW = p.cellSizeW;
    cellSizeH = p.cellSizeH;
    nbins = p.nbins;
    derivAperture = p.derivAperture;
    winSigma = p.winSigma;
    histogramNormType = p.histogramNormType;
    L2HysThreshold = p.L2HysThreshold;
    gammaCorrection = p.gammaCorrection;
    nlevels = p.nlevels;
    p.mask.copyTo(mask);
    return *this;
}


OpenCVHOG::OpenCVHOG()
{
    stepsX = 0;
    stepsY = 0;
}


OpenCVHOG::~OpenCVHOG()
{}


void OpenCVHOG::setParams(const OpenCVHOGParams & p)
{
    params = p;
    initDescriptor();
}


void OpenCVHOG::read(const cv::FileNode & fn)
{
    Algorithm::read(fn);
    initDescriptor();
}


Size OpenCVHOG::getNumOfSpatialSteps() const
{
    return Size(stepsX, stepsY);
}


void OpenCVHOG::getFeatureVector(int positionX,
                                 int positionY,
                                 Mat & featureVector) const
{
    CV_Assert(0 <= positionX && positionX < stepsX);
    CV_Assert(0 <= positionY && positionY < stepsY);
    featureVector.create(1, featureVectorSize, CV_32F);
    int windowIdx = positionY * stepsX + positionX;
    if (params.mask.empty())
    {
        featuresMat.row(windowIdx).copyTo(featureVector);
    }
    else
    {
        CV_Assert(params.mask.total() == static_cast<size_t>(featureVectorSize));
        Mat fv = featuresMat.row(windowIdx);
        size_t j = 0;
        for (size_t i = 0; i < params.mask.total(); ++i)
        {
            if (params.mask.at<uchar>(i))
            {
                featureVector.at<float>(j) = fv.at<float>(i);
                ++j;
            }
        }
    }
}


int OpenCVHOG::getFeatureVectorLength() const
{
    return featureVectorSize;
}


void OpenCVHOG::initDescriptor()
{
    hog = HOGDescriptor(Size(params.winSizeW, params.winSizeH),
                        Size(params.blockSizeW, params.blockSizeH),
                        Size(params.blockStrideW, params.blockStrideH),
                        Size(params.cellSizeW, params.cellSizeH),
                        params.nbins, params.derivAperture,
                        params.winSigma, params.histogramNormType,
                        params.L2HysThreshold, params.gammaCorrection,
                        params.nlevels);

    featureVectorSize = params.mask.empty() ?
                            hog.getDescriptorSize() :
                            (int)(sum(params.mask)[0]);
}


void OpenCVHOG::computeOnNewImage(const SourcesMap & sources)
{
    CV_Assert(sources.count(SOURCE_IMAGE));
    CV_Assert(!sources.at(SOURCE_IMAGE).empty());

    sources.at(SOURCE_IMAGE).copyTo(img);
    hog.compute(img, features, Size(params.winStrideW, params.winStrideH), Size());
    stepsX = (img.cols - params.winSizeW) / params.winStrideW + 1;
    stepsY = (img.rows - params.winSizeH) / params.winStrideH + 1;
    featuresMat = Mat(features);
    features.clear();
    featuresMat = featuresMat.reshape(0, stepsX * stepsY);
}


void OpenCVHOG::computeOnNewScale(const float scale)
{
    CV_Assert(0.0f < scale);
    Size scaledImageSize(static_cast<int>(img.cols / scale),
                         static_cast<int>(img.rows / scale));
    CV_Assert(0 < scaledImageSize.width && 0 < scaledImageSize.height);
    int method = (scale <= 1.0f) ? INTER_CUBIC : INTER_AREA;
    Mat scaledImg;
    resize(img, scaledImg, scaledImageSize, 0.0, 0.0, method);
    hog.compute(scaledImg, features, Size(params.winStrideW, params.winStrideH), Size());
    stepsX = (scaledImageSize.width - params.winSizeW) / params.winStrideW + 1;
    stepsY = (scaledImageSize.height - params.winSizeH) / params.winStrideH + 1;
    featuresMat = Mat(features);
    features.clear();
    featuresMat = featuresMat.reshape(0, stepsX * stepsY);
}


void OpenCVHOG::getROIDescription(cv::Mat & featureDescription,
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
    featureDescription.create(1, featureVectorSize, CV_32F);
    getFeatureVector(0, 0, featureDescription);
}


void OpenCVHOG::getROIDescription(cv::Mat & featureDescription,
                                  const SourcesMap & sources,
                                  const std::vector<cv::Rect> & roi)
{
    featureDescription.create(roi.size(), featureVectorSize, CV_32F);
    for (size_t i = 0; i < roi.size(); ++i)
    {
        Mat featureVector = featureDescription.row(i);
        getROIDescription(featureVector, sources, roi[i]);
    }
}


vector<vector<DataTypeTime> > OpenCVHOG::getRequiredSources() const
{
    vector<vector<DataTypeTime> > sources(1);
    sources[0].push_back(SOURCE_IMAGE);
    return sources;
}
