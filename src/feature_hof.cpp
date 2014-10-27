#include "swod/feature_hof.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using std::map;
using std::string;
using std::vector;


namespace
{
    //DataTypeTime SOURCE_FIRST_IMAGE("image", -1);
    //DataTypeTime SOURCE_SECOND_IMAGE("image", 0);
    DataTypeTime SOURCE_FIRST_IMAGE("image", 0);
    DataTypeTime SOURCE_SECOND_IMAGE("image", 1);
    DataTypeTime SOURCE_OPTICAL_FLOW("optical_flow", 0);
}


HOFParams::HOFParams()
    : winSizeW(64), winSizeH(128),
      winStrideW(8), winStrideH(8),
      blockSizeW(16), blockSizeH(16),
      blockStrideW(8), blockStrideH(8),
      orientBins(6), opticalFlowEpsilon(0.025), l2HysThreshold(0.2f)
{}


HOFParams::HOFParams(const HOFParams & p)
{
    winSizeW = p.winSizeW;
    winSizeH = p.winSizeH;
    winStrideW = p.winStrideW;
    winStrideH = p.winStrideH;
    blockSizeW = p.blockSizeW;
    blockSizeH = p.blockSizeH;
    blockStrideW = p.blockStrideW;
    blockStrideH = p.blockStrideH;
    orientBins = p.orientBins;
    opticalFlowEpsilon = p.opticalFlowEpsilon;
    l2HysThreshold = p.l2HysThreshold;
    p.mask.copyTo(mask);
}


HOFParams & HOFParams::operator= (const HOFParams & p)
{
    winSizeW = p.winSizeW;
    winSizeH = p.winSizeH;
    winStrideW = p.winStrideW;
    winStrideH = p.winStrideH;
    blockSizeW = p.blockSizeW;
    blockSizeH = p.blockSizeH;
    blockStrideW = p.blockStrideW;
    blockStrideH = p.blockStrideH;
    orientBins = p.orientBins;
    opticalFlowEpsilon = p.opticalFlowEpsilon;
    l2HysThreshold = p.l2HysThreshold;
    p.mask.copyTo(mask);
    return *this;
}


namespace
{
    inline Size calcGridSize(int imageSizeW,
                             int imageSizeH,
                             int winSizeW,
                             int winSizeH,
                             int winStrideW,
                             int winStrideH)
    {
        return Size((imageSizeW - winSizeW) / winStrideW + 1,
                    (imageSizeH - winSizeH) / winStrideH + 1);
    }
}


HOF::HOF()
{
    opticalFlowComputer = createOptFlow_DualTVL1();
    initDescriptor();
}


void HOF::initDescriptor()
{
    opticalFlowComputer->setDouble("epsilon", params.opticalFlowEpsilon);
    winSizeInBlocks = calcGridSize(params.winSizeW, params.winSizeH,
                                   params.blockSizeW, params.blockSizeH,
                                   params.blockStrideW, params.blockStrideH);
    featureVectorLength = winSizeInBlocks.area() * params.orientBins * 4;
}


void HOF::setParams(const HOFParams & p)
{
    CV_Assert(p.winStrideH % p.blockStrideH == 0);
    CV_Assert(p.winStrideW % p.blockStrideW == 0);
    params = p;
    initDescriptor();
}


void HOF::read(const FileNode & fn)
{
    Algorithm::read(fn);
    initDescriptor();
}


Size HOF::getNumOfSpatialSteps() const
{
    return winGridSize;
}


void HOF::getFeatureVector(int windowIndexX,
                           int windowIndexY,
                           Mat & featureVector) const
{
    CV_Assert(!hof.empty());
    CV_Assert(0 <= windowIndexX && windowIndexX < winGridSize.width);
    CV_Assert(0 <= windowIndexY && windowIndexY < winGridSize.height);

    featureVector.create(1, featureVectorLength, CV_32F);
    int winStrideInBlocksW = params.winStrideW / params.blockStrideW;
    int winStrideInBlocksH = params.winStrideH / params.blockStrideH;
    int rowSize = winSizeInBlocks.width * params.orientBins * 4 * sizeof(float);
    int rowStep = hof.step;
    uchar * featureVectorData = featureVector.data;
    uchar * hofData = hof.data + winStrideInBlocksH * windowIndexY * rowStep +
                      winStrideInBlocksW * windowIndexX *
                      params.orientBins * 4 * sizeof(float);
    for (int i = 0; i < winSizeInBlocks.height; ++i)
    {
        memcpy(featureVectorData, hofData, rowSize);
        featureVectorData += rowSize;
        hofData += rowStep;
    }
}


int HOF::getFeatureVectorLength() const
{
    return featureVectorLength;
}


void HOF::computeOnNewImage(const SourcesMap & sources)
{
    bool hasOpticalFlowAsSource = (0 < sources.count(SOURCE_OPTICAL_FLOW));
    bool hasImagesPairAsSource = (0 < sources.count(SOURCE_FIRST_IMAGE) &&
                                  0 < sources.count(SOURCE_SECOND_IMAGE));

    CV_Assert(hasOpticalFlowAsSource || hasImagesPairAsSource);
    if (hasOpticalFlowAsSource)
    {
        sources.at(SOURCE_OPTICAL_FLOW).copyTo(flow);
    }
    else if (hasImagesPairAsSource)
    {
        Mat firstImage;
        cvtColor(sources.at(SOURCE_FIRST_IMAGE), firstImage, CV_BGR2GRAY);
        cvtColor(sources.at(SOURCE_SECOND_IMAGE), secondImage, CV_BGR2GRAY);
        CV_Assert(firstImage.size == secondImage.size);
        opticalFlowComputer->calc(firstImage, secondImage, flow);
    }
}


namespace
{
    void normalizeBlockHistogram(float * hist, size_t n, float threshold)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            sum += hist[i] * hist[i];
        }

        float scale = 1.0f / (std::sqrt(sum) + n * 0.1f);
        sum = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            hist[i] = std::min(hist[i] * scale, threshold);
            sum += hist[i] * hist[i];
        }

        scale = 1.0f / (std::sqrt(sum) + 1e-3f);
        for (size_t i = 0; i < n; ++i)
        {
            hist[i] *= scale;
        }
    }


    void calcHistogram(Mat & values,
                       float * histogram,
                       int nbins)
    {
        CV_Assert(!values.empty());
        CV_Assert(values.channels() == 2);
        CV_Assert(histogram);

        vector<Mat> vectorComponents;
        split(values, vectorComponents);
        Mat angle;
        Mat magnitude;
        cartToPolar(vectorComponents[0], vectorComponents[1], magnitude, angle);

        float binSizeInv = nbins / CV_PI;
        for (int i = 0; i < values.rows; ++i)
        {
            for (int j = 0; j < values.cols; ++j)
            {
                float alpha = angle.at<float>(i, j);
                float r = magnitude.at<float>(i, j);

                float beta = alpha * binSizeInv - 0.5f;
                int histIdx = cvFloor(beta);
                beta -= histIdx;

                if (histIdx < 0)
                {
                    histIdx += nbins;
                }
                else if (nbins <= histIdx)
                {
                    histIdx -= nbins;
                }
                CV_Assert(static_cast<unsigned>(histIdx) < static_cast<unsigned>(nbins));

                histogram[histIdx] += r * (1.0f - beta);
                ++histIdx;
                if (nbins <= histIdx)
                {
                    histIdx = 0;
                }
                histogram[histIdx] += r * beta;
            }
        }
    }
}


void HOF::computeOnNewScale(const float scale)
{
    Mat scaledFlow;
    Size scaledSize(static_cast<int>(flow.cols / scale),
                    static_cast<int>(flow.rows / scale));
    resize(flow, scaledFlow, scaledSize);

    winGridSize = calcGridSize(scaledSize.width, scaledSize.height,
                               params.winSizeW, params.winSizeH,
                               params.winStrideW, params.winStrideH);

    // compute block histograms
    int blockHistogramLength = params.orientBins * 4;
    Size hofBlocksGrid = calcGridSize(scaledSize.width, scaledSize.height,
                                      params.blockSizeW, params.blockSizeH,
                                      params.blockStrideW, params.blockStrideH);
    hof = Mat(hofBlocksGrid.height, hofBlocksGrid.width * blockHistogramLength,
              CV_32F, Scalar(0.0f));
    float * hofData = reinterpret_cast<float*>(hof.data);
    int hofDataStep = hof.step / sizeof(float);
    int n = params.blockSizeH;
    int nHalf = n / 2;
    int m = params.blockSizeW;
    int mHalf = m / 2;
    
    for (int i = 0; i < hofBlocksGrid.height; ++i)
    {
        int ii = i * params.blockStrideH;
        for (int j = 0; j < hofBlocksGrid.width; ++j)
        {
            float * hist = hofData + i * hofDataStep + j * blockHistogramLength;
            int jj = j * params.blockStrideW;

            Mat topLeft = scaledFlow(Range(ii, ii + nHalf),
                                     Range(jj, jj + mHalf));
            Mat topRight = scaledFlow(Range(ii, ii + nHalf),
                                      Range(jj + mHalf, jj + m));
            Mat bottomLeft = scaledFlow(Range(ii + nHalf, ii + n),
                                        Range(jj, jj + mHalf));
            Mat bottomRight = scaledFlow(Range(ii + nHalf, ii + n),
                                         Range(jj + mHalf, jj + m));

            Mat diff;
            diff = bottomLeft - topLeft;
            calcHistogram(diff, hist, params.orientBins);

            diff = topRight - topLeft;
            calcHistogram(diff, hist + params.orientBins, params.orientBins);

            diff = bottomRight - topLeft;
            calcHistogram(diff, hist + 2 * params.orientBins, params.orientBins);

            diff = topRight - bottomLeft;
            calcHistogram(diff, hist + 3 * params.orientBins, params.orientBins);

            normalizeBlockHistogram(hist, blockHistogramLength,
                                    params.l2HysThreshold);
        }
    }
}


void HOF::getROIDescription(Mat & featureDescription,
                            const SourcesMap & sources,
                            const Rect & roi)
{
    CV_Assert(0 < roi.width);
    CV_Assert(0 < roi.height);
    bool hasOpticalFlowAsSource = (0 < sources.count(SOURCE_OPTICAL_FLOW));
    bool hasImagesPairAsSource = (0 < sources.count(SOURCE_FIRST_IMAGE) &&
                                  0 < sources.count(SOURCE_SECOND_IMAGE));
    CV_Assert(hasImagesPairAsSource || hasOpticalFlowAsSource);
    
    Mat opticalFlow;
    if (hasOpticalFlowAsSource)
    {
        opticalFlow = sources.at(SOURCE_OPTICAL_FLOW);
    }
    else // if (hasImagesPairAsSource)
    {
        Mat image1, image2;
        cvtColor(sources.at(SOURCE_FIRST_IMAGE), image1, CV_BGR2GRAY);
        cvtColor(sources.at(SOURCE_SECOND_IMAGE), image2, CV_BGR2GRAY);
        opticalFlowComputer->calc(image1, image2, opticalFlow);
    }

    Mat opticalFlowROI;
    getROI(opticalFlow, opticalFlowROI, roi);
    resize(opticalFlowROI, opticalFlowROI,
           Size(params.winSizeW, params.winSizeH));
    SourcesMap imageSource;
    imageSource[SOURCE_OPTICAL_FLOW] = opticalFlowROI;
    computeOnNewImage(imageSource);
    computeOnNewScale(1.0f);

    featureDescription.create(1, featureVectorLength, CV_32F);
    getFeatureVector(0, 0, featureDescription);
}


void HOF::getROIDescription(Mat & featureDescription,
                            const SourcesMap & sources,
                            const vector<Rect> & roi)
{
    bool hasOpticalFlowAsSource = (0 < sources.count(SOURCE_OPTICAL_FLOW));
    bool hasImagesPairAsSource = (0 < sources.count(SOURCE_FIRST_IMAGE) &&
                                  0 < sources.count(SOURCE_SECOND_IMAGE));
    CV_Assert(hasImagesPairAsSource || hasOpticalFlowAsSource);
    
    Mat opticalFlow;
    if (hasOpticalFlowAsSource)
    {
        opticalFlow = sources.at(SOURCE_OPTICAL_FLOW);
    }
    else // if (hasImagesPairAsSource)
    {
        Mat image1, image2;
        cvtColor(sources.at(SOURCE_FIRST_IMAGE), image1, CV_BGR2GRAY);
        cvtColor(sources.at(SOURCE_SECOND_IMAGE), image2, CV_BGR2GRAY);
        opticalFlowComputer->calc(image1, image2, opticalFlow);
    }

    featureDescription.create(roi.size(), featureVectorLength, CV_32F);

    for (size_t t = 0; t < roi.size(); ++t)
    {
        const Rect & r = roi[t];
        CV_Assert(0 < r.width);
        CV_Assert(0 < r.height);
        Mat opticalFlowROI;
        getROI(opticalFlow, opticalFlowROI, r);
        resize(opticalFlowROI, opticalFlowROI,
               Size(params.winSizeW, params.winSizeH));

        SourcesMap imageSource;
        imageSource[SOURCE_OPTICAL_FLOW] = opticalFlowROI;
        computeOnNewImage(imageSource);
        computeOnNewScale(1.0f);
        Mat featureDescriptionRow = featureDescription.row(t);
        getFeatureVector(0, 0, featureDescriptionRow);
    }
}


Mat HOF::visualizeOpticalFlow() const
{
    vector<Mat> flowComponents;
    split(flow, flowComponents);
    Mat magnitude, angle;
    cartToPolar(flowComponents[0], flowComponents[1], magnitude, angle, true);

    vector<Mat> opticalFlowImageChannels(3);
    angle.convertTo(opticalFlowImageChannels[0], CV_8UC1);
    opticalFlowImageChannels[1] = Mat(flow.size(), CV_8UC1, Scalar(255));
    Mat normalizedMagnitudes;
    normalize(magnitude, normalizedMagnitudes, 0.0, 255.0, cv::NORM_MINMAX);
    normalizedMagnitudes.convertTo(opticalFlowImageChannels[2], CV_8UC1);
    Mat opticalFlowImage;
    merge(opticalFlowImageChannels, opticalFlowImage);
    cvtColor(opticalFlowImage, opticalFlowImage, cv::COLOR_HSV2BGR);
    return opticalFlowImage;
}


vector<vector<DataTypeTime> > HOF::getRequiredSources() const
{
    vector<vector<DataTypeTime> > sources(2);
    sources[0].push_back(SOURCE_OPTICAL_FLOW);
    sources[1].push_back(SOURCE_FIRST_IMAGE);
    sources[1].push_back(SOURCE_SECOND_IMAGE);
    return sources;
}
