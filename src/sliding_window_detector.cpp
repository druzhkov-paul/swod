#include "swod/sliding_window_detector.hpp"
#include "swod/nms.hpp"
#include <vector>
#include <map>
#include <string>
#include <cstdio>

using namespace cv;
using namespace std;


DetectionParams::DetectionParams()
    : winSize(64, 128),
      winBorder(8, 8),
      spatialStride(8),
      minObjectHeight(50),
      maxObjectHeight(-1),
      scaleStep(1.2f)
{}


void DetectionParams::write(FileStorage & fs) const
{
    CV_Assert(fs.isOpened());
    fs << "{";
    fs << "window_size" << winSize;
    fs << "window_border_size" << winBorder;
    fs << "window_spatial_stride" << spatialStride;
    fs << "min_object_height" << minObjectHeight;
    fs << "max_object_height" << maxObjectHeight;
    fs << "scale_step" << scaleStep;
    fs << "}";
}


void DetectionParams::read(const FileNode & fn)
{
    CV_Assert(!fn.empty());
    fn["window_size"] >> winSize;
    fn["window_border_size"] >> winBorder;
    fn["window_spatial_stride"] >> spatialStride;
    fn["min_object_height"] >> minObjectHeight;
    fn["max_object_height"] >> maxObjectHeight;
    fn["scale_step"] >> scaleStep;
}


void write(FileStorage& fs,
           const string &,
           const DetectionParams & x)
{
    x.write(fs);
}


void read(const FileNode & fn,
          DetectionParams & x,
          const DetectionParams & defaultValue)
{
    if (!fn.empty())
    {
        x.read(fn);
    }
    else
    {
        x = defaultValue;
    }
}


namespace
{
    inline int getObjectHeight(const Size & winSize,
                               const Size & winBorder,
                               float scale)
    {
        int realObjHeight = winSize.height - 2 * winBorder.height;
        CV_Assert(0 < realObjHeight);
        return static_cast<int>(realObjHeight * scale);
    }


    // FIXME: return both max height and width
    inline int getMaxObjectHeight(const Size & winSize,
                                  const Size & winBorder,
                                  const Size & imgSize)
    {
        int realObjHeight = winSize.height - 2 * winBorder.height;
        int realObjWidth = winSize.width - 2 * winBorder.width;
        CV_Assert(0 < realObjHeight && 0 < realObjWidth);
        float maxScale = min(static_cast<float>(imgSize.height) / winSize.height,
                             static_cast<float>(imgSize.width) / winSize.width);
        return static_cast<int>(realObjHeight * maxScale);
    }
}

void detect(Features & features,
            const Ptr<Classifier> classifier,
            const SourcesMap & sources,
            const DetectionParams & params,
            ImageAnnotation & ann)
{
    const DataTypeTime SOURCE_IMAGE("image", 0);
    CV_Assert(features.featuresSet.size());
    //CV_Assert(classifier != nullptr);
    CV_Assert(sources.count(SOURCE_IMAGE));

    int realObjHeight = params.winSize.height - 2 * params.winBorder.height;
    float scale = static_cast<float>(params.minObjectHeight) /
                  static_cast<float>(realObjHeight);

    int maxObjectHeight = params.maxObjectHeight < 0 ?
                              getMaxObjectHeight(params.winSize,
                                                 params.winBorder,
                                                 sources.at(SOURCE_IMAGE).size()) :
                              params.maxObjectHeight;

    features.computeOnNewImage(sources);
    Mat featureVector;
    while (getObjectHeight(params.winSize, params.winBorder, scale) < maxObjectHeight)
    {
        features.computeOnNewScale(scale);
        Size spatialGrid = features.getNumOfSpatialSteps();
        for (int i = 0; i < spatialGrid.height; ++i)
        {
            for (int j = 0; j < spatialGrid.width; ++j)
            {
                features.getTotalFeatureVector(j, i, featureVector);
                vector<float> scores;
                int predictedLabel = classifier->predict(featureVector, scores);
                if (predictedLabel)
                {
                    Rect bbox;
                    bbox.x = static_cast<int>(j * params.spatialStride * scale);
                    bbox.y = static_cast<int>(i * params.spatialStride * scale);
                    bbox.width  = static_cast<int>(params.winSize.width * scale);
                    bbox.height = static_cast<int>(params.winSize.height * scale);
                    removeBBoxBorder(bbox, params.winSize, params.winBorder);
                    ann.addBBox(bbox, predictedLabel, scores[predictedLabel]);
                }
            }
        }
        scale *= params.scaleStep;
    }
}
