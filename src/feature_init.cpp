#include "swod/swod.hpp"
#include "opencv2/core/internal.hpp"

using namespace cv;


CV_INIT_ALGORITHM(OpenCVHOG, "SWOD.Feature.OpenCVHOG",
                  obj.info()->addParam(obj, "winSizeW", obj.params.winSizeW, true);
                  obj.info()->addParam(obj, "winSizeH", obj.params.winSizeH, true);
                  obj.info()->addParam(obj, "winStrideW", obj.params.winStrideW, true);
                  obj.info()->addParam(obj, "winStrideH", obj.params.winStrideH, true);
                  obj.info()->addParam(obj, "blockSizeW", obj.params.blockSizeW, true);
                  obj.info()->addParam(obj, "blockSizeH", obj.params.blockSizeH, true);
                  obj.info()->addParam(obj, "blockStrideW", obj.params.blockStrideW, true);
                  obj.info()->addParam(obj, "blockStrideH", obj.params.blockStrideH, true);
                  obj.info()->addParam(obj, "cellSizeW", obj.params.cellSizeW, true);
                  obj.info()->addParam(obj, "cellSizeH", obj.params.cellSizeH, true);
                  obj.info()->addParam(obj, "nbins", obj.params.nbins, true);
                  obj.info()->addParam(obj, "derivAperture", obj.params.derivAperture, true);
                  obj.info()->addParam(obj, "winSigma", obj.params.winSigma, true);
                  obj.info()->addParam(obj, "histogramNormType", obj.params.histogramNormType, true);
                  obj.info()->addParam(obj, "L2HysThreshold", obj.params.L2HysThreshold, true);
                  obj.info()->addParam(obj, "gammaCorrection", obj.params.gammaCorrection, true);
                  obj.info()->addParam(obj, "nlevels", obj.params.nlevels, true);
                  obj.info()->addParam(obj, "mask", obj.params.mask, true));


CV_INIT_ALGORITHM(PiotrHOG, "SWOD.Feature.PiotrHOG",
                  obj.info()->addParam(obj, "winSizeW", obj.params.winSizeW, true);
                  obj.info()->addParam(obj, "winSizeH", obj.params.winSizeH, true);
                  obj.info()->addParam(obj, "spatialStride", obj.params.spatialStride, true);
                  obj.info()->addParam(obj, "orientBins", obj.params.orientBins, true);
                  obj.info()->addParam(obj, "featureSubsetType", obj.params.featureSubsetType, true);
                  obj.info()->addParam(obj, "mask", obj.params.mask, true));


CV_INIT_ALGORITHM(HOF, "SWOD.Feature.HOF",
                  obj.info()->addParam(obj, "winSizeW", obj.params.winSizeW, true);
                  obj.info()->addParam(obj, "winSizeH", obj.params.winSizeH, true);
                  obj.info()->addParam(obj, "winStrideW", obj.params.winStrideW, true);
                  obj.info()->addParam(obj, "winStrideH", obj.params.winStrideH, true);
                  obj.info()->addParam(obj, "blockSizeW", obj.params.blockSizeW, true);
                  obj.info()->addParam(obj, "blockSizeH", obj.params.blockSizeH, true);
                  obj.info()->addParam(obj, "blockStrideW", obj.params.blockStrideW, true);
                  obj.info()->addParam(obj, "blockStrideH", obj.params.blockStrideH, true);
                  obj.info()->addParam(obj, "orientBins", obj.params.orientBins, true);
                  obj.info()->addParam(obj, "opticalFlowEpsilon", obj.params.opticalFlowEpsilon, true);
                  obj.info()->addParam(obj, "l2HysThreshold", obj.params.l2HysThreshold, true);
                  obj.info()->addParam(obj, "mask", obj.params.mask, true));


CV_INIT_ALGORITHM(Hue, "SWOD.Feature.Hue",
                  obj.info()->addParam(obj, "winSizeW", obj.params.winSize.width, true);
                  obj.info()->addParam(obj, "winSizeH", obj.params.winSize.height, true);
                  obj.info()->addParam(obj, "winStrideW", obj.params.spatialStride.width, true);
                  obj.info()->addParam(obj, "winStrideH", obj.params.spatialStride.height, true);
                  obj.info()->addParam(obj, "blockSizeW", obj.params.blockSize.width, true);
                  obj.info()->addParam(obj, "blockSizeH", obj.params.blockSize.height, true);
                  obj.info()->addParam(obj, "blockStrideW", obj.params.blockStride.width, true);
                  obj.info()->addParam(obj, "blockStrideH", obj.params.blockStride.height, true);
                  obj.info()->addParam(obj, "bins", obj.params.bins, true));


bool initFeatures()
{
  Ptr<Algorithm> opencvHOG = createOpenCVHOG();
  Ptr<Algorithm> piotrHOG = createPiotrHOG();
  Ptr<Algorithm> hof = createHOF();
  Ptr<Algorithm> hue = createHue();
  return (opencvHOG->info() != 0) &&
          (piotrHOG->info() != 0) &&
          (hof->info() != 0) &&
          (hue->info() != 0);
}
