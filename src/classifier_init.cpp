#include "swod/swod.hpp"
#include "opencv2/core/internal.hpp"

using namespace cv;


CV_INIT_ALGORITHM(SVMClassifier, "SWOD.Classifier.SVM",
                  obj.info()->addParam(obj, "svmType", obj.params.svm_type, true);
                  obj.info()->addParam(obj, "kernelType", obj.params.kernel_type, true);
                  obj.info()->addParam(obj, "degree", obj.params.degree, true);
                  obj.info()->addParam(obj, "gamma", obj.params.gamma, true);
                  obj.info()->addParam(obj, "coef0", obj.params.coef0, true);
                  obj.info()->addParam(obj, "C", obj.params.C, true);
                  obj.info()->addParam(obj, "nu", obj.params.nu, true);
                  obj.info()->addParam(obj, "crossValidationFolds", obj.crossValidationFolds, true);
                  obj.info()->addParam(obj, "modelFileName", obj.modelFileName, true);
                  obj.info()->addParam(obj, "modelName", obj.modelName, true));


CV_INIT_ALGORITHM(GBTClassifier, "SWOD.Classifier.GBT",
                  obj.info()->addParam(obj, "treeDepth", obj.params.max_depth, true);
                  obj.info()->addParam(obj, "minSamplesInLeaf", obj.params.min_sample_count, true);
                  obj.info()->addParam(obj, "useSurrogateSplits", obj.params.use_surrogates, true);
                  obj.info()->addParam(obj, "treesNum", obj.params.weak_count, true);
                  obj.info()->addParam(obj, "lossFunctionType", obj.params.loss_function_type, true);
                  obj.info()->addParam(obj, "subsamplePortion", obj.params.subsample_portion, true);
                  obj.info()->addParam(obj, "learningRate", obj.params.shrinkage, true);
                  obj.info()->addParam(obj, "modelFileName", obj.modelFileName, true);
                  obj.info()->addParam(obj, "modelName", obj.modelName, true));


bool initClassifiers()
{
  Ptr<Algorithm> svm = createSVMClassifier();
  Ptr<Algorithm> gbt = createGBTClassifier();
  return (svm->info() != 0) &&
         (gbt->info() != 0);
}
