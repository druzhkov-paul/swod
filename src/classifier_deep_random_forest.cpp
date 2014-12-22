#include "swod/classifier_deep_random_forest.hpp"
#include <sstream>

using namespace cv;
using std::vector;
using std::string;
using std::stringstream;


DRFClassifierParams::DRFClassifierParams()
    : layersNum(1),
      rfParams(1)
{}


DRFClassifier::DRFClassifier()
    : randomForests(0),
      modelFileName(""),
      modelName("")
{}


int DRFClassifier::predict(const Mat & sample,
                           vector<float> & weights) const
{
    Mat featureDescription(sample);
    for (int i = 0; i < params.layersNum - 1; ++i)
    {
        randomForests[i].getLeavesIndices(featureDescription, featureDescription);
    }

    const RandomForest & rf = randomForests[params.layersNum - 1];
    float prob = rf.predict_prob(sample);
    weights.resize(2);
    weights[0] = 1.0f - prob;
    weights[1] = prob;
    CV_DbgAssert(static_cast<int>(0.5f < prob) == static_cast<int>(rf.predict(sample)));

    return (0.5f < prob);
}


void DRFClassifier::train(const Mat & dataset,
                          const Mat & responses)
{
    CV_Assert(0 < params.layersNum);
    CV_Assert(static_cast<size_t>(params.layersNum) == params.rfParams.size());

    randomForests.resize(params.layersNum);
    CvRTrees & rf = randomForests[0];
    rf.train(dataset, CV_ROW_SAMPLE, responses, Mat(), Mat(),
             Mat(), Mat(), params.rfParams[0]);

    if (1 < params.layersNum)
    {
        int samplesNum = dataset.rows;
        Mat currentDataset = dataset.clone();

        for (int i = 1; i < params.layersNum; ++i)
        {
            Mat newDataset(samplesNum, randomForests[i - 1].get_tree_count(), CV_32F);
            for (int j = 0; j < dataset.rows; ++j)
            {
                Mat newFeatureVector = newDataset.row(j);
                randomForests[i - 1].getLeavesIndices(currentDataset.row(j), newFeatureVector);
            }
            currentDataset.release();
            currentDataset = newDataset;

            CvRTrees & rf = randomForests[i];
            rf.train(currentDataset, CV_ROW_SAMPLE, responses, Mat(), Mat(),
                     Mat(), Mat(), params.rfParams[i]);
        }
    }
}


void DRFClassifier::saveModel(string filePath,
                              string modelName) const
{
    FileStorage fs(filePath, FileStorage::WRITE);
    fs << modelName << "{";
    fs << "layers_num" << params.layersNum;
    for (int i = 0; i < params.layersNum; ++i)
    {
        const CvRTrees & rf = randomForests[i];
        stringstream s;
        s << "random_forest_" << i;
        rf.write(*fs, s.str().c_str());
    }
    fs << "}";
}


void DRFClassifier::loadModel(string filePath,
                              string modelName)
{
    FileStorage fs(filePath, FileStorage::READ);
    FileNode fn = fs[modelName];
    CV_Assert(!fn.empty());
    fn["layers_num"] >> params.layersNum;
    params.rfParams.clear();
    randomForests.resize(params.layersNum);

    for (int i = 0; i < params.layersNum; ++i)
    {
        CvRTrees & rf = randomForests[i];
        stringstream s;
        s << "random_forest_" << i;
        rf.read(*fs, *(fn[s.str()]));
    }
}


void DRFClassifier::setParams(const DRFClassifierParams & p)
{
    params = p;
}


void DRFClassifier::read(const FileNode & fn)
{
    info()->read(this, fn);
    FileNode rfsParams = fn["random_forests"];\
    params.rfParams.resize(params.layersNum);
    int j = 0;
    for (FileNodeIterator i = rfsParams.begin();
         i != rfsParams.end() && j < params.layersNum;
         ++i, ++j)
    {
        (*i)["treeDepth"] >> params.rfParams[j].max_depth;
        (*i)["minSamplesInLeaf"] >> params.rfParams[j].min_sample_count;
        (*i)["useSurrogateSplits"] >> params.rfParams[j].use_surrogates;
        (*i)["treesNum"] >> params.rfParams[j].term_crit.max_iter;
        (*i)["activeFeaturesPerNode"] >> params.rfParams[j].nactive_vars;
    }
}


void DRFClassifier::write(FileStorage & fs) const
{
    CV_Assert(params.layersNum == params.rfParams.size());

    info()->write(this, fs);
    fs << "random_forests" << "[";
    for (int i = 0; i < params.layersNum; ++i)
    {
        fs << "{";
        fs << "treeDepth" << params.rfParams[i].max_depth;
        fs << "minSamplesInLeaf" << params.rfParams[i].min_sample_count;
        fs << "useSurrogateSplits" << params.rfParams[i].use_surrogates;
        fs << "treesNum" << params.rfParams[i].term_crit.max_iter;
        fs << "activeFeaturesPerNode" << params.rfParams[i].nactive_vars;
        fs << "}";
    }
    fs << "]";
}
