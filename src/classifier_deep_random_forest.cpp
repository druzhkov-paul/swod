#include "swod/classifier_deep_random_forest.hpp"
#include <sstream>

using namespace cv;
using std::vector;
using std::string;
using std::stringstream;
using std::map;


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
        Mat newFeatureDescription;
        randomForests[i].getLeavesIndices(featureDescription, newFeatureDescription);
        featureDescription = newFeatureDescription;
    }

    const RandomForest & rf = randomForests[params.layersNum - 1];
    rf.predict(featureDescription, Mat(), weights);
    int label = 0;
    float maxScore = weights[0];
    for (size_t i = 1; i < weights.size(); ++i)
    {
        if (maxScore < weights[i])
        {
            maxScore = weights[i];
            label = i;
        }
    }
    //minMaxIdx(weights, 0, 0, 0, &label);

    return label;
}


namespace
{
    template<typename t>
    void inverseFrequencies(const Mat & m,
                     map<t, float> & freq)
    {
        freq.clear();
        for (size_t i = 0; i < m.total(); ++i)
        {
            t x = m.at<t>(i);
            if (freq.count(x))
            {
                ++freq[x];
            }
            else
            {
                freq[x] = 1.0f;
            }
        }
        for (auto i = freq.begin(); i != freq.end(); ++i)
        {
            i->second /= static_cast<float>(m.total());
        }
    }


    template<typename t>
    void inverseFrequencies(const Mat & m,
                     Mat & freq)
    {
        map<t, float> freqMap;
        inverseFrequencies<t>(m, freqMap);
        freq.create(1, freqMap.size(), CV_32F);
        int j = 0;
        for (auto i = freqMap.begin(); i != freqMap.end(); ++i, ++j)
        {
            freq.at<float>(j) = 1.0f / i->second;
        }
    }
}


void DRFClassifier::train(const Mat & dataset,
                          const Mat & responses)
{
    CV_Assert(0 < params.layersNum);
    CV_Assert(static_cast<size_t>(params.layersNum) == params.rfParams.size());

    if (params.equalizePriors[0])
    {
        Mat prior;
        inverseFrequencies<float>(responses, prior);
        params.priors[0] = prior;
        params.rfParams[0].priors = reinterpret_cast<const float*>(prior.data);
    }

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

            if (params.equalizePriors[i])
            {
                Mat prior;
                inverseFrequencies<float>(responses, prior);
                params.priors[i] = prior;
                params.rfParams[i].priors = reinterpret_cast<const float*>(prior.data);
            }

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
    FileNode rfsParams = fn["random_forests"];
    params.rfParams.resize(params.layersNum);
    params.priors.resize(params.layersNum);
    params.equalizePriors.assign(params.layersNum, false);
    int j = 0;
    for (FileNodeIterator i = rfsParams.begin();
         i != rfsParams.end() && j < params.layersNum;
         ++i, ++j)
    {
        (*i)["treeDepth"] >> params.rfParams[j].max_depth;
        (*i)["minSamplesInLeaf"] >> params.rfParams[j].min_sample_count;
        (*i)["useSurrogateSplits"] >> params.rfParams[j].use_surrogates;
        (*i)["treesNum"] >> params.rfParams[j].term_crit.max_iter;
        params.rfParams[j].term_crit.type = CV_TERMCRIT_ITER;
        params.rfParams[j].term_crit.epsilon = 0.0;
        (*i)["activeFeaturesPerNode"] >> params.rfParams[j].nactive_vars;
        FileNode priorNode = (*i)["prior"];
        if (priorNode.isString())
        {
            string s;
            priorNode >> s;
            if (s == "auto")
            {
                params.equalizePriors[j] = true;
                params.priors[j] = Mat();
                params.rfParams[j].priors = 0;
            }
        }
        else
        {
            priorNode >> params.priors[j];
            params.rfParams[j].priors = reinterpret_cast<const float*>(params.priors[j].data);
        }
    }
}


void DRFClassifier::write(FileStorage & fs) const
{
    CV_Assert(static_cast<size_t>(params.layersNum) == params.rfParams.size());

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
        if (params.equalizePriors[i])
        {
            fs << "prior" << "auto";
        }
        else if (params.priors[i].empty())
        {
            fs << "prior" << "none";
        }
        else
        {
            fs << "prior" << params.priors[i];
        }
        fs << "}";
    }
    fs << "]";
}
