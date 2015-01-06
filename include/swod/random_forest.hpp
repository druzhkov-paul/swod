#pragma once

#include <map>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"


class RandomForest : public CvRTrees
{
public:
    //int getNumOfLeaves(int treeIdx) const;
    void getLeavesMap(int treeIdx, std::map<CvDTreeNode *, int> & leavesMap) const;
    void getLeavesIndices(const cv::Mat & sample, cv::Mat & leavesIndices) const;
    void predict(const cv::Mat & sample,
                 const cv::Mat & missing,
                 std::vector<float> & scores) const;
    bool train(const cv::Mat & trainData,
               int tflag,
               const cv::Mat & responses,
               const cv::Mat & varIdx = cv::Mat(),
               const cv::Mat & sampleIdx = cv::Mat(),
               const cv::Mat & varType = cv::Mat(),
               const cv::Mat & missingDataMask = cv::Mat(),
               CvRTParams params = CvRTParams());
};
