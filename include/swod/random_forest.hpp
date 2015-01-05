#pragma once

#include <map>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"


class RandomForest : public CvRTrees
{
public:
    int getNumOfLeaves(int treeIdx) const;
    void getLeavesMap(int treeIdx, std::map<const CvDTreeNode *, int> &leavesMap) const;
    void getLeavesIndices(const cv::Mat & sample, cv::Mat & leavesIndices) const;
    void predict(const cv::Mat & sample, const cv::Mat & missing, std::vector<float> & scores) const;
};
