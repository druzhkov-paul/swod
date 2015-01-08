#include "swod/random_forest.hpp"

using std::map;
using std::vector;
using namespace cv;

namespace
{
    void getLeaves(map<CvDTreeNode*, int> & leaves,
                   CvDTreeNode * node)
    {
        if (node->left != NULL)
        {
            getLeaves(leaves, node->left);
        }
        if (node->right != NULL)
        {
            getLeaves(leaves, node->right);
        }
        if ((node->left == NULL) && (node->right == NULL))
        {
            leaves[node] = leaves.size();
        }
    }
}

/*
int RandomForest::getNumOfLeaves(int treeIdx) const
{
    CV_Assert(0 <= treeIdx);
    CV_Assert(treeIdx < ntrees);

    CvForestTree * tree = trees[treeIdx];
    map<CvDTreeNode*, int> leaves;
    getLeaves(leaves, const_cast<CvDTreeNode*>(tree->get_root()));
    return leaves.size();
}
*/

void RandomForest::getLeavesMap(int treeIdx,
                                map<CvDTreeNode*, int> & leavesMap) const
{
    CV_Assert(0 <= treeIdx);
    CV_Assert(treeIdx < ntrees);

    CvForestTree * tree = trees[treeIdx];
    getLeaves(leavesMap, const_cast<CvDTreeNode*>(tree->get_root()));
}

/*
void RandomForest::getLeavesIndices(const Mat & sample,
                                    Mat & leavesIndices) const
{
    leavesIndices.create(1, ntrees, CV_32F);
    for (int i = 0; i < ntrees; ++i)
    {
        map<CvDTreeNode*, int> leaves;
        getLeavesMap(i, leaves);
        CvDTreeNode * predictedLeaf = trees[i]->predict(sample);
        leavesIndices.at<float>(i) = static_cast<float>(leaves[predictedLeaf]);
    }
}
*/

void RandomForest::getLeavesIndices(const Mat & sample,
                                    Mat & leavesIndices) const
{
    leavesIndices.create(1, ntrees, CV_32F);
    for (int i = 0; i < ntrees; ++i)
    {
        CvDTreeNode * predictedLeaf = trees[i]->predict(sample);
        leavesIndices.at<float>(i) = predictedLeaf->value;
    }
}


void RandomForest::predict(const Mat & sample,
                           const Mat & missing,
                           vector<float> & scores) const
{
    if (1 < nclasses)
    {
        scores.assign(nclasses, 0.0f);
        for (int i = 0; i < ntrees; ++i)
        {
            CvDTreeNode * predictedNode = trees[i]->predict(sample, missing);
            int classIdx = predictedNode->class_idx;
            CV_Assert(0 <= classIdx && classIdx < nclasses);
            ++scores[classIdx];
        }

        for (size_t i = 0; i < scores.size(); ++i)
        {
            scores[i] /= static_cast<float>(ntrees);
        }
    }
    else
    {
        CV_Error(CV_StsBadArg, "This function works for classification problems only.");
    }
}


bool RandomForest::train(const Mat & trainData,
                         int tflag,
                         const Mat & responses,
                         const Mat & varIdx,
                         const Mat & sampleIdx,
                         const Mat & varType,
                         const Mat & missingDataMask,
                         CvRTParams params)
{
    bool res = CvRTrees::train(trainData, tflag, responses, varIdx,
                               sampleIdx, varType, missingDataMask, params);

    if (1 < nclasses)
    {
        for (int i = 0; i < ntrees; ++i)
        {
            map<CvDTreeNode*, int> leaves;
            getLeavesMap(i, leaves);
            for (auto j = leaves.begin(); j != leaves.end(); ++j)
            {
                j->first->value = j->second;
            }
        }
    }

    return res;
}


float RandomForest::getOOBError() const
{
    return static_cast<float>(oob_error);
}
