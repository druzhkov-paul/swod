#include "swod/random_forest.hpp"

using std::map;
using namespace cv;

namespace
{
    void getLeaves(map<const CvDTreeNode*, int> & leaves,
                   const CvDTreeNode * node)
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


int RandomForest::getNumOfLeaves(int treeIdx) const
{
    CV_Assert(0 <= treeIdx);
    CV_Assert(treeIdx < ntrees);

    CvForestTree * tree = trees[treeIdx];
    map<const CvDTreeNode*, int> leaves;
    getLeaves(leaves, tree->get_root());
    return leaves.size();
}


void RandomForest::getLeavesMap(int treeIdx,
                                map<const CvDTreeNode*, int> & leavesMap) const
{
    CV_Assert(0 <= treeIdx);
    CV_Assert(treeIdx < ntrees);

    CvForestTree * tree = trees[treeIdx];
    getLeaves(leavesMap, tree->get_root());
}


void RandomForest::getLeavesIndices(const Mat & sample,
                                    Mat & leavesIndices) const
{
    leavesIndices.create(1, ntrees, CV_32F);
    for (int i = 0; i < ntrees; ++i)
    {
        map<const CvDTreeNode*, int> leaves;
        getLeavesMap(i, leaves);
        CvDTreeNode * predictedLeaf = trees[i]->predict(sample);
        leavesIndices.at<float>(i) = static_cast<float>(leaves[predictedLeaf]);
    }
}
