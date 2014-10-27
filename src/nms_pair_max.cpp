#include "swod/nms_pair_max.hpp"
#include "algorithm"

using namespace cv;
using namespace std;


PairMaxNMS::PairMaxNMS()
    : threshold(0.5f)
{}


PairMaxNMS::~PairMaxNMS()
{}


namespace
{
    inline int intersectionArea(Rect r1, Rect r2)
    {
        return (r1 & r2).area();
    }

    inline int unionArea(Rect r1, Rect r2)
    {
        return r1.area() + r2.area() - (r1 & r2).area();
    }

    inline float coverageCoeff(Rect r1, Rect r2)
    {
        return intersectionArea(r1, r2) / static_cast<float>(unionArea(r1, r2));
    }

    inline float coverageCoeff1(Rect r1, Rect r2)
    {
        return intersectionArea(r1, r2) / static_cast<float>(max(r1.area(), r2.area()));
    }
}


void PairMaxNMS::operator()(vector<Rect> & bboxes,
                            vector<float> & scores) const
{
    size_t numRects = bboxes.size();
    vector<Rect> rectsSelected;
    vector<float> scoresSelected;
    vector<bool> isSuppressed(numRects, false);
    vector<int> idx(numRects);
    for (size_t i = 0; i < numRects; ++i)
    {
        idx[i] = i;
    }

    sort(idx.begin(), idx.end(), [&scores] (int i, int j)
    {
         return (scores[i] > scores[j]);
    });

    for (vector<int>::iterator idx1 = idx.begin(); idx1 != idx.end(); ++idx1)
    {
        int i = (*idx1);
        if (!isSuppressed[i])
        {
            rectsSelected.push_back(bboxes[i]);
            float currentScore = scores[i];
            for (vector<int>::iterator idx2 = idx1 + 1; idx2 != idx.end(); ++idx2)
            {
                int j = (*idx2);
                bool suppresedByCurrentBBox =
                        (threshold < coverageCoeff1(bboxes[i], bboxes[j]));
                isSuppressed[j] = isSuppressed[j] || suppresedByCurrentBBox;
                if (suppresedByCurrentBBox)
                {
                    currentScore += scores[j];
                }
            }
            scoresSelected.push_back(currentScore);
        }
    }
    bboxes.resize(rectsSelected.size());
    copy(rectsSelected.begin(), rectsSelected.end(), bboxes.begin());
    rectsSelected.clear();
    scores.resize(scoresSelected.size());
    copy(scoresSelected.begin(), scoresSelected.end(), scores.begin());
    scoresSelected.clear();
}
