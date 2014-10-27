#include "swod/nms_meanshift.hpp"
#include "algorithm"
#include "swod/timing.hpp"

using namespace cv;
using namespace std;


MeanshiftNMS::MeanshiftNMS()
    : threshold(0.0f),
      baseObjectSize(32, 96),
      kernelDiag(64.0f, 256.0f, std::log(1.3f) * std::log(1.3f))
{}


MeanshiftNMS::~MeanshiftNMS()
{}

namespace
{
    class Meanshift
    {
    public:
        Meanshift(vector<Point3f> & pos_,
                  const vector<float> & weight_,
                  const Point3f & kernel_,
                  float eps,
                  size_t maxIter = 20);

        void getModes(vector<Point3f> & modesV,
                      vector<float> & resWeightsV,
                      float eps,
                      float threshold = 0.0f) const;

    protected:
        vector<Point3f> & pos;
        const vector<float> & weight;
        vector<float> alpha;
        vector<Point3f> hInv;
        vector<Point3f> mode;
        vector<float> resWeight;
        const Point3f kernel;

        float dist(const Point3f & p,
                   const Point3f & q) const;
        Point3f shiftPoint(const Point3f & q,
                           float & sumW) const;
    };



    Meanshift::Meanshift(vector<Point3f> & pos_,
                         const vector<float> & weight_,
                         const Point3f & kernel_,
                         float eps,
                         size_t maxIter)
        : pos(pos_),
          weight(weight_),
          kernel(kernel_)
    {
        CV_Assert(pos.size() == weight.size());
        size_t n = pos.size();
        hInv.resize(n);
        alpha.resize(n);
        mode.resize(n);
        resWeight.resize(n);

        for (size_t i = 0; i < n; ++i)
        {
            float expScale = pos[i].z * pos[i].z;
            hInv[i] = Point3f(1.0f / (expScale * kernel.x),
                              1.0f / (expScale * kernel.y),
                              1.0f / kernel.z);
            alpha[i] = weight[i] / std::sqrt(expScale * kernel.x +
                                             expScale * kernel.y +
                                             kernel.z);
        }

        for (size_t i = 0; i < n; ++i)
        {
            pos[i].z = std::log(pos[i].z);
        }

        for (size_t i = 0; i < n; ++i)
        {
            Point3f p1 = pos[i];
            float w = 0.0f;
            for (size_t j = 0; j < maxIter; ++j)
            {
                Point3f p2 = p1;
                p1 = shiftPoint(p2, w);
                if (dist(p1, p2) <= eps)
                {
                    break;
                }
            }
            mode[i] = p1;
            resWeight[i] = w;
        }
    }


    void Meanshift::getModes(vector<Point3f> & modesV,
                             vector<float> & resWeightsV,
                             float eps,
                             float threshold) const
    {
        modesV.clear();
        resWeightsV.clear();
        for (size_t i = 0; i < mode.size(); ++i)
        {
            if (resWeight[i] < threshold)
            {
                continue;
            }
            bool isFound = false;
            for (size_t j = 0; j < modesV.size(); ++j)
            {
                if (dist(mode[i], modesV[j]) < eps)
                {
                    isFound = true;
                    break;
                }
            }
            if (!isFound)
            {
                modesV.push_back(mode[i]);
                resWeightsV.push_back(resWeight[i]);
            }
        }
    }


    float Meanshift::dist(const Point3f & p,
                          const Point3f & q) const
    {
        Point3f krn = kernel;
        float expScale = std::exp(2.0f * q.z);
        krn.x *= expScale;
        krn.y *= expScale;

        Point3f d = q - p;
        return d.x * d.x / krn.x +
                d.y * d.y / krn.y +
                d.z * d.z / krn.z;
    }


    Point3f Meanshift::shiftPoint(const Point3f & q,
                                  float & sumW) const
    {
        Point3f a(0.0f, 0.0f, 0.0f);
        Point3f b(0.0f, 0.0f, 0.0f);
        sumW = 0.0f;
        for (size_t i = 0; i < pos.size(); ++i)
        {
            const Point3f & p = pos[i];
            Point3f d = q - p;
            const Point3f & h = hInv[i];
            float w = alpha[i] * std::exp(-0.5f * (d.x * d.x * h.x +
                                                   d.y * d.y * h.y +
                                                   d.z * d.z * h.z));
            sumW += w;

            float x = w * h.x;
            float y = w * h.y;
            float z = w * h.z;
            a += Point3f(x, y, z);
            b += Point3f(x * p.x,
                         y * p.y,
                         z * p.z);
        }
        return Point3f(b.x / a.x,
                       b.y / a.y,
                       b.z / a.z);
    }

}


void MeanshiftNMS::operator() (std::vector<cv::Rect> & bboxes,
                               std::vector<float> & scores) const
{
    vector<Point3f> pos(bboxes.size());
    float objectHeight = static_cast<float>(baseObjectSize.height);
    for (size_t i = 0; i < bboxes.size(); ++i)
    {
        const Rect & r = bboxes[i];
        Point3f & p = pos[i];
        p.x = r.x + r.width * 0.5f;
        p.y = r.y + r.height * 0.5f;
        p.z = static_cast<float>(r.height) / objectHeight;
    }

    Meanshift ms(pos, scores, kernelDiag, 1e-5f, 100);

    pos.clear();
    scores.clear();
    ms.getModes(pos, scores, 1.0f, threshold);

    bboxes.resize(pos.size());
    for (size_t i = 0; i < pos.size(); ++i)
    {
        float scale = std::exp(pos[i].z);
        Rect & r = bboxes[i];
        r.width = static_cast<int>(scale * baseObjectSize.width);
        r.height = static_cast<int>(scale * baseObjectSize.height);
        r.x = pos[i].x - r.width / 2;
        r.y = pos[i].y - r.height / 2;
    }
}
