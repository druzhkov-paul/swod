#include "swod/annotation.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace cv;
using namespace std;

const float ImageAnnotation::GROUND_TRUTH = numeric_limits<float>::infinity();

namespace
{
	DataTypeTime IMAGE_0("image", 0);
}


ImageAnnotation::ImageAnnotation(int n)
{
    bboxes.reserve(n);
    labels.reserve(n);
    scores.reserve(n);
}


ImageAnnotation::~ImageAnnotation()
{
    clear();
}


void ImageAnnotation::clear()
{
    sources.clear();
    bboxes.clear();
    labels.clear();
    scores.clear();
}


void ImageAnnotation::write(FileStorage & fs) const
{
    CV_Assert(fs.isOpened());
    fs << "{";
    fs << "sources" << "[";
    for (auto i = sources.begin(); i != sources.end(); ++i)
    {
        fs << "{:" << "type" << "\"" + i->first.first + "\""
		   << "time" << i->first.second
           << "path" << "\"" + i->second + "\"" << "}";
    }
    fs << "]";

    fs << "bounding_boxes" << "[";
    CV_Assert(bboxes.size() == labels.size());
    CV_Assert(bboxes.size() == scores.size());
    for (size_t i = 0; i < bboxes.size(); ++i)
    {
        fs << "{:";
        fs << "bb" << bboxes[i];
        fs << "label" << labels[i];
        fs << "score" << scores[i];
        fs << "}";
    }
    fs << "]";
    fs << "}";
}


void ImageAnnotation::read(const FileNode & fn)
{
    CV_Assert(!fn.empty());
    clear();

    FileNode srcs = fn["sources"];
    CV_Assert(!srcs.empty());
    for (FileNodeIterator i = srcs.begin(); i != srcs.end(); ++i)
    {
        string type, path;
		int time;
        (*i)["type"] >> type;
		(*i)["time"] >> time;
        (*i)["path"] >> path;
        sources[DataTypeTime(type, time)] = path;
    }

    FileNode bbs = fn["bounding_boxes"];
    CV_Assert(!bbs.empty());
    for (FileNodeIterator i = bbs.begin(); i != bbs.end(); ++i)
    {
        Rect bb(0, 0, 0, 0);
        float score = 0.0f;
        int label = 0;

        (*i)["bb"] >> bb;
        bboxes.push_back(bb);
        (*i)["score"] >> score;
        scores.push_back(score);
        (*i)["label"] >> label;
        labels.push_back(label);
    }
}


void ImageAnnotation::load(const string & fileName,
                           const string & name)
{
    FileStorage fs(fileName, FileStorage::READ);
    CV_Assert(fs.isOpened());
    FileNode fn = fs[name];
    read(fn);
}


void ImageAnnotation::save(const string & fileName,
                           const string & name) const
{
    FileStorage fs(fileName, FileStorage::WRITE);
    CV_Assert(fs.isOpened());
    fs << name;
    write(fs);
}


void ImageAnnotation::generateRandomAnnotation(const cv::Size & detectionWindowSize,
                                               cv::Size imageSize,
                                               int minObjectHeight,
                                               int maxObjectHeight,
                                               int samples,
                                               int label,
                                               float score)
{
    // if image size is unknown read it and determine the size
    if (imageSize == Size())
    {
		CV_Assert(sources.count(IMAGE_0));
        string imageName = sources[IMAGE_0];
        Mat image = imread(imageName);
        imageSize = image.size();
    }

	if (maxObjectHeight < 0)
	{
		maxObjectHeight = imageSize.height;
	}
	minObjectHeight = min(minObjectHeight, imageSize.height);
	maxObjectHeight = min(maxObjectHeight, imageSize.height);
	// FIXME: check that corresponding object width is also valid
	CV_Assert(minObjectHeight <= maxObjectHeight);
	CV_Assert(0 < minObjectHeight && 0 < maxObjectHeight);

    // draw random samples
    RNG & rng = theRNG();
    for (int i = 0; i < samples; ++i)
    {
		int objectHeight = rng(maxObjectHeight - minObjectHeight) + minObjectHeight;
		int objectWidth = static_cast<int>(static_cast<float>(objectHeight) / detectionWindowSize.height * detectionWindowSize.width);
        int x = rng(imageSize.width - objectWidth);
        int y = rng(imageSize.height - objectHeight);
		Rect r(Point(x, y), Size(objectWidth, objectHeight));
        bboxes.push_back(r);
        labels.push_back(label);
        scores.push_back(score);
    }
}

void ImageAnnotation::removeBBoxesBorder(const Size & detectionWindowSize,
                                         const Size & detectionWindowBorder)
{
    float borderFractionWidth = static_cast<float>(detectionWindowBorder.width) /
                                detectionWindowSize.width;
    float borderFractionHeight = static_cast<float>(detectionWindowBorder.height) /
                                 detectionWindowSize.height;
    for (size_t i = 0; i < bboxes.size(); ++i)
    {
        Rect & r = bboxes[i];
        int borderX = static_cast<int>(borderFractionWidth * r.width);
        int borderY = static_cast<int>(borderFractionHeight * r.height);
        r += Point(borderX, borderY);
        r -= Size(2 * borderX, 2 * borderY);
    }
}


void ImageAnnotation::addBBoxesBorder(const Size & detectionWindowSize,
                                      const Size & detectionWindowBorder)
{
    float borderFractionWidth = static_cast<float>(detectionWindowBorder.width) /
                                (detectionWindowSize.width - 2 * detectionWindowBorder.width);
    float borderFractionHeight = static_cast<float>(detectionWindowBorder.height) /
                                 (detectionWindowSize.height - 2 * detectionWindowBorder.height);
    for (size_t i = 0; i < bboxes.size(); ++i)
    {
        Rect & r = bboxes[i];
		int borderX = static_cast<int>(borderFractionWidth * r.width);
        int borderY = static_cast<int>(borderFractionHeight * r.height);
        r -= Point(borderX, borderY);
        r += Size(2 * borderX, 2 * borderY);
    }
}


void ImageAnnotation::addBBox(const Rect & bbox, int label, float score)
{
    CV_Assert(bboxes.size() == labels.size());
    CV_Assert(bboxes.size() == scores.size());
    bboxes.push_back(bbox);
    labels.push_back(label);
    scores.push_back(score);
}


void readDatasetAnnotation(const FileNode & fn,
                           vector<ImageAnnotation> & imageAnnotations)
{
    CV_Assert(!fn.empty());
    imageAnnotations.clear();
    for (FileNodeIterator i = fn.begin(); i != fn.end(); ++i)
    {
        ImageAnnotation imAnn;
        imAnn.read(*i);
        imageAnnotations.push_back(imAnn);
    }
}


void writeDatasetAnnotation(FileStorage & fs,
                            string name,
                            const vector<ImageAnnotation> & imageAnnotations)
{
    CV_Assert(fs.isOpened());
    fs << name << "[";
    for (size_t i = 0; i < imageAnnotations.size(); ++i)
    {
        imageAnnotations[i].write(fs);
    }
    fs << "]";
}


void loadDatasetAnnotation(const string & fileName,
                           const string & name,
                           vector<ImageAnnotation> & imageAnnotations)
{
    FileStorage fs(fileName, FileStorage::READ);
    CV_Assert(fs.isOpened());
    FileNode fn = fs[name];
    readDatasetAnnotation(fn, imageAnnotations);
}


void saveDatasetAnnotation(const string & fileName,
                           const string & name,
                           const vector<ImageAnnotation> & imageAnnotations)
{
    FileStorage fs(fileName, FileStorage::WRITE);
    CV_Assert(fs.isOpened());
    writeDatasetAnnotation(fs, name, imageAnnotations);
}


void removeBBoxBorder(Rect & r,
                      const Size & detectionWindowSize,
                      const Size & detectionWindowBorder)
{
    float borderFractionWidth = static_cast<float>(detectionWindowBorder.width) /
                                detectionWindowSize.width;
    float borderFractionHeight = static_cast<float>(detectionWindowBorder.height) /
                                 detectionWindowSize.height;
    int borderX = static_cast<int>(borderFractionWidth * r.width);
    int borderY = static_cast<int>(borderFractionHeight * r.height);
    r += Point(borderX, borderY);
    r -= Size(2 * borderX, 2 * borderY);
}
