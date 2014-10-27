#pragma once

#include "swod/data_provider.hpp"
#include "opencv2/core/core.hpp"
#include <vector>
#include <string>
#include <map>


struct ImageAnnotation
{
    static const float GROUND_TRUTH;

    ImageAnnotation(int n = 5000);
    virtual ~ImageAnnotation();

    virtual void clear();
    virtual void read(const cv::FileNode & fn);
    virtual void load(const std::string & fileName,
                      const std::string & name);
    virtual void write(cv::FileStorage & fs) const;
    virtual void save(const std::string & fileName,
                      const std::string & name) const;
    virtual void generateRandomAnnotation(const cv::Size & detectionWindowSize,
                                          cv::Size imageSize,
                                          int minObjectHeight,
                                          int maxObjectHeight,
                                          int samples = 50,
                                          int label = 0,
                                          float score = GROUND_TRUTH);
    virtual void removeBBoxesBorder(const cv::Size & detectionWindowSize,
                                    const cv::Size & detectionWindowBorder);
    virtual void addBBoxesBorder(const cv::Size & detectionWindowSize,
                                 const cv::Size & detectionWindowBorder);
    virtual void addBBox(const cv::Rect & bbox, int label, float score);

	std::map<DataTypeTime, std::string> sources;
    std::vector<cv::Rect> bboxes;
    std::vector<int> labels;
    std::vector<float> scores;
};

void readDatasetAnnotation(const cv::FileNode & fn,
                           std::vector<ImageAnnotation> & imageAnnotations);
void writeDatasetAnnotation(cv::FileStorage & fs,
                            std::string name,
                            const std::vector<ImageAnnotation> & imageAnnotations);
void loadDatasetAnnotation(const std::string & fileName,
                           const std::string & name,
                           std::vector<ImageAnnotation> & imageAnnotations);
void saveDatasetAnnotation(const std::string & fileName,
                           const std::string & name,
                           const std::vector<ImageAnnotation> & imageAnnotations);

void removeBBoxBorder(cv::Rect & bbox,
                      const cv::Size & detectionWindowSize,
                      const cv::Size & detectionWindowBorder);
