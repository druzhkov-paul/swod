#include "test_utils.hpp"
#include "swod/swod.hpp"
#include <cstdio>

#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace cv;

namespace
{
    void saveLoadTest(const string & fileName)
    {
        ImageAnnotation ann;
        ann.sources[DataTypeTime("image", 0)] = "./some_image_path.jpg";
        ann.sources[DataTypeTime("another data source", -2)] = "./some_file_path.xxx";

        ann.bboxes.push_back(Rect(0, 1, 2, 3));
        ann.labels.push_back(0);
        EXPECT_ASSERTION_FAILURE(ann.save(fileName, "aaa"));
        EXPECT_ASSERTION_FAILURE(ann.addBBox(Rect(4, 5, 6, 7), 1, .0123f));
        remove(fileName.c_str());
        ann.scores.push_back(ImageAnnotation::GROUND_TRUTH);

        ann.addBBox(Rect(4, 5, 6, 7), 1, .0123f);

        ann.save(fileName, "test_annotation");

        ImageAnnotation ann2;
        EXPECT_ASSERTION_FAILURE(ann2.load(fileName, "annotation"));
        ann2.load(fileName, "test_annotation");

        remove(fileName.c_str());

        EXPECT_EQ(static_cast<size_t>(2), ann2.sources.size());
        EXPECT_NE(ann2.sources.end(), ann2.sources.find(DataTypeTime("image", 0)));
        EXPECT_NE(ann2.sources.end(), ann2.sources.find(DataTypeTime("another data source", -2)));
        EXPECT_EQ("./some_image_path.jpg", ann2.sources[DataTypeTime("image", 0)]);
        EXPECT_EQ("./some_file_path.xxx", ann2.sources[DataTypeTime("another data source", -2)]);

        EXPECT_EQ(static_cast<size_t>(2), ann2.bboxes.size());
        EXPECT_EQ(static_cast<size_t>(2), ann2.labels.size());
        EXPECT_EQ(static_cast<size_t>(2), ann2.scores.size());

        int idx = 0;
        if (ann2.labels[idx] != 0)
        {
            idx = 1;
        }
        EXPECT_EQ(Rect(0, 1, 2, 3), ann2.bboxes[idx]);
        EXPECT_EQ(0, ann2.labels[idx]);
        EXPECT_EQ(ImageAnnotation::GROUND_TRUTH, ann2.scores[idx]);

        idx = 1 - idx;
        EXPECT_EQ(Rect(4, 5, 6, 7), ann2.bboxes[idx]);
        EXPECT_EQ(1, ann2.labels[idx]);
        EXPECT_EQ(.0123f, ann2.scores[idx]);
    }
}


TEST(swod, annotationSaveLoadYML)
{
    char tempFileName[L_tmpnam];
    ASSERT_EQ(tempFileName, tmpnam(tempFileName));
    string fileName = string(tempFileName) + ".yml";

    saveLoadTest(fileName);
}


TEST(swod, annotationSaveLoadXML)
{
    char tempFileName[L_tmpnam];
    ASSERT_EQ(tempFileName, tmpnam(tempFileName));
    string fileName = string(tempFileName) + ".xml";

    saveLoadTest(fileName);
}


TEST(swod, annotationAddRemoveBorder)
{
    ImageAnnotation ann;
    ann.addBBox(Rect(200, 100, 32, 96), 0, 1.0f);
    ann.addBBox(Rect(0, 0, 30, 60), 1, 0.1f);
    ann.addBBoxesBorder(Size(64, 128), Size(16, 16));

    EXPECT_EQ(static_cast<size_t>(2), ann.bboxes.size());

    EXPECT_EQ(Rect(184, 84, 64, 128), ann.bboxes[0]);
    EXPECT_EQ(0, ann.labels[0]);
    EXPECT_EQ(1.0f, ann.scores[0]);

    EXPECT_EQ(Rect(-15, -10, 60, 80), ann.bboxes[1]);
    EXPECT_EQ(1, ann.labels[1]);
    EXPECT_EQ(0.1f, ann.scores[1]);

    ann.removeBBoxesBorder(Size(64, 128), Size(16, 16));

    EXPECT_EQ(static_cast<size_t>(2), ann.bboxes.size());

    EXPECT_EQ(Rect(200, 100, 32, 96), ann.bboxes[0]);
    EXPECT_EQ(0, ann.labels[0]);
    EXPECT_EQ(1.0f, ann.scores[0]);

    EXPECT_EQ(Rect(0, 0, 30, 60), ann.bboxes[1]);
    EXPECT_EQ(1, ann.labels[1]);
    EXPECT_EQ(0.1f, ann.scores[1]);
}


TEST(swod, annotationRandom)
{
    ImageAnnotation ann;
    ann.addBBox(Rect(0, 1, 20, 30), 2, 1.0f);

    ann.generateRandomAnnotation(Size(72, 96), Size(640, 480),
                                 50, -1, 10, 1, 0.2f);

    EXPECT_EQ(static_cast<size_t>(11), ann.bboxes.size());
    EXPECT_EQ(Rect(0, 1, 20, 30), ann.bboxes[0]);
    EXPECT_EQ(2, ann.labels[0]);
    EXPECT_EQ(1.0f, ann.scores[0]);

    for (int i = 1; i < 11; ++i)
    {
        EXPECT_EQ(1, ann.labels[i]);
        EXPECT_EQ(0.2f, ann.scores[i]);
    }
}
