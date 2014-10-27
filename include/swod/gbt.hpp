#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include <vector>


class GBTrees : public CvGBTrees
{
public:
    enum {SQUARED_LOSS=0, ABSOLUTE_LOSS, HUBER_LOSS=3, DEVIANCE_LOSS, DEVIANCE_2_LOSS};

    GBTrees();
    GBTrees( const CvMat* trainData, int tflag,
             const CvMat* responses, const CvMat* varIdx=0,
             const CvMat* sampleIdx=0, const CvMat* varType=0,
             const CvMat* missingDataMask=0,
             CvGBTreesParams params=CvGBTreesParams() );

    virtual ~GBTrees();
    
    virtual bool train( const CvMat* trainData, int tflag,
             const CvMat* responses, const CvMat* varIdx=0,
             const CvMat* sampleIdx=0, const CvMat* varType=0,
             const CvMat* missingDataMask=0,
             CvGBTreesParams params=CvGBTreesParams(),
             bool update=false );
    
    virtual bool train( CvMLData* data,
             CvGBTreesParams params=CvGBTreesParams(),
             bool update=false );

    virtual float predict_serial( const CvMat* sample, const CvMat* missing=0,
            CvMat* weakResponses=0, CvSlice slice = CV_WHOLE_SEQ,
            int k=-1 ) const;

    virtual float predict_serial( const CvMat* sample, const CvMat* missing=0,
            std::vector<float> * probs=0, CvSlice slice = CV_WHOLE_SEQ) const;
        
    virtual float predict( const CvMat* sample, const CvMat* missing=0,
            std::vector<float> * probs=0, CvSlice slice = CV_WHOLE_SEQ,
            int k=-1 ) const;

    virtual void clear();

    virtual float calc_error( CvMLData* _data, int type,
            std::vector<float> *resp = 0, CvSlice _slice=CV_WHOLE_SEQ );

    virtual void write( CvFileStorage* fs, const char* name ) const;

    virtual void read( CvFileStorage* fs, CvFileNode* node );

    int getEnsemblesNum() const;
    CvSeq ** getWeak() {return weak;}
    
    GBTrees( const cv::Mat& trainData, int tflag,
              const cv::Mat& responses, const cv::Mat& varIdx=cv::Mat(),
              const cv::Mat& sampleIdx=cv::Mat(), const cv::Mat& varType=cv::Mat(),
              const cv::Mat& missingDataMask=cv::Mat(),
              CvGBTreesParams params=CvGBTreesParams() );
    
    virtual bool train( const cv::Mat& trainData, int tflag,
                       const cv::Mat& responses, const cv::Mat& varIdx=cv::Mat(),
                       const cv::Mat& sampleIdx=cv::Mat(), const cv::Mat& varType=cv::Mat(),
                       const cv::Mat& missingDataMask=cv::Mat(),
                       CvGBTreesParams params=CvGBTreesParams(),
                       bool update=false );

    virtual float predict( const cv::Mat& sample, const cv::Mat& missing=cv::Mat(),
                           std::vector<float> * probs = 0,
                           const cv::Range& slice = cv::Range::all()) const;

    virtual float predict_serial( const cv::Mat& sample, const cv::Mat& missing=cv::Mat(),
                           std::vector<float> * probs = 0,
                           const cv::Range& slice = cv::Range::all()) const;
    
protected:

    virtual void find_gradient( const int k = 0);

    virtual void change_values(CvDTree* tree, const int k = 0);

    virtual float find_optimal_value( const CvMat* _Idx );

    virtual void do_subsample();

    void leaves_get( CvDTreeNode** leaves, int& count, CvDTreeNode* node );
    
    CvDTreeNode** GetLeaves( const CvDTree* dtree, int& len );

    virtual bool problem_type() const;

    virtual void write_params( CvFileStorage* fs ) const;

    virtual void read_params( CvFileStorage* fs, CvFileNode* fnode );
    int get_len(const CvMat* mat) const;

    float getInitValue() const;
    float getPrediction(float * sum) const;
    void getProb(std::vector<float> * probs, float * sum) const;
    
    CvDTreeTrainData* data;
    CvGBTreesParams params;

    CvSeq** weak;
    CvMat* orig_response;
    CvMat* sum_response;
    CvMat* sum_response_tmp;
    CvMat* sample_idx;
    CvMat* subsample_train;
    CvMat* subsample_test;
    CvMat* missing;
    CvMat* class_labels;

    cv::RNG* rng;

    int class_count;
    float delta;
    float base_value;
};
