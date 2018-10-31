
#ifndef _GROUND_ESTIMATOR_H_
#define _GROUND_ESTIMATOR_H_

#include <opencv2/opencv.hpp>

#include "hough_trans.h"

class GroundEstimatorParams {
public:

  GroundEstimatorParams() {};

  ~GroundEstimatorParams() {};

  int v_disp_height;
  int v_disp_width;
  int u_disp_height;
  int u_disp_width;
  bool show_v_disp;
  int min_disp;
  int max_disp;

  HoughTransParams hough_param;

};

class GroundModel {

public:

  GroundModel() :
    slope_ddisp_dv(0.0), intercept_disp(0.0) {}

  ~GroundModel() {}

  GroundModel(const GroundModel &obj) :
    slope_ddisp_dv(obj.slope_ddisp_dv), 
    intercept_disp(obj.intercept_disp) {}

  GroundModel &operator=(const GroundModel &obj) {
    if (this != &obj) {
      this->slope_ddisp_dv = obj.slope_ddisp_dv;
      this->intercept_disp = obj.intercept_disp;
    }
    return *this;
  }    
  
public:

  double slope_ddisp_dv;
  double intercept_disp;
};

class GroundEstimator {

public:
  GroundEstimator();

  ~GroundEstimator();

  void initialize(GroundEstimatorParams &param);

  void estimate_ground(cv::Mat &dispU8, std::vector<GroundModel> &models);

  void get_v_disp_img(cv::Mat &v_disp_u8);

  void get_hough_vote_img(cv::Mat &hough_vote_u8);

  void get_line_overlaid_img(cv::Mat &rgb_img);

private:

  void generate_v_disp_img(cv::Mat &disp_u8, cv::Mat &v_disp);

  void convert_v_disp_s32_2_u8(cv::Mat &v_disp_s32, cv::Mat &v_disp_u8);

  void convert_pnt_pairs_2_GroundModel(const std::pair<cv::Point2f, cv::Point2f> &pnt_pair, GroundModel &model);

private:
  GroundEstimatorParams param;
  HoughTrans hough_trans;
  cv::Mat v_disp_s32, v_disp_u8, u_disp_s32, u_disp_u8;

};

#endif // _GROUND_ESTIMATOR_H_
